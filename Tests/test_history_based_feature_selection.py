import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, SelectKBest, f_classif
from sklearn.datasets import make_classification, fetch_openml
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import math
import os
import sys
from time import process_time

sys.path.insert(0, "..")
from history_based_feature_selection import test_all_features, feature_selection_history


def generate_data(n_rows, n_cols, num_relevant_cols, frac_pos, target_type):
    def gen_target_col1(df):
        sum = np.array([0]*len(df))
        for col_idx, col_name in enumerate(df.columns[:num_relevant_cols]):
            sum = sum + ((len(df.columns) - col_idx) * df[col_name])
        ret = [x < np.quantile(sum, frac_pos) for x in sum]
        return ret

    def gen_target_col2(df):
        return (df[0] > 0.8) | (df[1] > 0.75) | (df[2] > 0.70) | (df[3] > 0.65) | (df[4] > 0.60) | (df[5] > 0.55)

    def gen_target_col3(df):
        product = np.array(1)*len(df)
        for col_idx, col_name in enumerate(df.columns[:num_relevant_cols]):
            product = product * df[col_idx].apply(lambda x: math.pow(x, col_idx))
        ret = [x < np.quantile(product, frac_pos) for x in product]
        return ret

    x_train = pd.DataFrame({x: np.random.rand(n_rows) for x in range(n_cols)})
    x_val = pd.DataFrame({x: np.random.rand(n_rows) for x in range(n_cols)})

    if target_type == 1:
        y_train = gen_target_col1(x_train)
        y_val = gen_target_col1(x_val)
    elif target_type == 2:
        y_train = gen_target_col2(x_train)
        y_val = gen_target_col2(x_val)
    elif target_type == 3:
        y_train = gen_target_col3(x_train)
        y_val = gen_target_col3(x_val)

    x_train.columns = ['F' + str(x) for x in x_train.columns]
    x_val.columns = ['F' + str(x) for x in x_val.columns]

    return x_train, y_train, x_val, y_val


def generate_make_classification(n_samples, n_features, n_informative, n_redundant, n_clusters_per_class):
    x, y = make_classification(n_samples=n_samples * 2, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class)
    x_train = pd.DataFrame(x[n_samples:], columns=list(range(n_features)))
    y_train = y[10_000:]
    x_val = pd.DataFrame(x[:n_samples], columns=list(range(n_features)))
    y_val = y[:10_000]
    return x_train, y_train, x_val, y_val


def generate_make_openml(filename):
    data = fetch_openml(filename, version=1, parser='auto')
    x = pd.DataFrame(data.data)
    y = data.target

    # If y is numeric, convert to a categorical target
    if pd.Series(y).astype(str).str.isnumeric().all():
        y = np.where(pd.Series(y).astype(float) > pd.Series(y).astype(float).median(), 1, 0)

    # If this is a multi-class classification problem, convert to binary classification. This makes the analysis
    # simpler and allows us to use target-encoding, which is more straightforward for feature selection than one-hot.
    elif pd.Series(y).nunique() > 2:
        y = np.where(y == pd.Series(y).mode()[0], 1, 0)

    # Shuffle to ensure the target classes are distributed in both the train and validation sets.
    x = x.sample(n=len(x))
    y = y[x.index]

    n_samples = len(x) // 2
    x_train = pd.DataFrame(x[:n_samples])
    y_train = y[:n_samples]
    x_val = pd.DataFrame(x[n_samples:])
    y_val = y[n_samples:]
    return x_train, y_train, x_val, y_val


def test_dataset_openml(filename):
    x_train, y_train, x_val, y_val = generate_make_openml(filename)
    if (pd.Series(y_train).nunique() == 1) or (pd.Series(y_val).nunique() == 1):
        return

    n_feats = len(x_train.columns)
    max_features_arr = [int(n_feats * 0.75), int(n_feats * 0.5), int(n_feats * 0.25)]
    max_features_arr = [x for x in max_features_arr if x > 0]

    for max_features in max_features_arr:
        res = test_dataset(filename, "Decision Tree", model_dt, x_train, y_train, x_val, y_val, max_features)
        results_arr.append(res)

        res = test_dataset(filename, "Catboost", model_catboost, x_train, y_train, x_val, y_val, max_features)
        results_arr.append(res)


def test_dataset(data_description, model_description, model, x_train, y_train, x_val, y_val, max_features):
    print()
    print('....................................................................................')
    print(f"Data: {data_description} ({len(x_train) + len(x_val)} rows, {len(x_train.columns)} columns)")
    print(f"Model: {model_description}")
    print(f"Max Features: {max_features}")
    print('....................................................................................')

    # Remove features that are all null
    drop_cols = []
    for col_name in x_train.columns:
        if x_train[col_name].count() == 0:
            drop_cols.append(col_name)
    x_train = x_train.drop(columns=drop_cols)
    x_val = x_val.drop(columns=drop_cols)

    # Fill in the missing values. Catboost can handle missing values fairly well, but not in all cases, for example
    # nan values in categorical columns.
    for col_name in x_train.columns:
        if str(x_train[col_name].dtype) == 'category':
            x_train[col_name] = x_train[col_name].fillna(x_train[col_name].cat.categories[0])
            x_val[col_name] = x_val[col_name].fillna(x_train[col_name].cat.categories[0])
        elif x_train[col_name].dtype in [np.float64, np.int64]:
            x_train[col_name] = x_train[col_name].fillna(x_train[col_name].median())
            x_val[col_name] = x_val[col_name].fillna(x_train[col_name].median())
        else:
            x_train[col_name] = x_train[col_name].fillna(x_train[col_name].mode())
            x_val[col_name] = x_val[col_name].fillna(x_train[col_name].mode())

    # Create copies of the x data that can be used with decision trees, random forests and so on. These have the
    # null values filled and the categorical columns encoded.
    use_cleaned = False
    if "Tree" in model_description:
        use_cleaned = True
    x_train_cleaned = x_train.copy()
    x_val_cleaned = x_val.copy()

    # For the cleaned data, encode the categorical columns. This is not necessary for catboost.
    for col_name in x_train.columns:
        if str(x_train[col_name].dtype) in ['category', 'object']:
            encoder = TargetEncoder()
            encoder.fit(x_train[[col_name]], y_train)
            x_train_cleaned[col_name] = encoder.transform(x_train[[col_name]]).flatten()
            x_val_cleaned[col_name] = encoder.transform(x_val[[col_name]]).flatten()

    # Test simply using all features
    start_time = process_time()
    if use_cleaned:
        score_val = test_all_features(model, x_train_cleaned, y_train, x_val_cleaned, y_val, metric=f1_score, average='macro')
    else:
        score_val = test_all_features(model, x_train, y_train, x_val, y_val, metric=f1_score, average='macro')
    time_for_all_features = process_time() - start_time

    # Test a filter method
    start_time = process_time()
    selector = SelectKBest(f_classif, k=max_features)
    selector.fit(x_train_cleaned.values, y_train)
    x_train_selected = pd.DataFrame(selector.transform(x_train_cleaned.values))
    x_val_selected = pd.DataFrame(selector.transform(x_val_cleaned.values))
    print("Testing with filter feature selection...")
    num_features_filter = len(x_train_selected.columns)
    score_from_filter = test_all_features(model, x_train_selected, y_train, x_val_selected, y_val, metric=f1_score, average='macro')
    time_for_filter_method = process_time() - start_time

    # Test a model method
    print("Testing with model-based feature selection...")
    start_time = process_time()
    clf = RandomForestClassifier()
    clf.fit(x_train_cleaned.values, y_train)
    selector = SelectFromModel(estimator=clf, prefit=True, max_features=max_features)
    x_train_selected = pd.DataFrame(selector.transform(x_train_cleaned.values))
    x_val_selected = pd.DataFrame(selector.transform(x_val_cleaned.values))
    num_features_model = len(x_train_selected.columns)
    score_from_model = test_all_features(model, x_train_selected, y_train, x_val_selected, y_val, metric=f1_score, average='macro')
    time_for_model_method = process_time() - start_time

    # Test forward selection. Skip cases where this would be prohibitively slow. This uses a RandomForest to identify
    # the best set of features, which is generally faster than using the actual model.
    if len(x_train.columns) < 40:
        print("Testing with wrapper feature selection...")
        start_time = process_time()
        clf = RandomForestClassifier()
        selector = SequentialFeatureSelector(estimator=clf, n_features_to_select=max_features).fit(x_train_cleaned, y_train)
        feats_map = selector.get_support()
        feats_selected = [x_train_cleaned.columns[x] for x in range(len(x_train_cleaned.columns)) if feats_map[x]]
        num_features_wrapper = len(feats_selected)
        score_from_wrapper = test_all_features(model, x_train_cleaned[feats_selected], y_train, x_val_cleaned[feats_selected], y_val, metric=f1_score, average='macro')
        time_for_wrapper = process_time() - start_time
    else:
        num_features_wrapper = -1
        score_from_wrapper = -1
        time_for_wrapper = -1

    # Test History feature selection
    print("Testing with History-based feature selection...")
    start_time = process_time()
    scores_df = feature_selection_history(
        model, x_train_cleaned, y_train, x_val_cleaned, y_val,
        num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25,
        max_features=max_features, penalty=None,
        verbose=True, draw_plots=False, plot_evaluation=False, metric=f1_score, average='macro')
    top_result = scores_df.iloc[0]
    num_features_history = top_result.tolist().count('Y')
    top_score = top_result['Score']
    time_for_history = process_time() - start_time

    return (data_description, model_description, len(x_train.columns), max_features,
            num_features_filter, num_features_model, num_features_wrapper, num_features_history,
            score_val, score_from_filter, score_from_model, score_from_wrapper, top_score,
            time_for_all_features, time_for_filter_method, time_for_model_method, time_for_wrapper, time_for_history)


def display_results():
    results_df = pd.DataFrame(
        results_arr,
        columns=['Data Description', 'Model Description', 'Num Features', 'Max Features',
                 'Num Features Filter', 'Num Features Model Method', 'Num Features Wrapper', 'Num Features History',
                 'Score Using all features', 'Score using Filter', 'Score using Model Method', "Score using Wrapper",
                 'Score using History',
                 'Time for All Features', 'Time for Filter Method', 'Time for Model Method', 'Time for Wrapper Method',
                 'Time for History Method'])
    print(results_df)
    results_df.to_csv(os.path.join("results", "results.csv"))


if __name__ == "__main__":
    pd.set_option('display.width', 32000)
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_colwidth', 3000)
    pd.set_option('display.max_rows', 5000)

    TEST_SYNTHETIC = False
    TEST_REAL = False
    TEST_REAL_ALL_COMBINATIONS = True
    results_arr = []

    # Specify the model to be used.
    # Specify the initial hyper-parameters prior to feature selection. Hyper-parameter tuning may be performed after
    # feature selection.
    model_catboost = CatBoostClassifier(verbose=False)
    model_dt = DecisionTreeClassifier()

    if TEST_SYNTHETIC:
        x_train, y_train, x_val, y_val = generate_data(n_rows=10_000, n_cols=14, num_relevant_cols=6, frac_pos=0.25, target_type=1)
        res = test_dataset("Synthetic data version 1", model_dt, x_train, y_train, x_val, y_val)
        results_arr.append(res)

        x_train, y_train, x_val, y_val = generate_data(n_rows=10_000, n_cols=14, num_relevant_cols=6, frac_pos=0.25, target_type=2)
        res = test_dataset("Synthetic data version 2", model_dt, x_train, y_train, x_val, y_val)
        results_arr.append(res)

        x_train, y_train, x_val, y_val = generate_data(n_rows=10_000, n_cols=14, num_relevant_cols=6, frac_pos=0.25, target_type=3)
        res = test_dataset("Synthetic data version 3", model_dt, x_train, y_train, x_val, y_val)
        results_arr.append(res)

        x_train, y_train, x_val, y_val = generate_make_classification(n_samples=10_000, n_features=20, n_informative=10, n_redundant=5)
        res = test_dataset("Make Classification 1", "Decision Tree", model_dt, x_train, y_train, x_val, y_val)
        results_arr.append(res)
        res = test_dataset("Make Classification 1", "Catboost", model_catboost, x_train, y_train, x_val, y_val)
        results_arr.append(res)

        x_train, y_train, x_val, y_val = generate_make_classification(n_samples=10_000, n_features=30, n_informative=10, n_redundant=5, n_clusters_per_class=4)
        res = test_dataset("Make Classification 2", "Decision Tree", model_dt, x_train, y_train, x_val, y_val)
        results_arr.append(res)
        res = test_dataset("Make Classification 2", "Catboost", model_catboost, x_train, y_train, x_val, y_val)
        results_arr.append(res)

    if TEST_REAL:
        real_files = [
            'iris',
            'soybean',
            'micro-mass',
            'mfeat-karhunen',
            'Amazon_employee_access',
            'abalone',
            'cnae-9',
            'semeion',
            'vehicle',
            'satimage',
            'analcatdata_authorship',
            'breast-w',
            'SpeedDating',
            'eucalyptus',
            'vowel',
            'wall-robot-navigation',
            'credit-approval',
            'artificial-characters',
            'splice',
            'har',
            'cmc',
            'segment',
            'JapaneseVowels',
            'jm1',
            'gas-drift',
            'mushroom',
            'irish',
            'profb',
            'adult',
            'higgs',
            'anneal',
            'credit-g',
            'blood-transfusion-service-center',
            'monks-problems-2',
            'tic-tac-toe',
            'qsar-biodeg',
            'wdbc',
            'phoneme',
            'diabetes',
            'ozone-level-8hr',
            'hill-valley',
            'kc2',
            'eeg-eye-state',
            'climate-model-simulation-crashes',
            'spambase',
            'ilpd',
            'one-hundred-plants-margin',
            'banknote-authentication',
            'mozilla4',
            'electricity',
            'madelon',
            'scene',
            'musk',
            'nomao',
            'bank-marketing',
            'MagicTelescope',
            'Click_prediction_small',
            'PhishingWebsites',
            'nursery',
            'page-blocks',
            'hypothyroid',
            'yeast',
            'kropt',
            'CreditCardSubset',
            'shuttle',
            'Satellite',
            'baseball',
            'mc1',
            'pc1',
            'cardiotocography',
            'kr-vs-k',
            'volcanoes-a1',
            'wine-quality-white',
            'car-evaluation',
            'solar-flare',
            'allbp',
            'allrep',
            'dis',
            'car',
            'steel-plates-fault'
        ]
        for file_name in real_files:
            test_dataset_openml(file_name)
            display_results()

    if TEST_REAL_ALL_COMBINATIONS:
        x_train, y_train, x_val, y_val = generate_make_openml('diabetes')

        for col_name in x_train.columns:
            if str(x_train[col_name].dtype) == 'category':
                x_train[col_name] = x_train[col_name].fillna(x_train[col_name].cat.categories[0])
                x_val[col_name] = x_val[col_name].fillna(x_train[col_name].cat.categories[0])
            elif x_train[col_name].dtype in [np.float64, np.int64]:
                x_train[col_name] = x_train[col_name].fillna(x_train[col_name].median())
                x_val[col_name] = x_val[col_name].fillna(x_train[col_name].median())
            else:
                x_train[col_name] = x_train[col_name].fillna(x_train[col_name].mode())
                x_val[col_name] = x_val[col_name].fillna(x_train[col_name].mode())

        model = CatBoostClassifier(verbose=False)

        test_all_features(model, model_args={'cat_features': []},
                          x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                          metric=f1_score, metric_args={'average': 'macro'})

        scores_df = feature_selection_history(
            model, {}, x_train, y_train, x_val, y_val,
            num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25,
            max_features=None, penalty=None,
            verbose=True, draw_plots=False, plot_evaluation=False, metric=f1_score, metric_args={'average': 'macro'})

        total_features = len(x_train.columns)
        scores_arr = []
        max_score = -1
        max_score_features = None
        for i in range(1, int(math.pow(2, total_features))):
            candidate = str(bin(i))[2:]
            if len(candidate) < total_features:
                candidate = '0'*(total_features - len(candidate)) + candidate
            candidate = list(candidate)
            candidate = [int(x) for x in candidate]
            cols = [x_train.columns[x] for x in range(total_features) if candidate[x] == 1]
            model = CatBoostClassifier(verbose=False)
            model.fit(x_train[cols], y_train)
            y_pred = model.predict(x_val[cols])
            score = f1_score(y_val, y_pred, average='macro')
            scores_arr.append(score)
            print(i, scores_arr[-1])
            if score > max_score:
                max_score = score
                max_score_features = cols
        print("Maximum score found testing every combination", max_score)
        print("Best set of features found", max_score_features)

    print()
    print('..................................................................................')
    print('..................................................................................')
    print('..................................................................................')
    print("Final Results:")
    display_results()
