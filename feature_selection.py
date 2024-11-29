import pandas as pd
import numpy as np
import copy
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import shap
from tqdm import tqdm
from IPython import get_ipython
from IPython.display import display, clear_output, Markdown
import matplotlib.pyplot as plt
import seaborn as sns


def test_all_features(model_in, x_train, y_train, x_val, y_val, metric, **kwargs):
    """
    Fits the model to the training data and evaluates on the validation data. This simply uses all available features,
    which creates a simple baseline to compare against. 
    
    Params
    model_in: model that is not yet fit but has the hyperparameters defined
    x_train: pandas dataframe
    y_train: list or series
    x_val: pandas dataframe
    y_val: list or series
    metric: metric used to evaluate the validation set
    kwargs: arguments used for the evaluation metric
    """

    model = clone(model_in)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)
    score_train = metric(y_train, y_pred_train, **kwargs)
    score_val = metric(y_val, y_pred_val, **kwargs)
    print()
    print((f"Testing with all ({len(x_train.columns)}) features: train score: {score_train}, "
           f"validation score: {score_val}"))


def __plot_bootstraps(model_in, x_train, y_train, x_val, y_val, sorted_col_names, metric, **kwargs):
    """
    This assumes we have the full set of available columns, sorted by their estimated predictive power. We then plot
    the train and validation scores using just the most predictive, the two most, the three most and so on. For each
    we give a range of scores based on bootstrap samples of the data.
    """
    num_bootstraps = 200

    print()
    msg1 = "Lineplots indicating model improvement with increased numbers of features:"
    msg2 = (("For each number of features tested, bootstrap samples were used to help estimate the uncertainty of the "
             "model given that number of features."))
    if is_notebook():
        display(Markdown(f'**{msg1}**'))
        display(Markdown(f'{msg2}'))
    else:
        print(msg1)
        print(msg2)

    num_cols_arr = []
    training_scores = []
    val_scores = []
    summary_arr = []
    for num_cols_used in tqdm(range(1, len(sorted_col_names) + 1)):
        model = clone(model_in)
        col_names = sorted_col_names[:num_cols_used]
        model.fit(x_train[col_names], y_train)
        inner_training_scores = []
        inner_val_scores = []
        for i in range(num_bootstraps):
            num_cols_arr.append(num_cols_used)

            x_train_sample = x_train.sample(n=len(x_train), replace=True, random_state=num_cols_used*1000+i)
            y_pred_train = model.predict(x_train_sample[col_names])
            score = metric(np.array(y_train)[x_train_sample.index.values], y_pred_train, **kwargs)
            training_scores.append(score)
            inner_training_scores.append(score)

            x_val_sample = x_val.sample(n=len(x_val), replace=True, random_state=num_cols_used*1000+i)
            y_pred_val = model.predict(x_val_sample[col_names])
            score = metric(np.array(y_val)[x_val_sample.index.values], y_pred_val, **kwargs)
            val_scores.append(score)
            inner_val_scores.append(score)
        summary_arr.append([num_cols_used,
                            np.mean(inner_training_scores), np.std(inner_training_scores),
                            np.mean(inner_val_scores), np.std(inner_val_scores),
                            ", ".join(sorted_col_names[:num_cols_used])])

    summary_df = pd.DataFrame(summary_arr, columns=['Number Features', 'Mean Training Score', "Std Dev Training Score",
                                                    "Mean Validation Score", "Std Dev Validation Score", "Features"])

    sns.lineplot(x=num_cols_arr, y=training_scores, label='Training Scores')
    sns.lineplot(x=num_cols_arr, y=val_scores, label="Validation Scores")
    plt.title(f"Train and Validation Scores \nover {num_bootstraps} Boostrap Samples for each Number of Features")
    plt.show()

    if is_notebook():
        display(summary_df)
    else:
        print(summary_df)


def feature_selection_filter(model_in, x_train, y_train, x_val, y_val, draw_plots, metric, **kwargs):
    """
    Evaluate the effectiveness of each single feature based on a 1d decision tree.
    """

    res = []
    for col_name in tqdm(x_train.columns):
        model = DecisionTreeClassifier()
        model.fit(x_train[[col_name]], y_train)
        y_pred_train = model.predict(x_train[[col_name]])
        y_pred_val = model.predict(x_val[[col_name]])
        score_train = metric(y_train, y_pred_train, **kwargs)
        score_val = metric(y_val, y_pred_val, **kwargs)
        res.append([col_name, score_train, score_val])
    res_df = pd.DataFrame(res, columns=['Column Name', 'Train Score', 'Validation Score'])
    res_df = res_df.sort_values('Validation Score', ascending=False)

    print()
    msg = "Listing the features from most to least predictive on the validation dataset:"
    if is_notebook():
        display(Markdown(f'**{msg}**'))
        display(res_df)
    else:
        print(msg)
        print(res_df)

    if draw_plots:
        print()
        msg = "Barplot indicating the features from most to least predictive on the validation dataset:"
        if is_notebook():
            display(Markdown(f'**{msg}**'))
        else:
            print(msg)

        sns.barplot(orient='h', y=res_df['Column Name'], x=res_df['Validation Score'], color='lightskyblue',
                    order=res_df['Column Name'])
        plt.title("Validation Scores Given Single Columns")
        plt.show()

        sorted_col_names = res_df['Column Name']
        __plot_bootstraps(model_in, x_train, y_train, x_val, y_val, sorted_col_names, metric, **kwargs)

    return res_df


def feature_selection_pairs(model_in, x_train, y_train, x_val, y_val, draw_plots, metric, **kwargs):
    res = []
    scores_matrix = np.zeros((len(x_train.columns), len(x_train.columns)))
    for col_0_idx, col_0_name in tqdm(enumerate(x_train.columns)):
        for col_1_idx in range(col_0_idx+1, len(x_train.columns)):
            col_1_name = x_train.columns[col_1_idx]
            model = clone(model_in)
            model.fit(x_train[[col_0_name, col_1_name]], y_train)
            y_pred_train = model.predict(x_train[[col_0_name, col_1_name]])
            y_pred_val = model.predict(x_val[[col_0_name, col_1_name]])
            score_train = metric(y_train, y_pred_train, **kwargs)
            score_val = metric(y_val, y_pred_val, **kwargs)
            res.append([col_0_name, col_1_name, score_train, score_val])
            scores_matrix[col_0_idx, col_1_idx] = score_val
    res_df = pd.DataFrame(res, columns=['Column 1 Name', 'Column 2 Name', 'Train Score', 'Validation Score']).sort_values('Validation Score', ascending=False)

    print()
    msg1 = "Listing each unique pair of features from most to least predictive on the validation dataset:"
    msg2 = "Displaying at most top 30."
    if is_notebook():
        display(Markdown(f'**{msg1}**'))
        display(Markdown(f'{msg2}'))
        display(res_df.head(30))
    else:
        print(msg1)
        print(msg2)
        print(res_df.head(30))

    if draw_plots:
        scores_matrix = pd.DataFrame(scores_matrix)
        scores_matrix = scores_matrix.replace(0, np.NaN)
        s = sns.heatmap(scores_matrix, cmap="Blues", annot=True, cbar=False, linewidths=1, linecolor='black')
        s.set_yticklabels(s.get_yticklabels(), rotation=0)
        s.set_xlim((0, len(x_train.columns)+1))  # Ensure the edges render properly
        s.set_ylim((len(x_train.columns)+1, 0))
        plt.title("Validation Scores Given Pairs of Distinct Columns")
        plt.tight_layout()
        plt.show()

    return res_df


def feature_selection_forward_wrapper(model_in, x_train, y_train, x_val, y_val, max_features, draw_plots, metric, **kwargs):
    """
    We first find the best single feature to use. Then, we try each other feature in combination with that, and select
    the best feature to use in combination with the one already selecte. We then find the next best one to add and so
    on.
    """

    def evaluate_candidate(col_names):
        model = clone(model_in)
        model.fit(x_train[col_names], y_train)
        y_pred_val = model.predict(x_val[col_names])
        score = metric(y_val, y_pred_val, **kwargs)
        return score

    num_feats = len(x_train.columns)

    # Find the first feature to add
    best_feature = None
    best_score = -np.inf
    for col_name in x_train.columns:
        score = evaluate_candidate([col_name])
        if score > best_score:
            best_feature = col_name
            best_score = score

    res = [[1, best_score, best_feature]]
    current_feature_set = [best_feature]

    # Loop through the rest of the features, adding one at a time
    for i in tqdm(range(1, max_features+1)):
        best_feature = None
        best_score = -np.inf
        for col_name in x_train.columns:
            if col_name in current_feature_set:
                continue
            score = evaluate_candidate(current_feature_set + [col_name])
            if score > best_score:
                best_feature = col_name
                best_score = score
        current_feature_set.append(best_feature)
        res.append([i+1, best_score, best_feature])

    res_df = pd.DataFrame(res, columns=['Number of Features', 'Score', 'Feature Added'])
    print()
    msg1 = "Listing each feature based on the order it is added to the set:"
    msg2 = ("This method adds features to the set one at a time in a greedy manner. The sooner a feature is added, "
            "on, average, the more predictive it is. The scores are provided for each number of features. Each set"
            "includes all additionally selected features as well as the currently-added feature.")
    if is_notebook():
        display(Markdown(f'**{msg1}**'))
        display(Markdown(f'{msg2}'))
        display(res_df)
    else:
        print(msg1)
        print(msg2)
        print(res_df)

    if draw_plots:
        sns.lineplot(data=res_df, x='Number of Features', y='Score')
        plt.show()

        sorted_col_names = res_df['Feature Added']
        __plot_bootstraps(model_in, x_train, y_train, x_val, y_val, sorted_col_names, metric, **kwargs)

    return res_df


def feature_selection_embedded(model_in, x_train, y_train, x_val, y_val, draw_plots, metric, **kwargs):
    """
    This trains a model using x_train and y_train, then uses SHAP values to determine the most relevant features to
    the model. Note: these are the features the model is using the most, and are not necessarily the most predictive.
    Using this method will tend to remove the features the model is not using anyway, which is useful (for example,
    it may reduce BigQuery costs), but will not tend to increase the accuracy.
    """

    x_train.columns = [str(x) for x in x_train.columns]

    print("Calculating SHAP values:")
    model = clone(model_in)
    model.fit(x_train, y_train)
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer(x_train)

    shap_df = pd.DataFrame(shap_values.values)
    shap_df.columns = x_train.columns
    shap_df = shap_df.abs()

    res_df = pd.DataFrame({"Feature Name": shap_df.mean().index, "Importance": shap_df.mean()})
    res_df = res_df.sort_values(by='Importance', ascending=False)
    display(res_df)

    if draw_plots:
        if is_notebook():
            shap.initjs()
        sns.barplot(orient='h', y=res_df['Feature Name'], x=res_df['Importance'], order=res_df['Feature Name'])
        plt.show()

        sorted_col_names = res_df['Feature Name']
        __plot_bootstraps(model_in, x_train, y_train, x_val, y_val, sorted_col_names, metric, **kwargs)

    return res_df


def feature_selection_SHAP(model_in, x_train, y_train, x_val, y_val, num_candidates, max_features, draw_plots, metric, **kwargs):
    """
    This uses SHAP values to estimate the skill of a model using various subsets of the features. These are then each
    evaluated.
    """

    def estimate_candidate(col_names):
        """
        Estimate the skill of a model using the SHAP values. Using these we can
        estimate what the model would predict for a given set of features for
        each row in the dataset, and can estimate the specified metric based on
        this.
        """
        pred = np.zeros(len(x_train))
        for col_name in col_names:
            pred = pred + shap_df[col_name]
        pred += base_value
        pred = pred > 0.0
        return f1_score(y_train, pred, average='macro')

    def evaluate_candidate(col_names):
        """
        In this function we actually train the model using the specified features
        and evaluate it using the specified metric.
        """
        model = clone(model_in)
        model.fit(x_train[col_names], y_train)
        y_pred_val = model.predict(x_val[col_names])
        score = metric(y_val, y_pred_val, **kwargs)
        return score

    x_train.columns = [str(x) for x in x_train.columns]

    print("Calculating SHAP values:")
    model = clone(model_in)
    model.fit(x_train, y_train)
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer(x_train)
    base_value = shap_values.base_values[0]
    shap_df = pd.DataFrame(shap_values.values)
    shap_df.columns = x_train.columns

    candidates_dict = {}
    current_features = []

    # Find the first feature to add
    print("\nIdentifying a set of cadidate sets of features based on the SHAP values:")
    best_feature = None
    best_score = -np.inf
    for col_name in x_train.columns:
        score = estimate_candidate([col_name])
        if score > best_score:
            best_feature = col_name
            best_score = score

    current_feature_set = [best_feature]

    # Loop 10 times, each time going through several passes of adding and
    # removing features
    for _ in tqdm(range(10)):
        best_feature = None
        best_score = -np.inf

        # Find the best feature to add up to 10 times before moving to
        # potentially removing features
        for _ in range(10):
            best_feature = None
            best_score = -np.inf
            for col_name in x_train.columns:
                if col_name in current_feature_set:
                    continue
                dict_key = tuple(sorted(current_feature_set + [col_name]))
                if dict_key in candidates_dict:
                    continue
                score = estimate_candidate(current_feature_set + [col_name])
                if score > best_score:
                    best_feature = col_name
                    best_score = score
            if best_feature:
                current_feature_set.append(best_feature)
                if len(current_feature_set) <= max_features:
                    # print(f"Adding {best_feature=}, {best_score=}")
                    candidates_dict[dict_key] = best_score

        # Find the best feature to remove up to 10 times before moving back to
        # potentially adding features
        for _ in range(10):
            best_set = None
            for col_name in current_feature_set:
                candidate_set = current_feature_set.copy()
                candidate_set.remove(col_name)
                dict_key = tuple(sorted(candidate_set))
                if dict_key in candidates_dict:
                    continue
                score = estimate_candidate(candidate_set)
                if score >= best_score:
                    best_set = candidate_set.copy()
                    best_score = score
            if best_set:
                current_feature_set = best_set.copy()
                if len(current_feature_set) < max_features:
                    candidates_dict[tuple(sorted(current_feature_set))] = best_score

    print("\nEvaluating the candidate sets:")
    candidates_tested = sorted([x for x in zip(candidates_dict.keys(), candidates_dict.values())], key=lambda x: x[1], reverse=True)[:num_candidates]
    res = []
    for candidate in tqdm(candidates_tested):
        res.append(evaluate_candidate(list(candidate[0])))
    res_df = pd.DataFrame({'Candidate': [x[0] for x in candidates_tested], 'Score': res})
    res_df = res_df.sort_values(by='Score', ascending=False)
    display(res_df)
    return res_df


def feature_selection_permutation(model_in, x_train, y_train, x_val, y_val, draw_plots, metric, **kwargs):
    """
    There are different variations on this test; this implements a quick and simple version. It trains a model on the
    full set of features and then determines how greatly the accuracy of the model drops by effectively removing (by
    scrambling) each feature one at a time.

    While SHAP scores indicate which features the model is currently (for better or worse) using the most, permutation
    scores indicate which features affect the accuracy of the model the most. Though, these are simple permutation tests
    that remove only one feature at a time, and do not fully consider feature interactions.
    """

    model = clone(model_in)
    model.fit(x_train, y_train)
    y_pred_val = model.predict(x_val)
    score_val = metric(y_val, y_pred_val, **kwargs)

    score_drops = []
    for col_name in x_train.columns:
        x_val_perm = copy.deepcopy(x_val)
        vals = x_val_perm[col_name].values
        np.random.shuffle(vals)
        x_val_perm[col_name] = vals
        y_pred_val_perm = model.predict(x_val_perm)
        score_val_perm = metric(y_val, y_pred_val_perm, **kwargs)
        score_drops.append(score_val - score_val_perm)

    res_df = pd.DataFrame({"Features": x_train.columns, "Drops in Score": score_drops})
    res_df = res_df.sort_values("Drops in Score", ascending=False)
    print()
    msg1 = "Listing each feature based on it's permuation score:"
    msg2 = ("This method trains a model of the full set of features, then assesses how predictive each feature is")
    if is_notebook():
        display(Markdown(f'**{msg1}**'))
        display(Markdown(f'{msg2}'))
        display(res_df)
    else:
        print(msg1)
        print(msg2)
        print(res_df)

    if draw_plots:
        sns.barplot(orient='h', y=res_df['Features'], x=res_df['Drops in Score'])
        plt.show()
        sorted_col_names = res_df['Features']
        __plot_bootstraps(model_in, x_train, y_train, x_val, y_val, sorted_col_names, metric, **kwargs)

    return res_df


def feature_selection_boruta(model_in, x_train, y_train, x_val, y_val, draw_plots, metric, **kwargs):
    """
    We create a set of shadow features (which are scrambled versions of the original features), one for each original
    feature. We then train a model on all features, the original and the shadow features. We take the maximum importance
    given to any shadow feature as the cutoff to determine what's a useful feature. We repeat this 20 times to get
    more stable results.

    model: Must include a feature_importances_ attribute
    x_train:
    y_train:
    draw_plots:
    metric:
    kwargs:
    """
    def create_boruta_df(df_orig):
        df_shadow = df_orig.apply(np.random.permutation)
        df_shadow.columns = ['shadow_' + feat for feat in df_orig.columns]
        df_boruta = pd.concat([df_orig, df_shadow], axis=1)
        return df_orig, df_shadow, df_boruta

    # Ensure all column names are in string format
    x_train.columns = [str(x) for x in x_train.columns]
    x_val.columns = [str(x) for x in x_val.columns]

    n_orig_feats = len(x_train.columns)
    x_train, x_train_shadow, x_train_boruta = create_boruta_df(x_train)
    x_val, x_val_shadow, x_val_boruta = create_boruta_df(x_val)

    hits = np.zeros((n_orig_feats * 2))
    n_iterations = 40
    res_df = pd.DataFrame({"Feature": x_train_boruta.columns})
    res_df['Is Shadow'] = res_df['Feature'].str.contains("shadow")
    iteration_features = []
    for iteration_idx in tqdm(range(n_iterations)):
        np.random.seed(iteration_idx)
        x_train, x_train_shadow, x_train_boruta = create_boruta_df(x_train)

        model = clone(model_in)
        model.fit(x_train_boruta, y_train)
        feat_importances = model.feature_importances_
        feat_name = f'Iteration_{iteration_idx}'
        iteration_features.append(feat_name)
        res_df[feat_name] = feat_importances

        feat_imp_shadow = feat_importances[n_orig_feats:]
        hits += (feat_importances > feat_imp_shadow.max())
    res_df['Average Importance'] = res_df[iteration_features].sum(axis=1) / n_iterations
    res_df['Hit %'] = hits * 100.0 / n_iterations

    print()
    msg1 = "Listing each feature and its Boruta Score"
    msg2 = ("This method trains a model on the full set of features as well as a set of shadow features. It then "
            "iterates 20 times, determining how often each real feature scores more highly than the maximum of the"
            "shadow features. This is based on the model's feature importance score. Features are ranked based on the"
            "number of iterations in which they outperform this cutoff.")
    if is_notebook():
        display(Markdown(f'**{msg1}**'))
        display(Markdown(f'{msg2}'))
        display(res_df.drop(columns=iteration_features))
    else:
        print(msg1)
        print(msg2)
        print(res_df.drop(columns=iteration_features))

    if draw_plots:
        msg = ("Vertical line shows the maximum importance assigned to any shadow feature, which is used as the cutoff "
               "to determine predictive features")
        if is_notebook():
            display(Markdown(f'{msg}'))
        else:
            print(msg)
        s = sns.barplot(orient='h', y=res_df['Feature'], x=res_df['Average Importance'], hue=res_df['Is Shadow'])
        s.axvline(res_df[res_df['Is Shadow']]['Average Importance'].max())
        plt.title("Average Feature Importance over 20 iterations for all actual and shadow features")
        plt.show()

        sns.barplot(orient='h', y=res_df['Feature'], x=res_df['Hit %'], hue=res_df['Is Shadow'])
        plt.title(("Percent of times each features is above the cutoff to estimate some \npredictive power over 20 "
                   "iterations.\nShowing for all actual and shadow features."))
        plt.show()

        res_df = res_df.sort_values(['Hit %'], ascending=False)
        sorted_col_names = res_df['Feature']
        sorted_col_names = [x for x in sorted_col_names if x in x_train.columns]
        __plot_bootstraps(model_in, x_train_boruta, y_train, x_val_boruta, y_val, sorted_col_names, metric, **kwargs)

    return res_df


def feature_selection_genetic(model_in, x_train, y_train, x_val, y_val, num_iterations, num_feats_selected, draw_plots,
                              metric, **kwargs):
    """
    This method returns the best set of features it can identify using the specified num_feats_selected. It considers
    sets of features as a whole and does not attempt to measure the quality of features on their own or in small sets.
    """

    def evaluate_candidate(candidate):
        col_names = [x_train.columns[x] for x in range(len(candidate)) if candidate[x] == 1]
        model = clone(model_in)
        model.fit(x_train[col_names], y_train)
        y_pred_val = model.predict(x_val[col_names])
        score = metric(y_val, y_pred_val, **kwargs)
        return score

    def crossover(parent1, parent2):
        """
        Combine two candidate sets of features, creating a single new candidate set of features.
        """

        # Find the superset of features selected in the two parents
        idxs = set(x for x in range(len(parent1)) if (parent1[x] or parent2[x]))

        # Select a random num_feats_selected of these
        idxs = np.random.choice(list(idxs), num_feats_selected, replace=False)

        offspring = np.zeros(len(parent1))
        for i in idxs:
            offspring[i] = 1
        return offspring

    def mutate(parent):
        """
        Randomly swap two values. Select two random features to swap. Most of the time this will select two features
        that have the same value in any case (both are included or both are excluded), so this will effectively not
        mutate the parent. However, in some cases, it will select two features such that one is included and one is
        excluded, and will mutate the the candidate.
        """
        child = copy.deepcopy(parent)
        i, j = np.random.choice(range(len(parent)), 2)
        orig_i = child[i]
        orig_j = child[j]
        child[i] = orig_j
        child[j] = orig_i
        return child

    num_feats = len(x_train.columns)

    # Create the initial population and evaluate each. Ensure we do not have duplicates
    scores_dict = {}
    print("Creating the initial set of candidates:")
    for _ in tqdm(range(20)):
        candidate = [0]*num_feats
        selected_feats = np.random.choice(range(0, num_feats), num_feats_selected, replace=False)
        for i in selected_feats:
            candidate[i] = 1
        candidate = tuple(candidate)
        if candidate in scores_dict:
            continue
        scores_dict[candidate] = evaluate_candidate(candidate)

    best_scores_hist = []
    print(f"Iterating to combine and mutate the current population:")
    for iteration_idx in tqdm(range(num_iterations)):
        # Select the top 5 candidates so far. Sort the scores_dict by the scores and take the tuples representing the
        # best candidates to date.
        population = [x[0] for x in sorted(list(zip(scores_dict.keys(), scores_dict.values())),
                                           key=lambda x: x[1],
                                           reverse=True)[:5]]
        best_scores_hist.append(max(scores_dict.values()))
        if draw_plots and is_notebook():
            clear_output()
            sns.lineplot(x=range(iteration_idx+1), y=best_scores_hist)
            plt.title(f"Best Score vs Num Iterations (up to {iteration_idx})")
            plt.show()

        # Pair up each candidate in the population for crossover. For each, create a single offspring based on
        # crossover, and create 5 mutated variations of each of these.
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                offspring = crossover(population[i], population[j])
                for k in range(5):
                    new_candidate = tuple(mutate(offspring))
                    if new_candidate in scores_dict:
                        continue
                    scores_dict[new_candidate] = evaluate_candidate(new_candidate)
                    assert sum(new_candidate) == num_feats_selected, f"{num_feats_selected=} {new_candidate=}"

    if draw_plots:
        sns.lineplot(x=range(num_iterations), y=best_scores_hist)
        plt.title(f"Best Score vs Num Iterations")
        plt.show()

    # Return the best set of features found for the specified number of features
    top_candidate = [x[0] for x in sorted(list(zip(scores_dict.keys(), scores_dict.values())),
                                          key=lambda x: x[1],
                                          reverse=True)][0]
    return [x_train.columns[x] for x in range(len(top_candidate)) if top_candidate[x]], max(scores_dict.values())


def feature_selection_genetic_range(model_in, x_train, y_train, x_val, y_val, num_iterations, num_feats_range,
                                    draw_plots, metric, **kwargs):
    """
    This is similar to feature_selection_genetic(), but accepts a range of values for num_feats_range, as opposed to
    a single value, so will find the best set of features (and the associated scores) for each number of features
    in the specified range. For each, this simply calls feature_selection_genetic().
    """
    res = []
    for num_feats_selected in tqdm(range(num_feats_range[0], num_feats_range[1]+1)):
        selected_features, score = feature_selection_genetic(model_in, x_train, y_train, x_val, y_val, num_iterations,
                                                             num_feats_selected, False, metric, **kwargs)
        res.append([num_feats_selected, score, selected_features])
    res_df = pd.DataFrame(res, columns=['Number of Features', 'Score', 'Features'])
    if is_notebook():
        display(res_df)
    else:
        print(res_df)

    if draw_plots:
        # todo: take bootstrap samples
        sns.lineplot(data=res_df, x='Number of Features', y='Score')
        plt.show()


def is_notebook():
    """
    Determine if we are currently operating in a notebook, such as Jupyter. Returns True if so, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False      # Probably standard Python interpreter


if __name__ == "__main__":
    pd.set_option('display.width', 32000)
    pd.set_option('display.max_columns', 3000)
    pd.set_option('display.max_colwidth', 3000)
    pd.set_option('display.max_rows', 5000)

    n_rows = 1_000
    n_cols = 14  # todo: test with about 100
    num_relevant_cols = 6
    frac_pos = 0.05

    def generate_data():
        def gen_target_col(df):
            sum = np.array([0]*len(df))
            for col_idx, col_name in enumerate(df.columns[:num_relevant_cols]):
                sum = sum + ((len(df.columns) - col_idx) * df[col_name])
            ret = [x < np.quantile(sum, frac_pos) for x in sum]

            # # Add random noise
            # modified_idxs = np.random.choice(range(n_rows), n_rows//20)
            # for i in modified_idxs:
            #     ret[i] = False if ret[i] else True

            return ret

        x_train = pd.DataFrame({x: np.random.rand(n_rows) for x in range(n_cols)})
        x_val = pd.DataFrame({x: np.random.rand(n_rows) for x in range(n_cols)})
        y_train = gen_target_col(x_train)
        y_val = gen_target_col(x_val)

        x_train.columns = ['F' + str(x) for x in x_train.columns]
        x_val.columns = ['F' + str(x) for x in x_val.columns]

        return x_train, y_train, x_val, y_val


    x_train, y_train, x_val, y_val = generate_data()
    model = DecisionTreeClassifier()

    # test_all_features(model, x_train, y_train, x_val, y_val, metric=f1_score, average='macro')
    # feature_selection_filter(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_pairs(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_forward_wrapper(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    feature_selection_embedded(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    # _ = feature_selection_SHAP(model, x_train, y_train, x_val, y_val, num_candidates=10, max_features=8, draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_permutation(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_boruta(model, x_train, y_train, x_val, y_val, draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_genetic(model, x_train, y_train, x_val, y_val, num_iterations=10, num_feats_selected=5,
    #                           draw_plots=True, metric=f1_score, average='macro')
    # feature_selection_genetic_range(model, x_train, y_train, x_val, y_val, num_iterations=10, num_feats_range=(2, 12),
    #                           draw_plots=True, metric=f1_score, average='macro')
