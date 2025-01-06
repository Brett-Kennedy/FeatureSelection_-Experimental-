import copy

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestRegressor
from IPython import get_ipython
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


def test_all_features(model_in, model_args, x_train, y_train, x_val, y_val, metric, metric_args):
    """
    Fits the model to the training data and evaluates on the validation data. This simply uses all available features,
    which creates a simple baseline to compare against.

    Params
    model_in: The model to be used, and for which we wish to find the optimal set of features.
        This should have the hyper-parameters set, but should not be fit.
    model_args: dictionary of parameters to be passed to the fit() method.
        For example, with CatBoost models, this may specify the categorical features.
    x_train: pandas dataframe of training data. This includes the full set of features, of which we wish to
        identify a subset.
    y_train: array. Must be the same length as x_train.
    x_val: pandas dataframe. Equivalent dataframe as x_train, but for validation purposes.
    y_val: list or series
    metric: metric used to evaluate the validation set
    metric_args: arguments used for the evaluation metric. For example, with F1_score, this may include the averaging
        method.

    Returns: float
        The score for the specified metric
    """

    if len(x_train.columns) == 0:
        print("No features specified")
        return None

    model = clone(model_in)
    model.fit(x_train, y_train, **model_args)
    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)
    score_train = metric(y_train, y_pred_train, **metric_args)
    score_val = metric(y_val, y_pred_val, **metric_args)
    print((f"Testing with all ({len(x_train.columns)}) features: train score: {score_train}, "
           f"validation score: {score_val}"))
    return score_val


def feature_selection_history(model_in, model_args,
                              x_train, y_train, x_val, y_val,
                              num_iterations=10, num_estimates_per_iteration=1000, num_trials_per_iteration=20,
                              max_features=None, penalty=None,
                              verbose=True, draw_plots=True, plot_evaluation=False,
                              metric=matthews_corrcoef, metric_args={}, higher_is_better=True,
                              previous_results=None):
    """
    model_in: model
        The model to be used, and for which we wish to find the optimal set of features.
        This should have the hyper-parameters set, but should not be fit.
    model_args: dictionary of parameters to be passed to the fit() method.
        For example, with CatBoost models, this may specify the categorical features.
    x_train: pandas dataframe of training data.
        This includes the full set of features, of which we wish to identify a subset.
    y_train: array
        Must be the same length as x_train.
    x_val: pandas dataframe
        Equivalent dataframe as x_train, but for validation purposes.
    y_val: list or series
        Must be the same length as x_val.
    num_iterations: int
        The number of times the process will iterate. Each iteration it retrains a Random Forest regressor that
        estimates the score using a given set of features, generates num_estimates_per_iteration random candidates,
        estimates their scores, and evaluates num_trials_per_iteration of these.
    num_estimates_per_iteration: int
        The number of random candidates generated and estimated using the Random Forest regressor.
    num_trials_per_iteration:
        The number of candidate feature sets that are evaluated each iteration. Each requires training a model and
        evaluating on the validation data.
    max_features: int
        The maximum number of features for any candidate subset of features considered. Set to None for no maximum.
    penalty: float
        The amount the score must improve for each feature added for two candidates to be considered equivalent, and
        so have the same scores with penalty. Set to None for no penalty. This allows the tool to evaluate and report
        candidates with any number of features, but favour those with fewer, favauring them to the degree specified by
        the penalty.
    verbose: bool
        If set True, some output will be displayed during the process of discovering the top-performing feature sets.
    draw_plots: bool
        If set True, plots will be displayed describing the process of finding the best features.
    plot_evaluation: bool
        If set True and draw_plots is also set True, an additional plot will be included in the display, which indicates
        how well the RandomForest regressor is able to predict the scores for candidate sets of features. In order to
        display this, it's necessary to evaluate more candidates, including many that are estimated to be weak, so
        setting this true will increase execution time, but does provide some insight into how well the process is
        able to work.
    metric: metric used to evaluate the validation set.
        This can be any metric supported by scikit-learn.This uses MCC by default, which is a less-commonly used, but
        effective metric for classification. It has the advantage of simplicity in that it balances FP, FN, TP and TN,
        and in that requires no parameters.
    metric_args: arguments used for the evaluation metric.
        For example, with F1_score, this may include the averaging method.
    higher_is_better: bool
        Set True for metrics where higher scores are better, such as MCC, F1, R2, etc. Set False for metrics where
        lower scores are better, such as MAE, MSE, etc.
    previous_results: dataframe
        The dataframe returned by a previous execution. Passing this allows us to continue searching for a stronger
        set of features for more iterations.
    """

    def evaluate_candidate(candidate):
        col_names = [x_train.columns[x] for x in range(len(candidate)) if candidate[x] == 1]
        model = clone(model_in)

        # Handle where the fit parameters include the list of categorical columns, as some may not be included in the
        # current candidate
        cleaned_model_args = copy.deepcopy(model_args)
        if 'cat_features' in model_args:
            cleaned_model_args['cat_features'] = [x for x in model_args['cat_features'] if x in col_names]

        model.fit(x_train[col_names], y_train, **cleaned_model_args)
        y_pred_val = model.predict(x_val[col_names])
        score = metric(y_val, y_pred_val, **metric_args)
        return score

    def create_df_from_scores_dict(d):
        res_arr = [list(x) for x in d.keys()]
        df = pd.DataFrame(res_arr, columns=list(range(len(x_train.columns))))
        scores_arr = [d[x] for x in d.keys()]
        return df, scores_arr

    def bin_num_features():
        interval_size = round(num_features/10)
        num_intervals = int(num_features / interval_size)
        bins = [x*interval_size for x in range(num_intervals+1)]
        return pd.cut(display_df['Num Features'].values, bins=bins)

    if len(x_train.columns) == 0:
        print("No features specified")
        return None

    # scores_dict records the score of every candidate evaluated.
    scores_dict = {}
    if previous_results is not None:
        features_list = list(previous_results.columns)
        features_list.remove('Num Features')
        features_list.remove('Score')
        if 'Score with Penalty' in previous_results.columns:
            features_list.remove('Score with Penalty')
        scores_dict = {tuple(previous_results.iloc[x][features_list].replace('Y', 1).replace('-', 0)):
                           previous_results.iloc[x]['Score']
                       for x in range(len(previous_results))}

    # Set parameter values if passed as None
    if max_features is None:
        max_features = len(x_train.columns)
    if penalty is None:
        penalty = 0.0

    # Generate an initial set of candidates randomly
    if verbose:
        print("\nGenerating the initial set of random candidates...")
    count_per_feature = [0]*len(x_train.columns)  # Used to ensure each feature is tested roughly equally often
    for i in range(num_trials_per_iteration):
        candidate = [0]*len(x_train.columns)
        if (i > 0) and ((i+1) % 3 == 0):
            # Every 3rd row, set the features to be the columns that are under-represented so far
            for j in range(len(x_train.columns)):
                if count_per_feature[j] < np.median(count_per_feature):
                    candidate[j] = 1
                    count_per_feature[j] += 1
        else:
            for j in range(len(x_train.columns)):
                r = np.random.rand()
                if r > 0.5:
                    candidate[j] = 1
                    count_per_feature[j] += 1
        if sum(candidate) == 0:
            continue
        dict_key = tuple(candidate)
        if dict_key in scores_dict:
            continue
        scores_dict[dict_key] = evaluate_candidate(candidate)
        if verbose:
            if len(x_train.columns) > 30:
                candidate_str = str(candidate)[:50] + "..."
            else:
                candidate_str = candidate
            print(f"{i:>3}, {candidate_str}, {scores_dict[dict_key]}")

    # Loop num_iterations times. Each loop train a RF on the candidates and their scores evaluated so far, generate
    # a large number of random additional candidates, estimate their scores using the RF, take the set from this
    # that the RF estimated to have the highest scores and determine their true scores
    if verbose:
        print(("\nRepeatedly generating random candidates, estimating their skill using a Random Forest, evaluating "
               "the most promising of these, and re-training the Random Forest..."))
    progress_arr = []
    estimated_vs_actual_scores_arr = []
    for iteration_num in range(num_iterations):
        regr = RandomForestRegressor()
        scores_df, scores = create_df_from_scores_dict(scores_dict)
        regr.fit(scores_df, scores)

        # Create a large number of random feature sets and estimate their score using the regressor
        estimated_scores_dict = {}
        for i in range(num_estimates_per_iteration):
            candidate = [0]*len(x_train.columns)
            threshold = np.random.rand()
            for j in range(len(x_train.columns)):
                r = np.random.rand()
                if r > threshold:
                    candidate[j] = 1
            if sum(candidate) == 0:
                continue
            if sum(candidate) > max_features:
                feats = [x for x in range(len(candidate)) if candidate[x]]
                feats = np.random.choice(feats, max_features, replace=False)
                candidate = [1 if x in feats else 0 for x in range(len(x_train.columns))]

            dict_key = tuple(candidate)
            if dict_key in estimated_scores_dict:
                continue
            predicted_score = regr.predict([candidate])
            num_features = sum(candidate)
            if higher_is_better:
                predicted_score_with_penalty = predicted_score - (num_features * penalty)
            else:
                predicted_score_with_penalty = predicted_score + (num_features * penalty)
            estimated_scores_dict[dict_key] = [predicted_score, predicted_score_with_penalty]

        # Find the top candidates based on the estimates of the RF
        candidates_arr = [list(x) for x in estimated_scores_dict.keys()]
        estimated_scores_df = pd.DataFrame(candidates_arr, columns=list(range(len(x_train.columns))))
        scores = [estimated_scores_dict[x][0] for x in estimated_scores_dict.keys()]
        scores_with_penalties = [estimated_scores_dict[x][1] for x in estimated_scores_dict.keys()]
        estimated_scores_df['Num Features'] = estimated_scores_df.sum(axis=1)
        estimated_scores_df['Score'] = np.array(scores).flatten()
        estimated_scores_df['Score with Penalty'] = np.array(scores_with_penalties).flatten()
        estimated_scores_df = estimated_scores_df.sort_values('Score with Penalty', ascending=not higher_is_better)

        # Evaluate the actual scores of the top candidates
        for i in range(min(len(estimated_scores_df), num_trials_per_iteration)):
            candidate = estimated_scores_df.drop(columns=['Num Features', 'Score', 'Score with Penalty']).iloc[i]
            dict_key = tuple(candidate)
            if dict_key in scores_dict:
                continue
            estimated_score = estimated_scores_df.iloc[i]['Score']
            score = evaluate_candidate(candidate)
            scores_dict[dict_key] = score
            progress_arr.append([iteration_num, score])
            estimated_vs_actual_scores_arr.append([estimated_score, score])

        if plot_evaluation:
            # If plot_evaluation is selected, we select an additional set of candidates to evaluate. This is done
            # strictly to evaluate how well the RandomForest is doing in terms of being able to predict the scores
            # produced by different subsets of features. To do this, it's necessary to evaluate both candidates where
            # the RF predicts low scores as well as where it predicts high scores. This doubles the number of
            # candidates evaluated. In order to not affect the process otherwise, these are not included in the scores
            # dictionary; they are used only to evaluate the RF regressor.
            for i in range(min(len(estimated_scores_df), num_trials_per_iteration)):
                idx = np.random.choice(list(estimated_scores_df.index))
                candidate = estimated_scores_df.drop(columns=['Num Features', 'Score', 'Score with Penalty']).iloc[idx]
                estimated_score = estimated_scores_df.iloc[idx]['Score']
                dict_key = tuple(candidate)
                if dict_key in scores_dict:
                    continue
                score = evaluate_candidate(candidate)
                estimated_vs_actual_scores_arr.append([estimated_score, score])

        if higher_is_better:
            print((f"Iteration number: {iteration_num:>3}, Number of candidates evaluated: {len(scores_dict):>4}, "
                   f"Mean Score: {np.mean(list(scores_dict.values())):.4f}, "
                   f"Max Score: {np.max(list(scores_dict.values())):.4f}"))
        else:
            print((f"Iteration number: {iteration_num:>3}, Number of candidates evaluated: {len(scores_dict):>4}, "
                   f"Mean Score: {np.mean(list(scores_dict.values())):.4f}, "
                   f"Min Score: {np.min(list(scores_dict.values())):.4f}"))

    scores_df, scores = create_df_from_scores_dict(scores_dict)
    scores_cols = scores_df.columns
    scores_df['Num Features'] = scores_df[scores_cols].sum(axis=1)
    scores_df['Score'] = scores
    if higher_is_better:
        scores_df['Score with Penalty'] = scores_df['Score'] - (penalty * scores_df['Num Features'])
    else:
        scores_df['Score with Penalty'] = scores_df['Score'] + (penalty * scores_df['Num Features'])
    for col_name in scores_cols:
        scores_df[col_name] = scores_df[col_name].replace(0, '-').replace(1, 'Y')
    scores_df = scores_df[scores_df['Num Features'] <= max_features]

    # Sort the values. If there is no penalty, sorting by Score with Penalty is equivalent to sorting by Score.
    scores_df = scores_df.sort_values('Score with Penalty', ascending=not higher_is_better)

    # Indicate the list of top features
    penalty_str = ""
    if penalty > 0:
        penalty_str = f" (considering a penalty of {penalty}"
    print(f"\nThe set of features in the top-scored feature set{penalty_str}")
    top_results = scores_df.iloc[0]
    for v_idx, v in enumerate(top_results):
        if v == 'Y':
            print(f"{v_idx:>4}, {x_train.columns[v_idx]}")

    if verbose:
        print("\nThe best-performing feature subsets found in terms of accuracy (showing at most 10) are:")
        if penalty == 0:
            scores_df = scores_df.drop(columns=['Score with Penalty'])
        if is_notebook():
            display(scores_df.head(10))
        else:
            print(scores_df.head(10))
        print("The full set of features is:")
        for col_idx, col_name in enumerate(x_train.columns):
            print(f"{col_idx:>4}: {col_name}")

    if draw_plots:
        if not estimated_scores_df.empty:
            estimated_vs_actual_scores_df = pd.DataFrame(estimated_vs_actual_scores_arr, columns=['Estimated Score', 'Score'])
            s = sns.regplot(data=estimated_vs_actual_scores_df, x='Estimated Score', y='Score',)
            s.set_title("Estimated scores vs actual scores \nfor all evaluated candidates")
            plt.show()
        else:
            print("Plot of estimated vs actual scores is not available given the small number of features.")

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        progress_df = pd.DataFrame(progress_arr, columns=['Iteration Number', 'Score'])
        if not progress_df.empty:
            if higher_is_better:
                s = sns.lineplot(data=progress_df, x='Iteration Number', y='Score', estimator='max', ax=ax[0][0])
                s.set_title("Scores for new candidates evaluated by iteration \n(highlighting the maximums)")
            else:
                s = sns.lineplot(data=progress_df, x='Iteration Number', y='Score', estimator='min', ax=ax[0][0])
                s.set_title("Scores for new candidates evaluated by iteration \n(highlighting the minimums)")
            s.set_xticks(list(range(progress_df['Iteration Number'].min(), progress_df['Iteration Number'].max()+1)))

        if higher_is_better:
            s = sns.lineplot(data=scores_df, x='Num Features', y='Score', estimator='max', ax=ax[0][1])
            s.set_title("Scores by Number of Features over all candidates evaluated\n(highlighting the maximums)")
        else:
            s = sns.lineplot(data=scores_df, x='Num Features', y='Score', estimator='min', ax=ax[0][1])
            s.set_title("Scores by Number of Features over all candidates evaluated\n(highlighting the minimums)")
        s.set_xticks(list(range(scores_df['Num Features'].min(), scores_df['Num Features'].max()+1)))
        clean_x_tick_labels(ax[0][1])

        display_df = estimated_scores_df.copy()
        num_features = len(x_train.columns)
        if num_features > 50:
            display_df['Num Features'] = bin_num_features()
        s = sns.boxplot(data=display_df, x='Num Features', y='Score', ax=ax[1][0])
        if num_features > 50:
            for label in ax[1][0].get_xmajorticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment("right")
        s.set_title("Estimated Scores by Number of Features \nover all candidates")

        display_df = scores_df.copy()
        if num_features > 50:
            display_df['Num Features'] = bin_num_features()
        s = sns.boxplot(data=display_df, x='Num Features', y='Score', ax=ax[1][1])
        if num_features > 50:
            for label in ax[1][1].get_xmajorticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment("right")
        s.set_title("Actual Scores by Number of Features \nover all evaluated (the top-estimated) candidates")

        plt.tight_layout()
        plt.show()

    return scores_df


def clean_x_tick_labels(ax):
    """
    n_axis: the number of axes in the figure
    """

    # Ensure the x tick labels are populated
    plt.draw()

    # Ensure there are at most 10 tick labels
    num_ticks = len(ax.xaxis.get_ticklabels())
    if num_ticks > 10:
        max_ticks = 10
        mod = num_ticks // max_ticks
        for label_idx, label in enumerate(ax.xaxis.get_ticklabels()):
            if label_idx % mod != 0:
                label.set_visible(False)


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
