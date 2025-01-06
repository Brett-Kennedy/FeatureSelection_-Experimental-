# History-based Feature Selection (HBFS)

## Introduction

HBFS is a feature selection tool. Similar to wrapper methods, genetic methods, and some other approaches, HBFS seeks to find the optimal subset of features for a given dataset, target, and performance metric. 

This is different than methods such as filter methods, which seek, instead, to evaluate and rank each feature with respect to their predictive power.

## Example
An [example notebook](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/Demo/Demo_History_Feature_Selection.ipynb) is provided, which provides a few examples performing feature selection using HBFS. The following is a portion of the notebook. 

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from history_based_feature_selection import test_all_features, feature_selection_history

# Collect the data to be used.
data = fetch_openml('credit-g', version=1, parser='auto')
x = pd.DataFrame(data.data)
y = data.target

# Pre-process the data (skipped here, but inlcuded in the notebook).
# This removes nulls and encodes categorical fields. 

# Divide the data into train and validate sets
n_samples = len(x) // 2
x_train = pd.DataFrame(x[:n_samples])
y_train = y[:n_samples]
x_val = pd.DataFrame(x[n_samples:])
y_val = y[n_samples:]

# Execute feature_selection_history(). This returns a dataframe tha lists the feature
# sets that were tested and their scores on the validation set. 
scores_df = feature_selection_history(
        model_dt, {},
        x_train, y_train, x_val, y_val,
        num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25, 
        max_features=None, penalty=None, 
        plot_evaluation=True, verbose=True, draw_plots=True,
        metric=f1_score, metric_args={'average':'macro'})
```

The output of the feature selection process (as verbose was set to True) includes:
```
Repeatedly generating random candidates, estimating their skill using a Random Forest, evaluating the most promising of these, and re-training the Random Forest...
Iteration number:   0, Number of candidates evaluated:   50, Mean Score: 0.6109, Max Score: 0.6857
Iteration number:   1, Number of candidates evaluated:   74, Mean Score: 0.6240, Max Score: 0.6868
Iteration number:   2, Number of candidates evaluated:   92, Mean Score: 0.6302, Max Score: 0.6905
Iteration number:   3, Number of candidates evaluated:  113, Mean Score: 0.6369, Max Score: 0.6946
Iteration number:   4, Number of candidates evaluated:  132, Mean Score: 0.6396, Max Score: 0.6969
Iteration number:   5, Number of candidates evaluated:  148, Mean Score: 0.6416, Max Score: 0.6969
Iteration number:   6, Number of candidates evaluated:  163, Mean Score: 0.6440, Max Score: 0.6969
Iteration number:   7, Number of candidates evaluated:  180, Mean Score: 0.6456, Max Score: 0.6975
Iteration number:   8, Number of candidates evaluated:  199, Mean Score: 0.6473, Max Score: 0.6975
Iteration number:   9, Number of candidates evaluated:  218, Mean Score: 0.6493, Max Score: 0.7061

The set of features in the top-scored feature set
   0, checking_status
   1, duration
   2, credit_history
   3, purpose
   4, credit_amount
   6, employment
   7, installment_commitment
   8, personal_status
   9, other_parties
  11, property_magnitude
  12, age
  13, other_payment_plans
  15, existing_credits
  17, num_dependents
  19, foreign_worker
```
This indicates that 10 iterations were executed (as specified) and that the maximum score progressed from 0.6857 on the first iteration (for the top-scoring feature set discovered by that point) to 0.7061 by the last iteration. We can also see the mean scores of the candidates that are evaluated steadily improving (from about 0.61 to about 0.65), indicating the process is able to learn at each step, and identify candidate feature sets that do, in fact, tend to perform better than previous iterations. 

The final output also includes a summary of the feature sets evaluated, sorted by their performance: 

![plot1](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output1.png)

Interally, HBFS works by using a Random Forest regressor to estimate the metric score that would be given to any given set of features. HBFS selects the feature sets that are estimated to perform the best and evaluates these (training a model on the training data using these features, and evaluating it on the validation data). In this way, HBFS evaluates the feature sets most likely to perform well, but does this
over several iterations, so that the Random Forest can become increasingly more skilled at estimating how well a given feature set would perform, based on the history of feature evaluated up until that iteration. 

To evaluate how well the Random Forest that's used internally was able to estimate the scores for random features sets, we plot the estimated versus actual scores for all feature sets that were actually evaluated:

![plot1](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output2.png)

In this case, the RandomForest was not perfect, but quite able to estimate the scores, and certainly able to distinguish strong from weak candidates. 

We also provide some information about the progress (with respect to identifying stronger candidates over time, and related to the estimated and actual performance versus the number of features used. 

![plot1](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output3.png)

The top-left plot shos the range of scores of the candidates evaluated each iteration. In this example, the maximum value is steadily increasing, which is most relevant. 

The top-right plot shows the scores given by the number of features for the feature sets that were evaluated. The maximum score is the most relevant for each number of features.

The bottom-left plot shows the range of estimated scores by the number of features. We see that, for this dataset, everything else equal, the Random Forest estimates higher scores when using more features, but that there is a large range for each number of features, and that other factors are taken into consideration.

The bottom-right plot shows that range of actual scores by the number of features, similar to the plot above it. 

## Example with a regression target, and demonstrating HBFS's ability to continue execution

HBFS can support binary and multiclass classification, as well as regression. In this example, we have a regression target, and we use MAE as the metric, which is an example of a metric where lower is better (the previous example used f1_score, where higher is better). For this, we the parameter higher_is_better to False. 

This calls feature_selection_history() twice. The first time it executes for just 2 iterations. The results from this are passed as a parameter in the 2nd call, which is able to continue from that point, executing an additional 10 iterations. 

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
from history_based_feature_selection import test_all_features, feature_selection_history

model_dt = DecisionTreeRegressor()
x, y, = make_regression(n_samples=10_000, n_features=20)

# Divide into train and validate sets
n_samples = len(x) // 2
x_train = pd.DataFrame(x[:n_samples])
y_train = y[:n_samples]
x_val = pd.DataFrame(x[n_samples:])
y_val = y[n_samples:]

test_all_features(model_dt, {}, x_train, y_train, x_val, y_val, metric=mean_absolute_error, metric_args={})

np.random.seed(0)

scores_df = feature_selection_history(
    model_dt, {},
    x_train, y_train, x_val, y_val,
    num_iterations=2, num_estimates_per_iteration=5_000, num_trials_per_iteration=25,
    max_features=None, penalty=None,
    verbose=True, draw_plots=True, plot_evaluation=True,
    metric=mean_absolute_error, metric_args={}, higher_is_better=False)

print("---------------------------------------------------------")
scores_df = feature_selection_history(
    model_dt, {},
    x_train, y_train, x_val, y_val,
    num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25,
    max_features=None, penalty=None,
    verbose=True, draw_plots=True, plot_evaluation=True,
    metric=mean_absolute_error, metric_args={}, higher_is_better=False,
    previous_results=scores_df
)
```
The output of this (edited slightly here for space) is:

```
Testing with all (20) features: train score: 0.0, validation score: 101.36362689302662

Repeatedly generating random candidates, estimating their skill using a Random Forest, evaluating the most promising of these, and re-training the Random Forest...
Iteration number:   0, Number of candidates evaluated:   50, Mean Score: 136.5973, Min Score: 97.0896
Iteration number:   1, Number of candidates evaluated:   72, Mean Score: 125.3618, Min Score: 97.0896
---------------------------------------------------------
Repeatedly generating random candidates, estimating their skill using a Random Forest, evaluating the most promising of these, and re-training the Random Forest...
Iteration number:   0, Number of candidates evaluated:  113, Mean Score: 133.1507, Min Score: 97.0743
Iteration number:   1, Number of candidates evaluated:  131, Mean Score: 128.3860, Min Score: 96.1117
Iteration number:   2, Number of candidates evaluated:  151, Mean Score: 124.3848, Min Score: 93.9759
Iteration number:   3, Number of candidates evaluated:  171, Mean Score: 121.3966, Min Score: 93.9759
Iteration number:   4, Number of candidates evaluated:  192, Mean Score: 118.8837, Min Score: 93.9759
Iteration number:   5, Number of candidates evaluated:  205, Mean Score: 117.6166, Min Score: 93.9759
Iteration number:   6, Number of candidates evaluated:  224, Mean Score: 115.9409, Min Score: 93.9759
Iteration number:   7, Number of candidates evaluated:  245, Mean Score: 114.4119, Min Score: 93.9759
Iteration number:   8, Number of candidates evaluated:  270, Mean Score: 112.9481, Min Score: 93.9759
Iteration number:   9, Number of candidates evaluated:  295, Mean Score: 111.6582, Min Score: 93.9759

```
Here, using all 20 features, we are able to achieve a MAE of 101.36. Executing HBFS for two iterations, we find a better (lower error) score of 97.09. Executing for an additional 10 iterations we get a better-still socre of 93.98. 

It's difficult to say how many iterations are necessary ahead of time (this depends on the original number of features, the difficulty in predicting the target from the features, the num_estimates_per_iteration, and num_trials_per_iteration parameters), but in this case, the minimum score did not improve for the last several iterations (even though the mean score was improving), so this is likely the optimal feature set in terms of MAE, or at least quite close. 

## Example using CatBoost, specifying the categorical columns

```python
scores_df = feature_selection_history(
    model_dt, {'cat_features': ['A']}, x_train, y_train, x_val, y_val,
    num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25,
    max_features=None, plot_evaluation=True, penalty=None,
    verbose=True, draw_plots=True, metric=mean_absolute_error, metric_args={}, higher_is_better=False,
    previous_results=scores_df
)
```
The second parameter here is the 'model_args' parameter, which may be used to pass parameters to the fit method. In this example, CatBoost model can accept a list of categorical columns in either the constructor or the fit() method. The fit() method must be used in this case, as each time HBFS fits a model using a candidate set of features, this candidate set may have any subset of the categorical features. 

## Algorithm

The algorihm, in psuedo-code is:
```
Loop a specfied number of times (by default, 20)
| Generate several random subsets of features, each covering about half 
|     the features
| Train a model using this set of features using the training data
| Evaluate this model using the validation set
| Record this set of features and their evaluated score

Loop a specified number of times (by default, 10)
|   Train a RandomForest regressor to predict, for any give subset of 
|      features, the score of a model using those features. This is trained
|      on the history model evaluations of so far.
|  
|   Loop for a specified number of times (by default, 1000)
|   |   Generate a random set of features
|   |   Use the RandomForest regressor estimate the score using this set 
|   |     of features
|   |   Store this estimate
|
|   Loop over a specfied number of the top-estimated candidates from the 
|   |       previous loop (by default, 20)
|   |   Train a model using this set of features using the training data
|   |   Evaluate this model using the validation set
|   |   Record this set of features and their evaluated score
```

The algorithm starts by generating a collection of random feature sets, each covering approximately half the features. So, if a dataset has, say 100 features, and the defaults are used, we'll generate 20 candidates, each with about 50 features. Each feature will then be included in about 10 of the candidates, and excluded in about 10 of the candidates, which provides some information about how predictive each feature is (for example, if models tend to perform better, about the same, or worse with the feature -- though Random Forests can capture feature interactions as well).

The algorithm then iterates for a specified number of iterations. Each iteration it begins by training a Random Forest regressor to predict, for any given set of features, the metric score that would be achieved by a model trained on those features and evaluated on a validation set. This is trained on all feature sets that have been evaluated so far.

It then generates a large number of random feature sets and passes these through the Random Forest predictor, which estimates the scores that would be associated with each. It then selects the feature sets that are estimated to be the top-performing and evaluates these. It then, for each of these top-estimated features sets (by default it takes 20 each iteration), trains a model (of the specified model type) on the provided training data and evaluates this on the provided validation data.

This process is then repeated a specfied number of times. Generally, about 4 to 12 iterations are necessary to find an optimal, or near-optimal, feature set. 

## Options

It's possible to use HBFS to either 1) simply search for the subset of features that result in the highest metric score; or 2) to balance this with minimizing the number of features returned. 

Both scenarios are quite common when performing feature selection, but are actually distinct goals. Simply maximizing the accuracy is relatively straight-forward. Balancing the goals of maximizing accuracy with minimizing the number of features is more complex. 

To support balancing these goals, it's possible to specifiy one or both of: a maximum number of features, and a penalty. 

Specifying a maximum number of features ensures HBFS only evaluates (other than the first iteration), and only returns, feature sets with this many or fewer features. 

If a penalty is specified, the tool may return any number of features (up to the maximum number if this is specified), but will favour feature sets with fewer features, and only return feature sets with more features if the increase in metric score warrants this. This is similar to Lasso regularization often used with linear regression, and to penalties for additional features used by AIC and BIC, though here specific penalty can be specified to best suite the balance between accuracy and reduced features best matching your project. 

An example using a penalty is provided in the [example notebook](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/Demo/Demo_History_Feature_Selection.ipynb).

## API

The tool provides two methods: test_all_features() and feature_selection_history(). 

### test_all_features

The test_all_features() method simply tests training and evaluating the model using all features. This is a smoke test to ensure
the specified model can work with the passed data (it has been properly pre-processed and so on). It also provides a score using the specified metric, which establishes a baseline we attempt to beat by calling feature_selection_history() and finding a subset of features. 

**test_all_features**(model_in, model_args, x_train, y_train, x_val, y_val, metric, metric_args)
  
    model_in: The model to be used, and for which we wish to find the optimal set of features.        
        This should have the hyper-parameters set, but should not be fit.
        
    model_args: dictionary of parameters to be passed to the fit() method.    
        For example, with CatBoost models, this may specify the categorical features.
    
    x_train: pandas dataframe of training data.     
        This includes the full set of features, of which we wish to identify a subset.
    
    y_train: array. 
        Must be the same length as x_train.
    
    x_val: pandas dataframe. 
        Equivalent dataframe as x_train, but for validation purposes.
    
    y_val: list or series
    
    metric: metric used to evaluate the validation set
    
    metric_args: arguments used for the evaluation metric.     
        For example, with F1_score, this may include the averaging method.

    Returns: float
        The score for the specified metric

### feature_selection_history 
feature_selection_history() is the main method provided by this tool. Given a model, dataset, target column, and metric, it seeks to find the set of features that maximizes the specified feature. 

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

    Returns: dataframe listing each feature set tested, the number of features, and the score on the validation set. If 
        a penalty was provided, this also returns, for each feature set tested, the score with penalty.         

## Installation

The tool uses a single .py file, which may be simply downloaded and used. It has no dependencies other than numpy, pandas, matplotlib, and seaborn.

## Testing Results - Comparing to other feature selection methods

A [test file](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/Tests/test_history_based_feature_selection.py) is included which provides some testing of HBFS. In the main code, there are 3 constants:

```python
    TEST_SYNTHETIC = False
    TEST_REAL = False
    TEST_REAL_ALL_COMBINATIONS = True
```
These may be used to control which tests are executed. However, the tests cover many test cases and take a very long time to execute. The main test (enabled by setting TEST_REAL to True), covers 80 random datasets from OpenML, comparing HBFS to: 

- Using all features
- A filter feature selection method
- A model-based feature selection method
- A wrapper feature selection method

Tests were done both with a Decision Tree classifier and a CatBoost classifier, and setting the maximum number of features to 1/4, 1/2, and 3/4 of the original number of features. 

The complete results can be seen in the [results csv file](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/Tests/results_full.csv).

A more thorough analysis of the results in provided in the Medium article. 

Comparing the distribution of scores over all datasets, model types, and maximum numbers of features:

![plot6](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output6.png)

This shows using all features and using HBFS tend to outperform the other feature selection methods in terms of the f1 macro score (the metric used in these tests).

Treating using all features as the baseline that we try to beat, we next normalize the scores for each test by their ratio to the score given when using all features. Doing this, using all features consistently has a normalized score of 1.0; the other feature selection methods have normalized scores above 1.0 if they performed better, and a normalized score below 1.0 if they did worse.

![plot7](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output7.png)

We can see that HBFS selection quite consistently outperformed using all features (though did not in 100% of cases), and that the other feature selection methods more often did worse. However, testing other features selection methods, or using these with different settings may have performed better. 

Looking at the fraction of the time that each feature selection method outperformed using all features:

![plot8](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output8.png)

HBFS did so about 65% of the time, while the other feature selection methods did so only about 20 or 25% of the time (more often performing worse than simply using all features). Using only 1/4 of the features in particular, it was difficult to beat using all features, though using 1/2 or 3/4 of the original features tended to be more achievable. In any case, when comparing the four feature selection methods, each was set to identify the same maximum number of features, so comparisons among the four feature selection methods may be more fair than comparing to using all features. 

Looking at the count of tests where each method was the top-performing feature selection method:

![plot9](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/images/output9.png)

We see here that HBFS was the most, followed by using all features, of the 5 methods tested here. 


## Testing Results - Comparing to an exhaustive search over all possible feature sets

Ideally, to confirm that HBFS provides an optimal, or nearly optimal, set of features (given the maximum features and penalty specified), we would compare the feature set it identifies with every possible subset of features.

If we have, say, 30 features, and HBFS returns a set of 8 features that result in an R2 score of 0.87 (let's assume this is a regression target, and that R2 is the most relevant metric), the only definitive way to determine how good of a solution this is is to test all possible subsets of the 30 features and determine if any did better than 0.87, and if any did significantly better. 

Unfortunately, in practice this is infeasible. Where there are d features, there are 2**d possible subsets. With 10 features, this is only 1024, but with 20, this is over a million. With 30 features, this is over a billion, and with 100 features, it's a number with 31 digits. 

So, for any cases where feature selection is most important (the more features there are, the more relevant feature selection is), this isn't possible. In these cases, it's necessary to either use HBFS or another feature selection method.

We do, though, provide here a quick test using the diabetes dataset from OpenML, which has only 9 features, which means there are only 2⁹, or 256 possible feature sets. In this case, HBFS is able to identify the top-performing feature set within two iterations. 

These results are informative (and testing with other datasets with reasonably few features has worked similarly), but shouldn't be taken to indicate the HBFS will do as well with datasets with many more features. In these cases, there are drastically more potential feature sets and it's unlikely any algorithm would identify the truly optimal set. It's also impossible to confirm if it did. 

However, testing has indicated that HBFS tends to find a very strong feature set, likely as strong as with any other method, and does so quickly given the quality of response it's able to produce.
The code for this experiment is also provided in the [test file](https://github.com/Brett-Kennedy/HistoryBasedFeatureSelection/blob/main/Tests/test_history_based_feature_selection.py) and is executed by setting TEST_REAL_ALL_COMBINATIONS to True. 

In this experiment, we again use macro f1 score for the metric. For the model type, we used CatBoostClassifier, and so attempted to find the set of features that optimized the macro f1 score for a CatBoost classifier.

We tested three cases:
- Using all features
- Using HBFS
- An exhaustive search over all ²⁹ subsets of features.

Using all features resulted in a score of 0.71. Using HBFS resulted in 0.76. Testing over all combinations confirmed that 0.76 was, in fact, the best possible score for this dataset.

We should note though: with any dataset where an exhaustive search over all combination is possible, while HBFS does well in my testing to date, the difficulty is not great. Setting the number of iterations to 2 and number of trials per iteration to 25 will test a maximum of 50 combinations, which can be close to the full set of possible combinations (though in practice HBFS will actually evaluate far fewer). In this case, HBFS will test about 1/5 of the total possible combinations of features (there are 256 possible subsets). 

In the example below, it's allowed to run for 10 iterations, though requires far fewer iterations in this case to identify the optimal feature set.
Testing first with all features outputs:

```
Testing with all (8) features: train score: 0.976, validation score: 0.718
```

HBFS identifies a feature set with 3 features (plas, mass, and age), which achieves 0.76. 

The output indicates the top 10 candidates that were evaluated:

```
   0 1 2 3 4 5 6 7 Num Features, Score
24 - Y - - - Y - Y     3         0.760027
29 - Y - Y - Y - Y     4         0.759284
25 - Y Y - Y Y - Y     5         0.749020
34 - Y Y Y Y Y - Y     6         0.746032
26 - Y Y - - Y Y Y     5         0.745394
31 - Y - Y Y Y - Y     5         0.742253
55 - Y - - Y - - Y     3         0.742090
43 Y Y - Y - Y - Y     5         0.740759
35 - Y Y Y - Y Y Y     6         0.738770
51 - Y - - - - - Y     2         0.738746

The full set of features is:
 0: preg
 1: plas
 2: pres
 3: skin
 4: insu
 5: mass
 6: pedi
 7: age
```


