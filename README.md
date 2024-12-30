# History-based Feature Selection (HBFS)

## Introduction

HBFS is a feature selection tool. Similar to wrapper methods, genetic methods, and some other approaches, HBFS seeks to find the optimal subset of features for a given dataset, target, and performance metric. 

This is different than methods such as filter methods, which seek instead to evaluate and rank each feature with respect to their predictive power.

## Example
An example notebook is provides, which provides a few examples performing feature selection using HBFS. The following is a portion of the notebook. 

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from history_based_feature_selection import test_all_features, feature_selection_history

# Divide into train and validate sets
n_samples = len(x) // 2
x_train = pd.DataFrame(x[:n_samples])
y_train = y[:n_samples]
x_val = pd.DataFrame(x[n_samples:])
y_val = y[n_samples:]

scores_df = feature_selection_history(
        model_dt, {}, x_train, y_train, x_val, y_val,
        num_iterations=10, num_estimates_per_iteration=5_000, num_trials_per_iteration=25, 
        max_features=None, plot_evaluation=True, penalty=None, 
        verbose=True, draw_plots=True, metric=f1_score, metric_args={'average':'macro'})
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

## Example for regression

HBFS can support binary and multiclass classification as well as regression. In this example, we use MAE as the metric, which is an example of a metric where lower is better (the previous example used f1_score, where higher is better).

```python
```


## Algorithm

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

## API

The tool provides two methods: test_all_features() and feature_selection_history(). 

The test_all_features() method simply tests training and evaluating the model using all features. This is a smoke test to ensure
the specified model can work with the passed data (it has been properly pre-processed and so on). It also provides a score using the specified metric, which establishes a baseline we attempt to beat by calling feature_selection_history() and finding a subset of features. 

feature_selection_history() is the main method provided by this tool. Given a model, dataset, target column, and metric, it seeks to find the set of features that maximizes the specified feature. 

## Installation

The tool uses a single .py file, which may be simply downloaded and used. It has no dependencies other than numpy, pandas, matplotlib, and seaborn.

## Testing Results

