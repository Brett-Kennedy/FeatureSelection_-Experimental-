# History-based Feature Selection (HBFS)

## Introduction

HBFS is a feature selection tool. Similar to wrapper methods, genetic methods, and some other approaches, HBFS seeks to find the optimal subset of features for a given dataset, target, and performance metric. 

This is different than methods such as filter methods, which seek instead to evaluate and rank each feature with respect to their predictive power.

# Algorithm

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

# Example

# API

# Installation

The tool uses a single .py file, which may be simply downloaded and used. It has no dependencies other than numpy, pandas, matplotlib, and seaborn.

# Testing Results

