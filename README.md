# model-selection
Library for the selection of machine learning models.

This library is part of my bachelor thesis, check out the other works.
- [timeSeries-processing](https://github.com/EnricoPittini/timeSeries-processing)
- [EEA-datasets-handler](https://github.com/EnricoPittini/EEA-datasets-handler)
- [ILMETEO-datasets-handler](https://github.com/EnricoPittini/ILMETEO-datasets-handler)
- [Air-quality-prediction](https://github.com/EnricoPittini/Air-quality-prediction)

## Description

### Purpose
The main purpose of this library is to perform the selection of the best machine learning model among several ones.
In other words, the purpose is to perform the validation of different machine learning models.

This library is meant to be simple and intuitive, but also rich. In this way, the user is able to personalize the model selection in a powerful and flexible way.  

In addition, the library allows the user to plot the results of the validation, in order to graphically visualize the model selection.

There are several different model selection functions. Each of them is specifically designed for a particular scenario. Certain contexts are more specific, and others are more general: on the whole, the vast majority of the possible scenarios are covered.
This separation of the single model selection task into multiple functions has been made to keep the interface of the library both easy and flexible.

### Functionalities
On the whole, there are six different model selection functions, divided into two groups.
1. In the first group there are the functions which perform the model selection with respect to a single dataset.
2. In the second group there are the functions which perform the model selection with respect to multiple datasets.

In particular, the functions in the first group are the following, sorted from the most specific to the most general.
- hyperparameter_validation. It performs the tuning of the hyperparameter of a certain model.
- hyperparameters_validation. It performs the tuning of some several hyperparameters of a certain model.
- models_validation. It performs the model selection among different models.

The functions in the second group are like the ones in the first group, but they perform the model selection among different datasets. This means that not only the best model is selected, but also the best dataset is selected.
The functions in the second group are the following, sorted from the most specific to the most general.
- datasets_hyperparameter_validation
- datasets_hyperparameters_validation
- datasets_models_validation

This library, besides the model selection functions, contains also some other secondary functionalities.
It contains the PolynomialRegression class, which is a machine learning model implementing the polynomial regression.
It also contains some functions which compute the training-validation-test scores and the bias^2-variance-error for a certain model on the given dataset.
Finally, it contains a function which is able to plot the predictions made by a certain model on the given dataset against the actual values. This is useful for graphically visualizing the goodness of a certain model.

### Implementation details
The selection of the best model is performed computing the validation scores, which means that the best model is the one associated with the best validation score.
In particular, each dataset is split into training and test sets, and then the cross validation strategy is applied on the training set. In this way, the validation score is computed.
Additionally, the training and test scores are also computed. These are simply additional indicators: the selection is performed considering only the validation scores.
While the validation and training scores are computed for each model, the test score is computed only for the best one. The test score can be considered as a final measure of goodness for the best model.

As described above, this library has a rich interface, in order to allow the user to personalize the model selection in a powerful and flexible way.  
Some examples are now presented.
- The user can specify whether the machine learning problem is a regression or a classification task. This choice influences the technique used for the selection.
- The user can specify whether the given datasets are time-series datasets or not. This choice influences the technique used for the selection.
- The user can specify whether the explanatory features have to be scaled or not.

This library is built on top of the scikit-learn library.
The machine learning models are indeed represented as scikit-learn models, i.e. they are compliant with the [scikit-learn estimators interface](https://scikit-learn.org/stable/developers/develop.html).
In addition, under the hood, the selection is performed using the grid search cross validation provided by scikit-learn (i.e.  GridSearchCV);
Moreover, several other scikit-learn utilities are used.

The datasets are instead represented as NumPy arrays, i.e. they have type np.ndarray.

Finally, the plots are made using the Matplotlib library.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install model-selection.

```bash
pip install ml-model-selection
```

## Usage

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import model_selection as ms


###### Single dataset X,y (numpy array)

# hyperparameter_validation
model = DecisionTreeRegressor()
hyperparameter = "max_leaf_nodes"
hyperparameter_values = [2,3,4]
train_val_scores, best_index, test_score, ax = ms.hyperparameter_validation(X, y, model, hyperparameter,
                                                                     hyperparameter_values, plot=True,
                                                                     plot_train=True)

# hyperparameters_validation                                                        
model = DecisionTreeRegressor()
param_grid = {"max_leaf_nodes":[2,3,4], "max_features":[None,"sqrt"]}
params, train_val_scores, best_index, test_score = ms.hyperparameters_validation(X, y, model, param_grid,
                                                                                time_series=True)

# models_validation
model_paramGrid_list = [
                         ("DT", DecisionTreeRegressor(), {"max_leaf_nodes":[2,3,4],
                                                          "max_features":[None,"sqrt"]} ),
                         ("kNN", KNeighborsRegressor(), {"n_neighbors":[1,2,3],
                                                         "weights":["uniform","distance"]})
                       ]
models_train_val_score, models_best_params, best_index, test_score, ax = ms.models_validation(X, y,
                                                                                     model_paramGrid_list,
                                                                                     plot=True)


###### Multiple datasets, in dataset_list
dataset_list = [(X1,y1),(X2,y2),(X3,y3)]

# datasets_hyperparameter_validation
model = DecisionTreeRegressor()
hyperparameter = "max_leaf_nodes"
hyperparameter_values = [2,3,4]
(datasets_train_val_score, datasets_best_hyperparameter_value,
 best_index, test_score, axes) = ms.datasets_hyperparameter_validation(dataset_list, model, hyperparameter,
                                                                    hyperparameter_values, plot=True,
                                                                    xvalues=["D1","D2","D3"])

# datasets_hyperparameters_validation
model = DecisionTreeRegressor()
param_grid = {"max_leaf_nodes":[2,3,4], "max_features":[None,"sqrt"]}
(datasets_train_val_score, datasets_best_params,
 best_index, test_score, ax) = ms.datasets_hyperparameters_validation(dataset_list, model, param_grid,
                                                                     plot=True, xvalues=["D1","D2","D3"])

# datasets_models_validation
model_paramGrid_list = [
                         ("DT", DecisionTreeRegressor(), {"max_leaf_nodes":[2,3,4],
                                                          "max_features":[None,"sqrt"]}),
                         ("kNN", KNeighborsRegressor(), {"n_neighbors":[1,2,3],
                                                         "weights":["uniform","distance"]})
                       ]                       
(datasets_train_val_score, datasets_best_model,
 best_index, test_score, axes) = ms.datasets_models_validation(dataset_list, model_paramGrid_list,
                                                               time_series=True, plot=True,
                                                               xvalues=["D1","D2","D3"])
```

## References
- [numpy](https://numpy.org/), the fundamental package for scientific computing with Python.
- [matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [sklearn](https://scikit-learn.org/stable/index.html), machine Learning in Python.

## License
[MIT](https://choosealicense.com/licenses/mit/)
