# model-selection
Library for the selection of machine learning models.

There are several different functions which can perform the model selection: all of them have an intuitive interface, but
are also powerful and flexible.
In addition, almost all these functions optionally can make plots, which sum up the performed selection in a visual way.

These different functions perform the model selection in different contexts, i.e. each function is specifically meant for a
specific scenario. Certain contexts are more specific, and other are more general.
On the whole, there are six different model selection functions, divided into two main groups:
1. functions that perform the model selection with respect to a **single dataset**;
2. functions that perform the model selection with respect to **multiple datasets**.

The six functions, sorted from the most specific context to the most general one, are:
- *hyperparameter_validation*, *hyperparameters_validation*, *models_validation* (single dataset);
- *datasets_hyperparameter_validation*, *datasets_hyperparameters_validation*, *datasets_models_validation* (multiple
      datasets).

This library deeply uses the [numpy](https://numpy.org/) library. Is built on the top of it. In fact, the datasets are represented as np.array.
In addition, the plots are made using [matplotlib](https://matplotlib.org/stable/index.html). Moreover, is built on the top of the [sklearn](https://scikit-learn.org/stable/index.html) library:
    - the machine learning models are represented as sklearn models (i.e. sklearn estimators);
    - under the hood, the selection is performed using the grid search cross validation provided by sklearn (i.e.
      GridSearchCV);
    - several other operations are done using the functionalities provided by sklearn.

This library, besides the model selection functions, contains also some utilities:
    - PolynomialRegression class;
    - some utility functions.

This library is part of my bachelor thesis, do check it out the other works.
- [timeSeries-processing](https://github.com/EnricoPittini/timeSeries-processing) 
- [EEA-datasets-handler](https://github.com/EnricoPittini/EEA-datasets-handler) 
- [ILMETEO-datasets-handler](https://github.com/EnricoPittini/ILMETEO-datasets-handler) 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ml-model-selection.

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
- [pandas](https://pandas.pydata.org/) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.
- [sklearn](https://scikit-learn.org/stable/index.html), machine Learning in Python

## License
[MIT](https://choosealicense.com/licenses/mit/)
