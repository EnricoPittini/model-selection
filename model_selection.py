"""
Module for the selection of machine learning models.

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

This module deeply uses the **numpy** library. Is built on the top of it. In fact, the datasets are represented as np.array.
Moreover, the plots are made using the **matplotlib** library. In addition, is built on the top of the **sklearn** module:
- the machine learning models are represented as sklearn models (i.e. sklearn estimators);
- under the hood, the selection is performed using the grid search cross validation provided by sklearn (i.e.
GridSearchCV);
- several other operations are done using the functionalities provided by sklearn.

This module, besides the model selection functions, contains also some utilities:
- PolynomialRegression class;
- some utility functions.

"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression




#----------------------------------------------------------------------------------------------------------------------------
# POLYNOMIAL REGRESSOR MODEL

# from sklearn.base import BaseEstimator
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegression(BaseEstimator):
    """
    Polynomial regression model.

    It's a sklearn model: it's compliant to the sklearn estimators interface.
    `Example <https://scikit-learn.org/stable/developers/develop.html>`_

    Parameters
    ----------
    degree: int
        Degree to apply for the polynomial transformation.

    Notes
    ----------
    The polynomial transformation is performed using the sklearn PolynomialFeatures.
    """

    def __init__(self, degree=1):
        self.degree=degree

    def fit(self, X, y):
        self.poly_transformer = PolynomialFeatures(self.degree, include_bias=False)
        self.poly_transformer.fit(X)
        X = self.poly_transformer.transform(X)
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X,y)
        return self

    def predict(self, X):
        X = self.poly_transformer.transform(X)
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"degree": self.degree}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




#----------------------------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS


def split_X_y(df, y_col=None, scale_y=True):
    """
    Split the given DataFrame in X and y.

    X is a matrix which contains the explanatory variables of `df`, y is a vector which contains the response variable of
    `df` (i.e. the variable that is the target of prediction analysis tasks).
    Optionally, the values in y can be scaled.

    Parameters
    ----------
    df: pd.DataFrame
    y_col: str
        Indicates which is the `df` column that is the response feature.
        If is None, the last `df` column is considered.
    scale_y: bool
        Indicates wheter scale or not the values in y.

    Returns
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of `df`.
    y: np.array
        Mono dimensional np.array, containing the response feature of `df`.

    Notes
    ----------
    The scaling of the values in y is performed using the sklearn MinMaxScaler.
    """
    if y_col is None:
        y_col = df.columns[-1]

    y = df[y_col].values # Numpy vector y
    X = df.drop([y_col],axis=1).values # Numpy matrix X

    if scale_y: # Scale the y
        scaler=MinMaxScaler()
        scaler.fit(y.reshape(y.shape[0],1))
        y = scaler.transform(y.reshape(y.shape[0],1)).reshape(y.shape[0],)

    return X,y


def compute_train_val_test(X, y, model, scale=False, test_size=0.2, time_series=False, random_state=123, n_folds=5,
                           regr=True):
    """
    Compute the training-validation-test scores for the given model on the given dataset.

    The training and test scores are computed simply splitting the dataset in the training and test sets. The validation
    score is performed applying the cross validation on the training set.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model for which computes the scores.
    scale: bool
        Indicates wheter scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. is indexed by days).
        (This affects the computing of the scores).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.

    Returns
    ----------
    train_score: float
    val_score: float
    test_score: float

    Notes
    ----------
    - If `regr` is True, the returned scores are errors, computed using MSE (i.e. Mean Squared Error).
      Otherwise, the returned scores are accuracy measures.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting the dataset in two contiguous
      parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    if regr:
        scoring="neg_mean_squared_error"
    else:
        scoring="accuracy"

    # Split in training e test.
    if not time_series : # Random splitting (not time series)
        X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else: # time series splitting
        train_len = int(X.shape[0]*(1-test_size))
        X_train_80 = X[:train_len]
        y_train_80 = y[:train_len]
        X_test =   X[train_len:]
        y_test = y[train_len:]

    if(scale): # Scale the features in X
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    # Cross validation
    if not time_series: # k-fold cross validation
        cv = n_folds
    else: # cross validation for time series
        cv = TimeSeriesSplit(n_splits = n_folds)
    scores = cross_val_score(model, X_train_80, y_train_80, cv=cv, scoring=scoring)
    val_score = scores.mean() # validation score
    if regr:
        val_score = -val_score

    model.fit(X_train_80,y_train_80) # Fit the model using all the training

    # Compute training and test scores
    train_score=0
    test_score=0
    if regr:
        train_score = mean_squared_error(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_score = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))
    else:
        train_score = accuracy_score(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_score = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))

    return train_score, val_score, test_score # Return a triple


def compute_bias_variance_error(X, y, model, scale=False, N_TESTS = 20, sample_size=0.67):
    """
    Compute the bias^2-variance-error scores for the given model on the given dataset.

    These measures are computed in an approximately way, using `N_TESTS` random samples of size `sample_size` from the
    dataset.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model for which computes the scores.
    scale: bool
        Indicates wheter scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    N_TESTS: int
        Number of samples that are made to compute the measures.
    sample_size: float
        Decimal number between 0 and 1, which indicates the proportion of the sample.

    Returns
    ----------
    bias: float
    variance: float
    error: float
    """

    # Scale the features in `X`
    if(scale):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Vector 'vector_ypred': at the end it will be a matrix with as many rows as `N_TESTS` (each row correspond to a sample)
    # and as many columns as the instances in `X` (each column is a point of the dataset).
    # Row 'i' --> there are the predictions made by the model on the sample 'i' using all the dataset points.
    # Column 'j' --> there are the predictions made by the model on the point 'j' using all the `N_TESTS` samples.
    vector_ypred = []

    # Iterate through N_TESTS. At each iteration extract a new sample and fit the model on it.
    for i in range(N_TESTS):
        # Extract a new sample (sample 'i')
        Xs, ys = resample(X,y, n_samples=int(sample_size*len(y)) )

        # Fit the model on this sample 'i'
        model.fit(Xs,ys)

        # Add the predictions made by the model on all the dataset points
        vector_ypred.append(list(model.predict(X)))

    vector_ypred = np.array(vector_ypred) # Transform to numpy

    # Vector that has as many elements as the dataset points, and for each of them has the associated bias^2 computed on the
    # `N_TEST` samples.
    vector_bias = (y - np.mean(vector_ypred, axis=0))**2

    # Vector that has as many elements as the dataset points, and for each of them has the associated variance computed on the
    # `N_TEST` samples.
    vector_variance = np.var(vector_ypred, axis=0)

    # Vector that has as many elements as the dataset points, and for each of them has the associated error computed on the
    # `N_TEST` samples.
    vector_error = np.sum((vector_ypred - y)**2, axis=0)/N_TESTS

    bias = np.mean(vector_bias) # Total bias^2 of the model
    variance = np.mean(vector_variance) # Total variance of the model
    error = np.mean(vector_error) # Total error of the model

    return bias,variance,error # Return a triple


def plot_predictions(X, y, model, scale=False, test_size=0.2, plot_type=0, xvalues=None, xlabel="Index",
                     title="Actual vs Predicted values", figsize=(6,6)):
    """
    Plot the predictions made by the given model on the given dataset, versus its actual values.

    The dataset is split in training-test sets: the former is used to train the `model`, on the latter are made the
    predictions.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model used to make the predictions.
    scale: bool
        Indicates wheter scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    plot_type: int
        Indicates the type of the plot.
            - 0 -> In the same plot are drawn two different curves: the first has on the x axis `xvalues` and on the y axis
                   the actual values (i.e. `y`); the second has on the x axis `xvalues` and on the y axis the computed
                   predicted values.
            - 1 -> On the x axis are put the actual values, on the y axis the predicted ones.
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis.
        (It's used only if `plot_type` is 0).
    xlabel: str
        Label of the x axis.
        (It's used only if `plot_type` is 0).
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    matplotlib.axes.Axes
        The matplotlib Axes where it has been made the plot.

    Notes
    ----------
    The splitting of the datasets in training-test sets, is simply made dividing the dataset in two contigous sequences. I.e.
    is the same technique used usually when the dataset is a time series dataset.
    (This is done in order to simplify the visualization).
    For this reason, typically this function is applied on time series datasets.
    """

    train_len = int(X.shape[0]*(1-test_size))
    X_train_80 = X[:train_len]
    y_train_80 = y[:train_len]
    X_test =   X[train_len:]
    y_test = y[train_len:]

    if(scale): # Scale the features in X
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    model.fit(X_train_80,y_train_80) # Fit using all the training set

    predictions = model.predict(X_test)

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type==0:
        if xvalues is None:
            xvalues=range(len(X))
        ax.plot(xvalues,y, 'o:', label='actual values')
        ax.plot(xvalues[train_len:],predictions, 'o:', label='predicted values')
        ax.legend()
    elif plot_type==1:
        ax.plot(y[train_len:],predictions,'o')
        ax.plot([0, 1], [0, 1], 'r-',transform=ax.transAxes)
        xlabel="Actual values"
        ax.set_ylabel("Predicted values")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid()

    return ax


def _plot_TrainVal_values(xvalues, train_val_scores, plot_train, xlabel, title, figsize=(6,6), bar=False):
    """
    Plot the given list of training-validation scores.

    This function is an auxiliary function for the model selection functions. It's meant to be private in the
    module.

    Parameters
    ----------
    xvalues: list (in general iterable)
        Values to put in the x axis of the plot.
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        Basically, is a list of training-validation scores.
    plot_train: bool
        Indicates wheter plot also the training scores or only the validation ones.
    xlabel: str
        Label of the x axis.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.
    bar: bool
        Indicates wheter plot the scores using bars or points.
        If `bar` it's True, `xvalues` must contains string (i.e. labels).
    Returns
    ----------
    matplotlib.axes.Axes
        The matplotlib Axes where it has been made the plot.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if not bar: # Points
        if plot_train: # Plot also the training scores
            ax.plot(xvalues,train_val_scores[:,0], 'o:', label='Train')
        ax.plot(xvalues,train_val_scores[:,1], 'o:', label='Validation') # Validation scores
    else: # Bars
        if plot_train: # Plot also the training scores
            x = np.arange(len(xvalues))  # The label locations
            width = 0.35  # The width of the bars
            ax.bar(x-width/2,train_val_scores[:,0], width=width, label='Train')
            ax.bar(x+width/2,train_val_scores[:,1], width=width, label='Validation') # Validation scores
            ax.set_xticks(x)
            ax.set_xticklabels(xvalues)
        else:
            ax.bar(xvalues,train_val_scores[:,1],label='Validation')


    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()

    return ax




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS THAT PERFORM THE MODEL SELECTION WITH RESPECT TO A SINGLE DATASET


def hyperparameter_validation(X, y, model, hyperparameter, hyperparameter_values, scale=False, test_size=0.2,
                              time_series=False, random_state=123, n_folds=5, regr=True, plot=False, plot_train=False,
                              xvalues=None, xlabel=None, title="Hyperparameter validation", figsize=(6,6)):
    """
    Select the best value for the specified hyperparameter of the specified model on the given dataset.

    In other words, perform the tuning of the `hyperparameter`, among the values in `hyperparameter_values`.

    This selection is made using the validation score (i.e. the best hyperparameter value is the one with best validation
    score).
    The validation score is computed splitting the dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Optionally, the validation scores of the `hyperparameter_values` can be plotted, making a graphical visualization of the
    selection.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified `hyperparameter`.
    hyperparameter: str
        The name of the hyperparameter that has to be validated.
    hyperparameter_values: list
        List of values for `hyperparameter` that have to be taken into account in the selection.
    scale: bool
        Indicates wheter to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. is indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates wheter to plot or not the validation score values.
    plot_train: bool
        Indicates wheter to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis.
    xlabel: str
        Label of the x axis.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as the values in `hyperparameter_values` (i.e. values to test).
    best_index: int
        Index of `hyperparameter_values` that indicates which is the best hyperparameter value.
    test_score: float
        Test score associated to the best hyperparameter value.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where it has been made the plot.
        If `plot` is False, then it is None.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      hyperparameter is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best hyperparameter is the one with associated the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting the dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    param_grid = {hyperparameter:hyperparameter_values} # Create the hyperparameter grid
    # Call the function for the validation of an arbitrary number of hyperparameters
    params, train_val_scores, best_index, test_score = hyperparameters_validation(X, y, model, param_grid, scale=scale,
                                                                                  test_size=test_size,
                                                                                  time_series=time_series,
                                                                                  random_state=random_state, n_folds=n_folds,
                                                                                  regr=regr)

    ax = None

    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = hyperparameter_values
        if not xlabel: # Default label on the x axis
            xlabel = hyperparameter
        ax = _plot_TrainVal_values(xvalues, train_val_scores, plot_train, xlabel, title, figsize)

    return train_val_scores, best_index, test_score, ax


def hyperparameters_validation(X, y, model, param_grid, scale=False, test_size=0.2, time_series=False, random_state=123,
                               n_folds=5, regr=True):
    """
    Select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    In other words, perform the tuning of multiple hyperparameters.
    The parameter `param_grid` is a dictionary that indicates which are the specified hyperparameters and what are the
    associated values to test.

    Are tested all the possible combinations of values, in an exaustive way (i.e. grid search).

    This selection is made using the validation score (i.e. the best combination of hyperparameters values is the one with
    best validation score).
    The validation score is computed splitting the dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified hyperparameters.
    param_grid: dict
        Dictionary which has, as keys, the names of the specified hyperparameters and, as values, the associated list of
        values to test.
    scale: bool
        Indicates wheter to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. is indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.

    Returns
    ----------
    params: list
        List which enumerates all the possible combinations of hyperparameters values.
        It's a list of dictionaries: each dictionary represents a specific combination of hyperparameters values. (It's a
        dictionary with keys the hyperparameters names and with values the specific associated values of that combination).
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as possible combinations of hyperparameters values.
        (It has as many rows as the elements of `params`).
    best_index: int
        Index of `params` that indicates which is the best combination of hyperparameters values.
    test_score: float
        Test score associated to the best combination of hyperparameters values.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      combination of hyperparameters values is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best combination of hyperparameters values is the
      one with associated the maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting the dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    if regr:
        scoring="neg_mean_squared_error"
    else:
        scoring="accuracy"

    # Split in training-test sets
    if not time_series : # Random splitting
        X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else: # Time series splitting
        train_len = int(X.shape[0]*(1-test_size))
        X_train_80 = X[:train_len]
        y_train_80 = y[:train_len]
        X_test =   X[train_len:]
        y_test = y[train_len:]

    if(scale): # Scale the features in `X`
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    # Cross validation strategy
    if not time_series: # The strategy is the classical k-fold cross validation
        cv = n_folds
    else: # Time series cross validation strategy
        cv = TimeSeriesSplit(n_splits = n_folds)

    # Grid search
    grid_search = GridSearchCV(model,param_grid,scoring=scoring,cv=cv,return_train_score=True)
    grid_search.fit(X_train_80,y_train_80)

    params = grid_search.cv_results_["params"] # List of all the possible combinations of hyperparameters values
    # List where for all the possible combinations of hyperparameters values there is the associated training score
    train_scores = grid_search.cv_results_["mean_train_score"]
    # List where for all the possible combinations of hyperparameters values there is the associated validation score
    val_scores = grid_search.cv_results_["mean_test_score"]
    # Index of `params`, correspondent to the best combination of hyperparameters values
    best_index = grid_search.best_index_
    # Model with the best combination of hyperparameters values
    best_model = grid_search.best_estimator_

    if regr: # The scores are negative: moltiply by -1
        train_scores = train_scores*(-1)
        val_scores = val_scores*(-1)
    train_val_scores = np.concatenate((train_scores.reshape(-1,1), val_scores.reshape(-1,1)), axis=1)

    # Fir the best model on all the training set
    best_model.fit(X_train_80,y_train_80)

    # Compute the test score of the best model
    test_score=0
    if regr:
        test_score = mean_squared_error(y_true=y_test, y_pred=best_model.predict(X_test))
    else:
        test_score = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))

    return params, train_val_scores, best_index, test_score


def models_validation(X, y, model_paramGrid_list, scale_list=None, test_size=0.2, time_series=False, random_state=123,
                      n_folds=5, regr=True, plot=False, plot_train=False, xvalues=None, xlabel="Models",
                      title="Models validation", figsize=(6,6)):
    """
    Select the best model for the given dataset.

    The parameter `model_paramGrid_list` is the list of the models to test. It also contains, for each model, the grid of
    the hyperparameters that have to be tested on that model (i.e. the grid which contains the values to test for each
    specified hyperparameter of the model).
    (That grid has the same structure of the `param_grid` parameter of the function `hyperparameters_validation`. See
    `hyperparameters_validation`).

    For each specified model, is selected the best combination of hyperparameters values in an exaustive way (i.e. grid
    search).
    Actually, it's used the function `hyperparameters_validation`.
    (See `hyperparameters_validation`).

    The selection of the best model is made using the validation score (i.e. the best model is the one with best validation
    score).
    The validation score is computed splitting the dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Optionally, the validation scores of the different models can be plotted, making a graphical visualization of the
    selection.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model_paramGrid_list: list
        List that specifies the models and the grid of hyperparameters to be tested.
        It's a list of triples (i.e. tuples), where each triple represent a model:
            - the first element is a string, which is a mnemonic name of that model;
            - the second element is the sklearn model;
            - the third element is the grid of hyperparameters to test for that model. It's a dictionary, with the same
              structure of parameter `param_grid` of the function `hyperparameters_validation`.
    scale_list: list or bool
        List of booleans, which has as many elements as the models to test (i.e. as the elements of the
        `model_paramGrid_list` list).
        This list indicates, for each different model, if the features in `X` has to be scaled or not.
        `scale_list` can be None or False: in this case the `X` features aren't scaled for any model. `scale_list` can be
        True: in this case the `X` features are scaled for all the models.
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. is indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates wheter to plot or not the validation score values.
    plot_train: bool
        Indicates wheter to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis.
    xlabel: str
        Label of the x axis.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    models_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as the models to test (i.e. as the `model_paramGrid_list` list).
    models_best_params: list
        List which indicates, for each model, the best combination of hyperparameters values for that model.
        It has as many elements as the models to test (i.e. as the elements of the `model_paramGrid_list` list), and it
        contains dictionaries: each dictionary represents the best combination of hyperparameters values for the associated
        model.
    best_index: int
        Index of `model_paramGrid_list` that indicates which is the best model.
    test_score: float
        Test score associated to the best model.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where it has been made the plot.
        If `plot` is False, then it is None.

    See also
    ----------
    hyperparameters_validation:
        select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      model is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best model is the one with associated the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting the dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    if not scale_list: # `scale_list` is either None or False
        scale_list = [False]*len(model_paramGrid_list)
    elif scale_list is True: # `scale_list` is True
        scale_list = [True]*len(model_paramGrid_list)

    # Numpy matrix (np.array) with as many rows as the models and two columns, one for the training scores and the other for
    # the validation scores. At the beginning it a list of tuples.
    models_train_val_score = []
    # List which has as many elements as the models: for each model there is the dictionary of the best combination of
    # hyperparameters values.
    models_best_params = []
    # List which has as many elements as the models: for each model there is the test score (associated at the best
    # combination of hyperparameters values).
    models_test_score = []

    for i,triple in enumerate(model_paramGrid_list): # Iterate through all the cuples model-param_grid
        model,param_grid = triple[1:]

        # Apply the grid search on model-param_grid
        params, train_val_scores, best_index, test_score = hyperparameters_validation(X, y, model, param_grid,
                                                                                      scale=scale_list[i],
                                                                                      test_size=test_size,
                                                                                      time_series=time_series,
                                                                                      random_state=random_state,
                                                                                      n_folds=n_folds, regr=regr)

        models_train_val_score.append(tuple(train_val_scores[best_index])) # Add row for that model
        models_best_params.append(params[best_index]) # Add element for that model
        models_test_score.append(test_score) # Add element for that model

    models_train_val_score = np.array(models_train_val_score) # Transform in numpy matrix (i.e. np.array)

    # Find the best index (i.e. the best model)
    if regr:
        best_index = np.argmin(models_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(models_train_val_score,axis=0)[1]

    # Test score of the best model
    test_score = models_test_score[best_index]

    ax = None
    if(plot): # Make the plot
        if not xvalues: # Default values for the x axis
            xvalues = [model_paramGrid_list[i][0] for i in range(len(model_paramGrid_list))]
        ax = _plot_TrainVal_values(xvalues, models_train_val_score, plot_train, xlabel, title, figsize, bar=True)

    return models_train_val_score, models_best_params, best_index, test_score, ax




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS THAT PERFORM THE MODEL SELECTION WITH RESPECT TO MULTIPLE DATASETS


def datasets_hyperparameter_validation(dataset_list, model, hyperparameter, hyperparameter_values, scale=False,
                                       test_size=0.2, time_series=False, random_state=123, n_folds=5, regr=True, plot=False,
                                       plot_train=False, xvalues=None, xlabel="Datasets", title="Datasets validation",
                                       figsize=(6,6) ,verbose=False, figsize_verbose=(6,6)):
    """
    Select the best dataset and the best value for the specified hyperparameter of the specified model (i.e. select the best
    couple dataset-hyperparameter value).

    For each dataset in `dataset_list`, are tested all the specified values `hyperparameter_values` for the specified
    `hyperparameter` of `model`.
    In other words, on each dataset is performed the tuning of `hyperparameter`: in fact, on each dataset, is applied the
    function `hyperparameter_validation`. (See `hyperparameter_validation`).
    At the end is selected the best couple dataset-hyperparameter value.

    Despite the fact that is selected a couple dataset-hyperparameter value, the main viewpoint is focused with respect to
    the datasets. It's a validation focused on the datasets.
    In fact, first of all, for each dataset it's performed the hyperparameter tuning: in this way is selected the best value
    and its relative score is associated to the dataset (i.e. it's the dataset score). (In other words, on each dataset is
    applied the function `hyperparameter_validation`). And, after that, is selected the best dataset.
    It's a selection on two levels.

    This selection is made using the validation score (i.e. the best couple dataset-hyperparameter value is the one with best
    validation score).
    The validation score is computed splitting each dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection. This is the 'main' plot.
    Moreover, still optionally, can be done the 'secondary' plots: for each dataset, the validation scores of the
    `hyperparameter_values` are plotted, making a graphical visualization of the hyperparameter tuning. (As the plot of
    `hyperparameter_validation`).

    Parameters
    ----------
    dataset_list: list
        List of couple, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified `hyperparameter`.
    hyperparameter: str
        The name of the hyperparameter that has to be validated.
    hyperparameter_values: list
        List of values for `hyperparameter` that have to be taken into account in the selection.
    scale: bool
        Indicates wheter to scale or not the features in 'X' (for all the datasets).
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series dataset (i.e. are indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates wheter to plot or not the validation score values of the datasets (i.e. 'main' plot).
    plot_train: bool
        Indicates wheter to plot also the training scores (both in the 'main' and 'secondary' plots).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the 'main' plot.
    xlabel: str
        Label of the x axis of the 'main' plot.
    title: str
        Title of the 'main' plot.
    figsize: tuple
        Two dimensions of the 'main' plot.
    verbose: bool
        If True, for each dataset are plotted the validation scores of the hyperparameter tuning (i.e. 'secondary' plots).
        (See 'hyperparameter_validation').
    figsize_verbose: tuple
        Two dimensions of the 'secondary' plots.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as the datasets to test, i.e. as the elements of `dataset_list`.
    datasets_best_hyperparameter_value: list
        List which has as many elements as the datasets (i.e. as the elements of `dataset_list`). For each dataset, it
        contains the best `hyperparameter` value on that dataset.
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated to the best couple dataset-hyperparameter value.
    axes: list
        List of the matplotlib Axes where are made the plots.
        Firstly, are put the 'secondary' plots (if any). And, as last, is put the 'main' plot (if any).
        If it hasn't been made any plot, `axes` is an empty list.

    See also
    ----------
    hyperparameter_validation:
        select the best value for the specified hyperparameter of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-hyperparameter value is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one with associated the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting each dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and as columns it has the training and validation
    # scores. At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best hyperparameter value
    datasets_best_hyperparameter_value = []
    # List which contains, for each dataset, its test score (associated to the best hyperparameter value)
    datasets_test_score = []
    # List of axes
    axes = []

    for i,dataset in enumerate(dataset_list): # Iterate through all the datasets

        X,y = dataset

        # Perform the hyperparameter tuning on the current dataset
        train_val_scores, best_index, test_score, ax = hyperparameter_validation(X, y, model, hyperparameter,
                                hyperparameter_values, scale=scale, test_size=test_size, time_series=time_series,
                                random_state=random_state, n_folds=n_folds, regr=regr, plot=verbose, plot_train=plot_train,
                                xvalues=hyperparameter_values, xlabel=hyperparameter,
                                title="Dataset "+str(i)+" : hyperparameter validation", figsize=figsize_verbose)

        datasets_train_val_score.append(tuple(train_val_scores[best_index,:])) # Add the row related to that dataset
        datasets_best_hyperparameter_value.append(hyperparameter_values[best_index]) # Add the element related to that dataset
        datasets_test_score.append(test_score) # Add the row related to that dataset
        if ax:
            axes.append(ax)

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform to numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-hyperparameter value)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-hyperparameter value
    test_score = datasets_test_score[best_index]

    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)
        axes.append(ax)

    return datasets_train_val_score, datasets_best_hyperparameter_value, best_index, test_score, axes


def datasets_hyperparameters_validation(dataset_list, model, param_grid, scale=False, test_size=0.2, time_series=False,
                                        random_state=123, n_folds=5, regr=True, plot=False, plot_train=False, xvalues=None,
                                        xlabel="Datasets", title="Datasets validation",figsize=(6,6)):
    """
    Select the best dataset and the best combination of values for the specified hyperparameters of the specified model (i.e.
    select the best couple dataset-combination of hyperparameters values).

    For each dataset in `dataset_list`, are tested all the possible combinations of the hyperparameters values (specified
    with `param_grid`) for `model`.
    In other words, on each dataset is performed the tuning of the specified hyperparameters, in an exaustive way: in fact,
    on each dataset, is applied the function `hyperparameters_validation`. (See `hyperparameters_validation`).
    At the end, is selected the best couple dataset-combination of hyperparameters values.

    Despite the fact that is selected a couple dataset-combination of hyperparameters values, the main viewpoint is focused
    with respect to the datasets. It's a validation focused on the datasets.
    In fact, first of all, for each dataset it's performed the hyperparameters tuning: in this way is selected the best
    combination of values and its relative score is associated to the dataset (i.e. it's the dataset score). (In other words,
    on each dataset is applied the function `hyperparameters_validation`). And, after that, is selected the best dataset.
    It's a selection on two levels.

    This selection is made using the validation score (i.e. the best couple dataset-combination of hyperparameters values, is
    the one with best validation score).
    The validation score is computed splitting each dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection.

    Parameters
    ----------
    dataset_list: list
        List of couple, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified hyperparameters.
    param_grid: str
        Dictionary which has, as keys, the names of the specified hyperparameters and, as values, the associated list of
        values to test.
    scale: bool
        Indicates wheter to scale or not the features in 'X' (for all the datasets).
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series dataset (i.e. are indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates wheter to plot or not the validation score values of the datasets.
    plot_train: bool
        Indicates wheter to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis.
    xlabel: str
        Label of the x axis.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as the datasets to test, i.e. as the elements of `dataset_list`.
    datasets_best_params: list
        List which has as many elements as the datasets (i.e. as the elements of `dataset_list`). For each dataset, it
        contains the best combination of hyperparameters values on that dataset.
        Each combination is represented as a dictionary, with keys the hyperparameters names and values the associated
        values.
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated to the best couple dataset-combination of hyperparameters values.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where it has been made the plot.

    See also
    ----------
    hyperparameters_validation:
        select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-combination of hyperparameters values, is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one with associated the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting each dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and as columns it has the training and validation
    # scores. At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best combination of hyperparameters values (i.e. a dictionary)
    datasets_best_params = []
    # List which contains, for each dataset, its test score (associated to the best combination of hyperparameters values)
    datasets_test_score = []

    for X,y in dataset_list: # Iterate through all the datasets

        # Perform the exaustive hyperparameters tuning on the current dataset
        params, train_val_scores, best_index, test_score = hyperparameters_validation(X, y, model, param_grid, scale=scale,
                                                                                      test_size=test_size,
                                                                                      time_series=time_series,
                                                                                      random_state=random_state,
                                                                                      n_folds=n_folds, regr=regr)

        datasets_train_val_score.append(tuple(train_val_scores[best_index,:])) # Add the row related to that dataset
        datasets_best_params.append(params[best_index]) # Add the element related to that dataset
        datasets_test_score.append(test_score) # Add the row related to that dataset

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform to numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-combination of hyperparameters
    # values)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-combination of hyperparameters values
    test_score = datasets_test_score[best_index]

    ax = None
    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)

    return datasets_train_val_score, datasets_best_params, best_index, test_score, ax


def datasets_models_validation(dataset_list, model_paramGrid_list, scale_list=None, test_size=0.2, time_series=False,
                               random_state=123, n_folds=5, regr=True, plot=False, plot_train=False, xvalues=None,
                               xlabel="Datasets", title="Datasets validation", figsize=(6,6) ,verbose=False,
                               figsize_verbose=(6,6)):
    """
    Select the best dataset and the best model (i.e. select the best couple dataset-model).

    For each dataset in `dataset_list`, are tested all the model in `model_paramGrid_list`: each model is tested performing
    an exaustive tuning of the specified hyperparameters. In fact, `model_paramGrid_list` also contains, for each model, the
    grid of the hyperparameters that have to be tested on that model (i.e. the grid which contains the values to test for
    each specified hyperparameter of the model).
    In other words, on each dataset is performed the selection of the best model: in fact, on each dataset, is applied the
    function `models_validation`. (See `models_validation`).
    At the end is selected the best couple dataset-model.

    Despite the fact that is selected a couple dataset-model, the main viewpoint is focused with respect to the datasets.
    It's a validation focused on the datasets.
    In fact, first of all, for each dataset it's performed the model selection: in this way is selected the best model
    and its relative score is associated to the dataset (i.e. it's the dataset score). (In other words, on each dataset is
    applied the function `models_validation`). And, after that, is selected the best dataset.
    It's a selection on two levels.

    This selection is made using the validation score (i.e. the best couple dataset-model is the one with best validation
    score).
    The validation score is computed splitting each dataset in training-test sets and then applying the cross validation on
    the training set.
    Additionally, are also computed the training and test scores.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection. This is the 'main' plot.
    Moreover, still optionally, can be done the 'secondary' plots: for each dataset, the validation scores of the models are
    plotted, making a graphical visualization of the models selection. (As the plot of `models_validation`).

    Parameters
    ----------
    dataset_list: list
        List of couple, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model_paramGrid_list: list
        List that specifies the models and the grid of hyperparameters to be tested.
        It's a list of triples (i.e. tuples), where each triple represent a model:
            - the first element is a string, which is a mnemonic name of that model;
            - the second element is the sklearn model;
            - the third element is the grid of hyperparameters to test for that model. It's a dictionary, with the same
              structure of parameter `param_grid` of the function `hyperparameters_validation`.
    scale_list: list or bool
        List of booleans, which has as many elements as the models to test (i.e. as the elements of the
        `model_paramGrid_list` list).
        This list indicates, for each different model, if the features in 'X' has to be scaled or not (for all the datasets).
        `scale_list` can be None or False: in this case the 'X' features aren't scaled for any model. `scale_list` can be
        True: in this case the 'X' features are scaled for all the models.
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series dataset (i.e. are indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates wheter to plot or not the validation score values of the datasets (i.e. 'main' plot).
    plot_train: bool
        Indicates wheter to plot also the training scores (both in the 'main' and 'secondary' plots).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the 'main' plot.
    xlabel: str
        Label of the x axis of the 'main' plot.
    title: str
        Title of the 'main' plot.
    figsize: tuple
        Two dimensions of the 'main' plot.
    verbose: bool
        If True, for each dataset are plotted the validation scores of the models (i.e. 'secondary' plots).
        (See 'models_validation').
    figsize_verbose: tuple
        Two dimensions of the 'secondary' plots.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        It has as many rows as the datasets to test, i.e. as the elements of `dataset_list`.
    datasets_best_model: list
        List which has as many elements as the datasets (i.e. as the elements of `dataset_list`). For each dataset, it
        contains the best model for that dataset.
        More precisely, it is a list of triple:
            - the first element is the index of `model_paramGrid_list`, indicating the best model;
            - the second element is the mnemonic name of the best model;
            - the third element is the best combination of hyperparameters values on that best model (i.e. it's a dictionary
              with keys the hyperparameters names and values their associated values).
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated to the best couple dataset-model.
    axes: list
        List of the matplotlib Axes where are made the plots.
        Firstly, are put the 'secondary' plots (if any). And, as last, is put the 'main' plot (if any).
        If it hasn't been made any plot, `axes` is an empty list.

    See also
    ----------
    models_validation: select the best model for the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-model is the one with associated the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one with associated the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classical k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply splitting each dataset in two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and as columns it has the training and validation
    # scores. At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best model. I.e. there is the triple index-model name-best combination of
    # hyperparameters values
    datasets_best_model = []
    # List which contains, for each dataset, its test score (associated to the best model)
    datasets_test_score = []
    # List of axes
    axes = []

    for i,dataset in enumerate(dataset_list): # Iterate through all the datasets

        X,y = dataset

        # Perform the models validation on the current dataset
        models_train_val_score, models_best_params, best_index, test_score, ax = models_validation(X, y,
                                                                                                   model_paramGrid_list,
                                                                                                   scale_list=scale_list,
                                                                                                   test_size=test_size,
                                                                                                   time_series=time_series,
                                                                                                   random_state=random_state,
                                                                                                   n_folds=n_folds,
                                                                                                   regr=regr, plot=verbose,
                                                                                                   plot_train=plot_train,
                                                                                                   xlabel="Models",
                                                                                                   title=("Dataset "+str(i)+
                                                                                                     " : models validation"),
                                                                                                   figsize=figsize_verbose)

        datasets_train_val_score.append(tuple(models_train_val_score[best_index,:])) # Add the row related to that dataset
        # Add the element related to that dataset
        datasets_best_model.append((best_index,model_paramGrid_list[best_index][0],models_best_params[best_index]))
        datasets_test_score.append(test_score) # Add the element related to that dataset
        if ax:
            axes.append(ax)

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform to numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-model)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-model
    test_score = datasets_test_score[best_index]

    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)
        axes.append(ax)

    return datasets_train_val_score, datasets_best_model, best_index, test_score, axes
