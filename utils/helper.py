#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/6 15:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from pandas import DataFrame
from plotly.express import scatter, scatter_3d
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from time import perf_counter


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


def scatter_visualiser(data: DataFrame, categories: DataFrame = None, dims: int = 3):
    """ Visualise the data using scatter plots.
    :param data: the DataFrame containing the data
    :param categories: the DataFrame containing the categories for colouring and symbolising the data points
    :param dims: number of dimensions to reduce to if data has more than 3 dimensions (2 or 3)
    :return: a scatter plot with different colours and symbols for each category
    """
    if categories is not None:
        df = data.join(categories)
        category_name = categories.columns[0]
    else:
        df = data
        category_name = None

    dimensions = data.shape[1]

    if dimensions == 2:
        fig = scatter(
            df,
            x=data.columns[0],
            y=data.columns[1],
            color=category_name,
            symbol=category_name,
            hover_data=[data.columns[0], data.columns[1], category_name]
        ).update_layout(coloraxis_showscale=False)
    elif dimensions == 3:
        fig = scatter_3d(
            df,
            x=data.columns[0],
            y=data.columns[1],
            z=data.columns[2],
            color=category_name,
            symbol=category_name,
            hover_data=[data.columns[0], data.columns[1], data.columns[2], category_name]
        ).update_layout(coloraxis_showscale=False)
    else:
        pca = PCA(n_components=dims)
        components = pca.fit_transform(data)
        if dims == 2:
            cols: list[str] = ["PAC-X", "PAC-Y"]
        else:
            cols: list[str] = ["PAC-X", "PAC-Y", "PAC-Z"]
        df = DataFrame(components, columns=cols)
        df = df.join(categories)

        colour_name: str = str(categories.columns[0])

        if dims == 2:
            fig = scatter(
                df,
                x=cols[0],
                y=cols[1],
                color=colour_name,
                symbol=colour_name,
                hover_data=cols + ([colour_name] if colour_name else [])
            ).update_layout(coloraxis_showscale=False)
        else:
            fig = scatter_3d(
                df,
                x=cols[0],
                y=cols[1],
                z=cols[2],
                color=colour_name,
                symbol=colour_name,
                hover_data=cols + ([colour_name] if colour_name else [])
            ).update_layout(coloraxis_showscale=False)
    return fig


def data_preprocessor(selected_data: DataFrame) -> tuple[DataFrame, StandardScaler]:
    """ Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
    :param selected_data: the DataFrame containing the selected features for training
    :return: a DataFrame containing the preprocessed features
    """
    cols_num = selected_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_type = selected_data.select_dtypes(include=["object", "category"]).columns.tolist()
    # print(f"The cols filed with number: {cols_num}")
    # print(f"The cols filled with category: {cols_type}")

    # Establish a pipe to process numerical features
    pipe_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Set a list of transformers for the ColumnTransformer
    transformers = [("number", pipe_num, cols_num)]

    # Establish a pipe to process categorical features
    if cols_type:
        pipe_type = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("category", pipe_type, cols_type))

    # Establish a column transformer to process numerical and categorical features
    preprocessor = ColumnTransformer(transformers=transformers)
    # Fit and transform the data
    processed = preprocessor.fit_transform(selected_data)

    # If the processed data is a sparse matrix, convert it to a dense array
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    # Set the feature names for the processed data
    cols_names: list[str] = cols_num
    if cols_type:
        # Due to the OneHotEncoder, the feature names will be obtained throughout
        encoder = preprocessor.named_transformers_["category"]["encoder"]
        cols_names += encoder.get_feature_names_out(cols_type).tolist()

    # Convert the processed data to a DataFrame
    return DataFrame(processed, columns=cols_names), preprocessor.named_transformers_["number"]["scaler"]


def tree_model_seeker(x_train: DataFrame, y_train: DataFrame, randomness: int, cv: int = 10, n_job: int = -1):
    """ Seek the best hyperparameters for a Decision Tree Classifier using GridSearchCV.
    :param x_train: the DataFrame containing the training features
    :param y_train: the DataFrame containing the training labels
    :param randomness: the random state for reproducibility
    :param cv: the number of cross-validation folds
    :param n_job: the number of jobs to run in parallel
    :return: the best model, the best hyperparameters, and the best score
    """
    param_grid: dict[str, list] = {
        "criterion": ["gini", "entropy"],
        "max_depth": [1, 2, 3, 4, 5, None],
        "min_samples_leaf": [1, 2, 3, 4, 5]
    }
    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=randomness),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_job,
        verbose=1,
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_params_score = grid.best_score_
    return best_model, best_params, best_params_score
