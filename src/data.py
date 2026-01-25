from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.config import SEED


def load_adult_income(test_size=0.2, random_state=SEED):
    """
    Load and preprocess the Adult Income dataset from OpenML.

    Returns preprocessed train/test splits and the fitted preprocessing pipeline.
    """
    data = fetch_openml(name="adult", version=2, as_frame=True)
    X = data.data
    y = (data.target == ">50K").astype(int)

    return _prepare_data(X, y, test_size, random_state)


def load_german_credit(test_size=0.2, random_state=SEED):
    """
    Load and preprocess the German Credit dataset from OpenML.

    Returns preprocessed train/test splits and the fitted preprocessing pipeline.
    """
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    X = data.data
    y = (data.target == "good").astype(int)

    return _prepare_data(X, y, test_size, random_state)


def _prepare_data(X, y, test_size, random_state):
    """
    Shared preprocessing logic for tabular datasets.
    """
    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    numerical_features = X.select_dtypes(exclude=["object", "category"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor
