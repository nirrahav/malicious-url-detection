import joblib
import numpy as np

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def prepare_features(
    X_struct,
    X_char,
    X_token,
    scale_struct: bool = True
):
    """
    Combines structural and NLP features into a single matrix.

    Args:
        X_struct (pd.DataFrame): Structural features.
        X_char (sparse matrix): Character TF-IDF features.
        X_token (sparse matrix): Token TF-IDF features.
        scale_struct (bool): Whether to apply StandardScaler to structural features.

    Returns:
        sparse matrix: Combined feature matrix.
        scaler (optional): Fitted scaler if used.
    """

    scaler = None

    if scale_struct:
        scaler = StandardScaler()
        X_struct_scaled = scaler.fit_transform(X_struct)
    else:
        X_struct_scaled = X_struct.values

    X_full = hstack([X_struct_scaled, X_char, X_token])

    return X_full, scaler


def train_model(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits the data, trains a RandomForest model and evaluates it.

    Args:
        X: Feature matrix.
        y: Labels.

    Returns:
        model, metrics dict
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    return model, metrics


def save_model(
    model,
    scaler=None,
    model_path: str = "models/model.pkl",
    scaler_path: str = "models/scaler.pkl"
):
    """
    Saves the trained model and scaler.

    Args:
        model: Trained model
        scaler: Fitted scaler (optional)
    """

    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)