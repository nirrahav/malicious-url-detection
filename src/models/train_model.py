import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import (
    extract_basic_features,
    fit_char_tfidf_vectorizer,
    transform_char_tfidf_features,
    fit_token_tfidf_vectorizer,
    transform_token_tfidf_features,
)


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
        n_estimators=30,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    return model, metrics, X_train, X_test, y_train, y_test


def train_model_with_validation(
    df: pd.DataFrame,
    url_column: str = "url",
    label_column: str = "type",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    scale_struct: bool = True,
    char_max_features: int = 5000,
    token_max_features: int = 3000
):
    """
    Full pipeline: splits data, extracts features with proper leakage prevention,
    trains model with validation, and evaluates on test set.

    Args:
        df: DataFrame with URLs and labels
        url_column: Column name for URLs
        label_column: Column name for labels
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test split)
        random_state: Random state for reproducibility
        scale_struct: Whether to scale structural features
        char_max_features: Max features for char TF-IDF
        token_max_features: Max features for token TF-IDF

    Returns:
        model, scaler, vectorizers, metrics dict
    """

    # First split: separate test set
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_column]
    )

    # Second split: train and validation from train_val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the remaining data
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=df_train_val[label_column]
    )

    # Extract structural features for all sets (no fitting, so no leakage)
    X_struct_train = extract_basic_features(df_train, url_column)
    X_struct_val = extract_basic_features(df_val, url_column)
    X_struct_test = extract_basic_features(df_test, url_column)

    # Fit TF-IDF vectorizers on training data only
    char_vectorizer = fit_char_tfidf_vectorizer(df_train, url_column, char_max_features)
    token_vectorizer = fit_token_tfidf_vectorizer(df_train, url_column, token_max_features)

    # Transform all sets
    X_char_train = transform_char_tfidf_features(df_train, char_vectorizer, url_column)
    X_char_val = transform_char_tfidf_features(df_val, char_vectorizer, url_column)
    X_char_test = transform_char_tfidf_features(df_test, char_vectorizer, url_column)

    X_token_train = transform_token_tfidf_features(df_train, token_vectorizer, url_column)
    X_token_val = transform_token_tfidf_features(df_val, token_vectorizer, url_column)
    X_token_test = transform_token_tfidf_features(df_test, token_vectorizer, url_column)

    # Prepare features
    X_train, scaler = prepare_features(X_struct_train, X_char_train, X_token_train, scale_struct)
    X_val, _ = prepare_features(X_struct_val, X_char_val, X_token_val, scale_struct)
    X_test_final, _ = prepare_features(X_struct_test, X_char_test, X_token_test, scale_struct)

    # Get labels
    y_train = df_train[label_column].values
    y_val = df_val[label_column].values
    y_test = df_test[label_column].values

    # Train model
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=20,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluate on train (for overfitting check)
    y_train_pred = model.predict(X_train)
    train_metrics = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "report": classification_report(y_train, y_train_pred, output_dict=True)
    }

    # Evaluate on validation
    y_val_pred = model.predict(X_val)
    val_metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "report": classification_report(y_val, y_val_pred, output_dict=True)
    }

    # Evaluate on test
    y_test_pred = model.predict(X_test_final)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "report": classification_report(y_test, y_test_pred, output_dict=True)
    }

    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics
    }

    vectorizers = {
        "char": char_vectorizer,
        "token": token_vectorizer
    }

    return model, scaler, vectorizers, metrics


def save_model(
    model,
    scaler=None,
    vectorizers=None,
    model_path: str = "outputs/models/model.pkl",
    scaler_path: str = "outputs/models/scaler.pkl",
    char_vectorizer_path: str = "outputs/models/char_vectorizer.pkl",
    token_vectorizer_path: str = "outputs/models/token_vectorizer.pkl"
):
    """
    Saves the trained model, scaler, and vectorizers.

    Args:
        model: Trained model
        scaler: Fitted scaler (optional)
        vectorizers: Dict with fitted vectorizers (optional)
    """

    joblib.dump(model, model_path)

    if scaler is not None:
        joblib.dump(scaler, scaler_path)

    if vectorizers is not None:
        if "char" in vectorizers:
            joblib.dump(vectorizers["char"], char_vectorizer_path)
        if "token" in vectorizers:
            joblib.dump(vectorizers["token"], token_vectorizer_path)