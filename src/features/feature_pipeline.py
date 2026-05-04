import pandas as pd

from src.features.structural_features import extract_features

from src.features.nlp_features import (
    get_char_tfidf_vectorizer,
    get_token_tfidf_vectorizer,
)


def extract_basic_features(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    """
    Extracts structural features from URL strings.

    Args:
        df (pd.DataFrame): Input dataframe containing URLs.
        url_column (str): Name of the column containing the URLs.

    Returns:
        pd.DataFrame: DataFrame containing extracted structural features.
    """

    features = df[url_column].astype(str).apply(extract_features)
    features_df = pd.DataFrame(features.tolist())

    return features_df


def extract_char_tfidf_features(
    df: pd.DataFrame,
    url_column: str = "url",
    max_features: int = 5000
) -> tuple[object, object]:
    """
    Extracts character-level TF-IDF features from URLs.

    Returns:
        tuple: TF-IDF sparse matrix and the fitted vectorizer.
    """
    vectorizer = get_char_tfidf_vectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(df[url_column].astype(str))

    return matrix, vectorizer


def fit_char_tfidf_vectorizer(
    df: pd.DataFrame,
    url_column: str = "url",
    max_features: int = 5000
) -> object:
    """
    Fits a character-level TF-IDF vectorizer on the given data.

    Args:
        df: DataFrame to fit on
        url_column: Column name with URLs
        max_features: Max features for vectorizer

    Returns:
        Fitted vectorizer
    """
    vectorizer = get_char_tfidf_vectorizer(max_features=max_features)
    vectorizer.fit(df[url_column].astype(str))
    return vectorizer


def transform_char_tfidf_features(
    df: pd.DataFrame,
    vectorizer: object,
    url_column: str = "url"
) -> object:
    """
    Transforms URLs using a fitted TF-IDF vectorizer.

    Args:
        df: DataFrame to transform
        vectorizer: Fitted vectorizer
        url_column: Column name with URLs

    Returns:
        Sparse matrix of features
    """
    return vectorizer.transform(df[url_column].astype(str))


def extract_token_tfidf_features(
    df: pd.DataFrame,
    url_column: str = "url",
    max_features: int = 3000
) -> tuple[object, object]:
    """
    Extracts token-level TF-IDF features from URLs.

    Returns:
        tuple: TF-IDF sparse matrix and the fitted vectorizer.
    """
    vectorizer = get_token_tfidf_vectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(df[url_column].astype(str))

    return matrix, vectorizer


def fit_token_tfidf_vectorizer(
    df: pd.DataFrame,
    url_column: str = "url",
    max_features: int = 3000
) -> object:
    """
    Fits a token-level TF-IDF vectorizer on the given data.

    Args:
        df: DataFrame to fit on
        url_column: Column name with URLs
        max_features: Max features for vectorizer

    Returns:
        Fitted vectorizer
    """
    vectorizer = get_token_tfidf_vectorizer(max_features=max_features)
    vectorizer.fit(df[url_column].astype(str))
    return vectorizer


def transform_token_tfidf_features(
    df: pd.DataFrame,
    vectorizer: object,
    url_column: str = "url"
) -> object:
    """
    Transforms URLs using a fitted TF-IDF vectorizer.

    Args:
        df: DataFrame to transform
        vectorizer: Fitted vectorizer
        url_column: Column name with URLs

    Returns:
        Sparse matrix of features
    """
    return vectorizer.transform(df[url_column].astype(str))