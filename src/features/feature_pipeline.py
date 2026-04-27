import pandas as pd

from src.features.structural_features import (
    get_url_length,
    get_domain_length,
    get_path_length,
    get_query_length,
    get_num_dots,
    get_num_hyphens,
    get_num_underscores,
    get_num_slashes,
    get_num_question_marks,
    get_num_equal_signs,
    get_num_ampersands,
    get_num_digits,
    get_num_letters,
    get_digit_ratio,
    get_special_char_ratio,
)


def extract_basic_features(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    """
    Extracts basic structural features from URL strings.

    Args:
        df (pd.DataFrame): Input dataframe containing URLs.
        url_column (str): Name of the column containing the URLs.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """

    features_df = pd.DataFrame()

    features_df["url_length"] = df[url_column].apply(get_url_length)
    features_df["domain_length"] = df[url_column].apply(get_domain_length)
    features_df["path_length"] = df[url_column].apply(get_path_length)
    features_df["query_length"] = df[url_column].apply(get_query_length)

    features_df["num_dots"] = df[url_column].apply(get_num_dots)
    features_df["num_hyphens"] = df[url_column].apply(get_num_hyphens)
    features_df["num_underscores"] = df[url_column].apply(get_num_underscores)
    features_df["num_slashes"] = df[url_column].apply(get_num_slashes)
    features_df["num_question_marks"] = df[url_column].apply(get_num_question_marks)
    features_df["num_equal_signs"] = df[url_column].apply(get_num_equal_signs)
    features_df["num_ampersands"] = df[url_column].apply(get_num_ampersands)

    features_df["num_digits"] = df[url_column].apply(get_num_digits)
    features_df["num_letters"] = df[url_column].apply(get_num_letters)
    features_df["digit_ratio"] = df[url_column].apply(get_digit_ratio)
    features_df["special_char_ratio"] = df[url_column].apply(get_special_char_ratio)

    return features_df