import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_url(url: str) -> list[str]:
    """Splits a URL string into meaningful tokens."""
    url = str(url).lower()
    tokens = re.split(r"[./\-?_=&:%]+", url)
    return [token for token in tokens if token]


def get_char_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (3, 5),
    max_features: int = 5000
) -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer using character n-grams."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features
    )


def get_token_tfidf_vectorizer(
    max_features: int = 3000
) -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer using URL tokens."""
    return TfidfVectorizer(
        tokenizer=tokenize_url,
        token_pattern=None,
        lowercase=False,
        max_features=max_features
    )