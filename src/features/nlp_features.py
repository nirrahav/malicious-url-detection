import re

from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_url(url: str) -> list[str]:
    """
    Splits a URL string into meaningful tokens.

    The tokenizer separates the URL by common URL delimiters such as:
    dots, slashes, hyphens, query symbols, equal signs and ampersands.
    """
    url = str(url).lower()
    tokens = re.split(r"[./\-?_=&:%]+", url)

    return [token for token in tokens if token]


def get_char_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (3, 5),
    max_features: int = 5000
) -> TfidfVectorizer:
    """
    Creates a TF-IDF vectorizer using character n-grams.

    Character n-grams are useful for URLs because malicious patterns
    may appear inside parts of words, domains or random-looking strings.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True
    )


def get_token_tfidf_vectorizer(
    max_features: int = 3000
) -> TfidfVectorizer:
    """
    Creates a TF-IDF vectorizer using URL tokens.

    Token-level TF-IDF captures meaningful URL components such as
    domain parts, path parts and query parameters.
    """
    return TfidfVectorizer(
        tokenizer=tokenize_url,
        token_pattern=None,
        max_features=max_features,
        lowercase=True
    )