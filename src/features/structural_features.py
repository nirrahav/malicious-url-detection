from urllib.parse import urlparse


def get_url_length(url: str) -> int:
    """Returns the total length of the URL."""
    return len(url)


def get_num_dots(url: str) -> int:
    """Returns the number of dots in the URL."""
    return url.count(".")


def get_num_hyphens(url: str) -> int:
    """Returns the number of hyphens in the URL."""
    return url.count("-")


def get_num_underscores(url: str) -> int:
    """Returns the number of underscores in the URL."""
    return url.count("_")


def get_num_slashes(url: str) -> int:
    """Returns the number of slashes in the URL."""
    return url.count("/")


def get_num_question_marks(url: str) -> int:
    """Returns the number of question marks in the URL."""
    return url.count("?")


def get_num_equal_signs(url: str) -> int:
    """Returns the number of equal signs in the URL."""
    return url.count("=")


def get_num_ampersands(url: str) -> int:
    """Returns the number of ampersands in the URL."""
    return url.count("&")


def get_num_digits(url: str) -> int:
    """Returns the number of digits in the URL."""
    return sum(char.isdigit() for char in url)


def get_num_letters(url: str) -> int:
    """Returns the number of alphabetic characters in the URL."""
    return sum(char.isalpha() for char in url)


def get_digit_ratio(url: str) -> float:
    """Returns the ratio of digits out of the total URL length."""
    if len(url) == 0:
        return 0.0

    return get_num_digits(url) / len(url)


def get_special_char_ratio(url: str) -> float:
    """Returns the ratio of non-alphanumeric characters out of the total URL length."""
    if len(url) == 0:
        return 0.0

    special_chars = sum(not char.isalnum() for char in url)
    return special_chars / len(url)


def get_domain_length(url: str) -> int:
    """Returns the length of the URL domain."""
    parsed_url = urlparse(url)

    domain = parsed_url.netloc

    # Handles URLs without scheme, e.g. "example.com/path"
    if not domain:
        domain = urlparse("http://" + url).netloc

    return len(domain)


def get_path_length(url: str) -> int:
    """Returns the length of the URL path."""
    parsed_url = urlparse(url)

    # Handles URLs without scheme
    if not parsed_url.netloc:
        parsed_url = urlparse("http://" + url)

    return len(parsed_url.path)


def get_query_length(url: str) -> int:
    """Returns the length of the URL query string."""
    parsed_url = urlparse(url)

    # Handles URLs without scheme
    if not parsed_url.netloc:
        parsed_url = urlparse("http://" + url)

    return len(parsed_url.query)