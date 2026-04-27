import re
from urllib.parse import urlparse


def safe_urlparse(url: str):
    """
    Safely parses a URL.
    If parsing fails, tries to parse it with http:// prefix.
    If it still fails, returns None.
    """
    try:
        url = str(url)
        parsed_url = urlparse(url)

        if not parsed_url.netloc:
            parsed_url = urlparse("http://" + url)

        return parsed_url

    except ValueError:
        return None


def get_url_length(url: str) -> int:
    """Returns the total length of the URL."""
    return len(str(url))


def get_num_dots(url: str) -> int:
    """Returns the number of dots in the URL."""
    return str(url).count(".")


def get_num_hyphens(url: str) -> int:
    """Returns the number of hyphens in the URL."""
    return str(url).count("-")


def get_num_underscores(url: str) -> int:
    """Returns the number of underscores in the URL."""
    return str(url).count("_")


def get_num_slashes(url: str) -> int:
    """Returns the number of slashes in the URL."""
    return str(url).count("/")


def get_num_question_marks(url: str) -> int:
    """Returns the number of question marks in the URL."""
    return str(url).count("?")


def get_num_equal_signs(url: str) -> int:
    """Returns the number of equal signs in the URL."""
    return str(url).count("=")


def get_num_ampersands(url: str) -> int:
    """Returns the number of ampersands in the URL."""
    return str(url).count("&")


def get_num_digits(url: str) -> int:
    """Returns the number of digits in the URL."""
    url = str(url)
    return sum(char.isdigit() for char in url)


def get_num_letters(url: str) -> int:
    """Returns the number of alphabetic characters in the URL."""
    url = str(url)
    return sum(char.isalpha() for char in url)


def get_digit_ratio(url: str) -> float:
    """Returns the ratio of digits out of the total URL length."""
    url = str(url)

    if len(url) == 0:
        return 0.0

    return get_num_digits(url) / len(url)


def get_special_char_ratio(url: str) -> float:
    """Returns the ratio of non-alphanumeric characters out of the total URL length."""
    url = str(url)

    if len(url) == 0:
        return 0.0

    special_chars = sum(not char.isalnum() for char in url)
    return special_chars / len(url)


def get_domain_length(url: str) -> int:
    """Returns the length of the URL domain."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return len(parsed_url.netloc)


def get_path_length(url: str) -> int:
    """Returns the length of the URL path."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return len(parsed_url.path)


def get_query_length(url: str) -> int:
    """Returns the length of the URL query string."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return len(parsed_url.query)


def has_https(url: str) -> int:
    """Returns 1 if the URL uses HTTPS, otherwise 0."""
    return int(str(url).lower().startswith("https://"))


def has_http(url: str) -> int:
    """Returns 1 if the URL uses HTTP, otherwise 0."""
    return int(str(url).lower().startswith("http://"))


def has_ip_address(url: str) -> int:
    """Returns 1 if the URL contains an IPv4 address, otherwise 0."""
    url = str(url)
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    return int(bool(re.search(ip_pattern, url)))


def has_port(url: str) -> int:
    """Returns 1 if the domain contains a port number, otherwise 0."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return int(bool(re.search(r":\d+", parsed_url.netloc)))


def get_num_subdomains(url: str) -> int:
    """
    Returns the number of subdomains in the URL.

    Example:
        www.example.com -> 1
        login.secure.example.com -> 2
    """
    parsed_url = safe_urlparse(url)

    if parsed_url is None or not parsed_url.netloc:
        return 0

    domain = parsed_url.netloc.split(":")[0]
    return max(domain.count(".") - 1, 0)


def has_query(url: str) -> int:
    """Returns 1 if the URL contains a query string, otherwise 0."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return int(bool(parsed_url.query))


def get_num_query_params(url: str) -> int:
    """Returns the number of query parameters in the URL."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None or not parsed_url.query:
        return 0

    return parsed_url.query.count("&") + 1


def get_domain_digit_ratio(url: str) -> float:
    """Returns the ratio of digits in the domain."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None or not parsed_url.netloc:
        return 0.0

    domain = parsed_url.netloc

    if len(domain) == 0:
        return 0.0

    num_digits = sum(char.isdigit() for char in domain)
    return num_digits / len(domain)


def has_hyphen_in_domain(url: str) -> int:
    """Returns 1 if the domain contains a hyphen, otherwise 0."""
    parsed_url = safe_urlparse(url)

    if parsed_url is None:
        return 0

    return int("-" in parsed_url.netloc)


def extract_features(url: str) -> dict:
    """
    Extracts all URL features into a single dictionary.
    """
    return {
        "url_length": get_url_length(url),
        "num_dots": get_num_dots(url),
        "num_hyphens": get_num_hyphens(url),
        "num_underscores": get_num_underscores(url),
        "num_slashes": get_num_slashes(url),
        "num_question_marks": get_num_question_marks(url),
        "num_equal_signs": get_num_equal_signs(url),
        "num_ampersands": get_num_ampersands(url),

        "num_digits": get_num_digits(url),
        "num_letters": get_num_letters(url),
        "digit_ratio": get_digit_ratio(url),
        "special_char_ratio": get_special_char_ratio(url),

        "domain_length": get_domain_length(url),
        "path_length": get_path_length(url),
        "query_length": get_query_length(url),

        "has_https": has_https(url),
        "has_http": has_http(url),
        "has_ip_address": has_ip_address(url),
        "has_port": has_port(url),
        "num_subdomains": get_num_subdomains(url),
        "has_query": has_query(url),
        "num_query_params": get_num_query_params(url),
        "domain_digit_ratio": get_domain_digit_ratio(url),
        "has_hyphen_in_domain": has_hyphen_in_domain(url),
    }