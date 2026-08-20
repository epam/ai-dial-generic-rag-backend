import pytest
from aidial_sdk.exceptions import InvalidRequestError

from generic_rag.services.document_service import SUPPORTED_CONTENT_TYPES, validate_content_type


@pytest.mark.parametrize("content_type", sorted(SUPPORTED_CONTENT_TYPES))
def test_supported_types_are_accepted(content_type):
    validate_content_type(content_type)


@pytest.mark.parametrize(
    "content_type",
    [
        "text/markdown; charset=utf-8",
        "text/plain;charset=UTF-8",
        "TEXT/MARKDOWN",
        " application/pdf ",
    ],
)
def test_parameters_and_casing_do_not_make_a_type_unsupported(content_type):
    """A client being explicit about the encoding is not a client uploading the wrong thing."""
    validate_content_type(content_type)


@pytest.mark.parametrize(
    "content_type",
    [
        "application/msword",
        "image/png",
        "application/octet-stream",
        "text/markdownish",
        "",
    ],
)
def test_unsupported_types_are_refused(content_type):
    with pytest.raises(InvalidRequestError) as exc_info:
        validate_content_type(content_type)

    # The message names what is accepted: the caller cannot see the allow-list otherwise.
    assert "unsupported file type" in str(exc_info.value)
    assert "text/markdown" in str(exc_info.value)
