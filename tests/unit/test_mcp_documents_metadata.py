import re

import pytest

from generic_rag.app.mcp import DOCUMENT_IDS_PATTERN


@pytest.mark.parametrize("document_ids", ["1", "7", "1,5,9", "10,200,3000"])
def test_pattern_accepts_a_single_id_and_a_comma_separated_list(document_ids):
    assert re.fullmatch(DOCUMENT_IDS_PATTERN, document_ids)


@pytest.mark.parametrize("document_ids", ["", "1,", ",1", "1,,2", "0", "0,3", "abc", "1, 2", "1;2", "-1"])
def test_pattern_rejects_everything_a_consumer_must_not_send(document_ids):
    """A malformed segment must fail the read, because a dropped id becomes a missing citation."""
    assert re.fullmatch(DOCUMENT_IDS_PATTERN, document_ids) is None
