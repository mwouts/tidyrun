from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from tidyrun.keys import (
    TidyRunKeyDecodingError,
    TidyRunKeyEncodingError,
    decode_key,
    encode_key,
)


@pytest.mark.parametrize(
    "key",
    [
        "simple",
        "with spaces and #symbols!",
        "1",
        "true",
        "2026-05-10",
        0,
        42,
        3.14,
        True,
        False,
        date(2026, 5, 10),
        time(12, 34, 56),
        datetime(2026, 5, 10, 1, 2, 3),
        datetime(2026, 5, 10, 1, 2, 3, tzinfo=timezone.utc),
    ],
)
def test_encode_decode_roundtrip(key: object) -> None:
    encoded = encode_key(key)  # type: ignore[arg-type]

    assert not encoded.startswith(".")
    assert "/" not in encoded
    assert "\\" not in encoded
    assert decode_key(encoded) == key


def test_strings_are_toml_encoded() -> None:
    assert encode_key("plain") == '"plain"'


@pytest.mark.parametrize("key", [None, {"a": 1}, [1, 2], "with/slash"])
def test_encode_key_rejects_unsupported_values(key: object) -> None:
    with pytest.raises(TidyRunKeyEncodingError):
        encode_key(key)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ["", "abc", "/bad", "a/b", "a\\b", ".hidden", '"unterminated']
)
def test_decode_key_rejects_invalid_names(name: str) -> None:
    with pytest.raises(TidyRunKeyDecodingError):
        decode_key(name)
