from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from tidyrun.keys import (
    Key,
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
def test_encode_decode_roundtrip(key: Key) -> None:
    encoded = encode_key(key)

    assert not encoded.startswith(".")
    assert "/" not in encoded
    assert "\\" not in encoded
    assert decode_key(encoded) == key


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("plain", "plain"),
        ("7", '"7"'),
        ("true", '"true"'),
        ("2026-03-07", '"2026-03-07"'),
    ],
)
def test_string_encoding_quotes_only_when_needed(key: str, expected: str) -> None:
    assert encode_key(key) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("simple string", "simple string"),
        (7, "7"),
        (date(2026, 3, 7), "2026-03-07"),
    ],
)
def test_encode_key_requested_mappings(key: Key, expected: str) -> None:
    assert encode_key(key) == expected


@pytest.mark.parametrize("key", [None, {"a": 1}, [1, 2], "with/slash"])
def test_encode_key_rejects_unsupported_values(key: object) -> None:
    with pytest.raises(TidyRunKeyEncodingError):
        encode_key(key)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ["", "/bad", "a/b", "a\\b", ".hidden", '"unterminated']
)
def test_decode_key_rejects_invalid_names(name: str) -> None:
    with pytest.raises(TidyRunKeyDecodingError):
        decode_key(name)


def test_decode_key_accepts_bare_string_names() -> None:
    assert decode_key("abc") == "abc"
    assert decode_key("simple string") == "simple string"
