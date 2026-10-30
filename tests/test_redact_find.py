import pytest

from faultgate.redact import builtin, find

R = builtin()


@pytest.mark.parametrize("text, expected_kind, expected_val", [
    ("jwt token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U in header",
     "SECRET", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
     "SECRET", "abcdefghijklmnopqrstuvwxyz012345"),
    ("api key sk-proj-abcdefghijklmnopqrstuvwx",
     "SECRET", "sk-proj-abcdefghijklmnopqrstuvwx"),
    ("aws key AKIAIOSFODNN7EXAMPLE",
     "SECRET", "AKIAIOSFODNN7EXAMPLE"),
    ("contact user at jane.doe+x@corp.example.com today",
     "EMAIL", "jane.doe+x@corp.example.com"),
    ("ssn is 123-45-6789 here",
     "US_SSN", "123-45-6789"),
    ("card 4111 1111 1111 1111 on file",
     "CREDIT_CARD", "4111 1111 1111 1111"),
    ("card 4111-1111-1111-1111",
     "CREDIT_CARD", "4111-1111-1111-1111"),
    ("amex 3782 822463 10005 charged",
     "CREDIT_CARD", "3782 822463 10005"),
    ("connect to 192.168.10.24 now",
     "IP_ADDRESS", "192.168.10.24"),
    ("connect to 2001:0db8:85a3:0000:0000:8a2e:0370:7334 now",
     "IP_ADDRESS", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
    ("call +1 (415) 555-0132 today",
     "PHONE", "+1 (415) 555-0132"),
    ("call 020 7946 0958",
     "PHONE", "020 7946 0958"),
])
def test_find_each_kind(text, expected_kind, expected_val):
    spans = find(text)
    assert len(spans) == 1
    start, end, kind = spans[0]
    assert kind == expected_kind
    assert text[start:end] == expected_val


def test_find_luhn_invalid_card_not_returned():
    assert find("card 4111 1111 1111 1112") == []
    assert find("card 4111 1111 1111 1112 rejected by gateway") == []


def test_find_card_plus_ssn_in_one_run():
    text = "4111 1111 1111 1111 123-45-6789"
    spans = find(text)
    assert len(spans) == 2
    assert spans[0] == (0, 19, "CREDIT_CARD")
    assert spans[1] == (20, 31, "US_SSN")
    assert text[spans[0][0]:spans[0][1]] == "4111 1111 1111 1111"
    assert text[spans[1][0]:spans[1][1]] == "123-45-6789"

    # Non-card digits in a run are not included in the card span
    text_extra = "4111 1111 1111 1111 1234"
    spans_extra = find(text_extra)
    assert spans_extra == [(0, 19, "CREDIT_CARD")]
    assert text_extra[spans_extra[0][0]:spans_extra[0][1]] == "4111 1111 1111 1111"

    # Multiple Luhn-valid cards in one digit run
    text_multi = "4111 1111 1111 1111 5555 4444 3333 1111"
    spans_multi = find(text_multi)
    assert len(spans_multi) == 2
    assert spans_multi[0] == (0, 19, "CREDIT_CARD")
    assert spans_multi[1] == (20, 39, "CREDIT_CARD")
    assert text_multi[spans_multi[0][0]:spans_multi[0][1]] == "4111 1111 1111 1111"
    assert text_multi[spans_multi[1][0]:spans_multi[1][1]] == "5555 4444 3333 1111"


def test_find_overlap_removed():
    # Bearer secret containing an SSN pattern: Bearer (SECRET) precedes US_SSN in PATTERNS
    text = "Bearer 123-45-6789-abcdef0123456789"
    spans = find(text)
    assert len(spans) == 1
    start, end, kind = spans[0]
    assert kind == "SECRET"
    assert text[start:end] == "123-45-6789-abcdef0123456789"


def test_find_positions_slice_back_to_value():
    text = (
        "Refund for jane@corp.example.com of card 4111 1111 1111 1111 "
        "verified with ssn 123-45-6789 on server 192.168.1.50 now. "
        "Call +1 (415) 555-0132 or use token sk-proj-abcdefghijklmnopqrstuvwx."
    )
    spans = find(text)
    assert len(spans) == 6

    # Verify sorted strictly by start and non-overlapping
    for i in range(len(spans) - 1):
        assert spans[i][0] < spans[i + 1][0]
        assert spans[i][1] <= spans[i + 1][0]

    # Verify each position slices back to the exact value
    expected_values = [
        "jane@corp.example.com",
        "4111 1111 1111 1111",
        "123-45-6789",
        "192.168.1.50",
        "+1 (415) 555-0132",
        "sk-proj-abcdefghijklmnopqrstuvwx",
    ]
    for (start, end, _), expected in zip(spans, expected_values):
        assert text[start:end] == expected


def test_find_empty_string():
    assert find("") == []


def test_find_phone_digit_count_filter():
    # 8 digits: not kept
    assert find("call 123-45678") == []
    # 9 digits: kept
    assert len(find("call 123-456-789")) == 1
    # 11 digits: kept
    assert len(find("call +1 (415) 555-0132")) == 1


def test_builtin_replaces_spans_right_to_left():
    text = "4111 1111 1111 1111 123-45-6789"
    assert R(text) == "<CREDIT_CARD> <US_SSN>"

    text2 = "call +1 (415) 555-0132 and mail test@example.com"
    assert R(text2) == "call <PHONE> and mail <EMAIL>"
