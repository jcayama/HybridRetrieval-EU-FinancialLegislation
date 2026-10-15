import pytest

from tests.helpers import skip_or_fail


def test_outside_ci_a_missing_service_skips(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception):
        skip_or_fail("sin postgres")


def test_in_ci_a_missing_service_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match=r"\[CI\] sin postgres"):
        skip_or_fail("sin postgres")
