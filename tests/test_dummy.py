import pytest


def test_pass():
    assert True


@pytest.mark.xfail
def test_fail():
    assert False
