"""Tests for triton_inference.domain.entities.User (TDD scaffold)."""

import pytest
from pydantic import ValidationError

from triton_inference.domain.entities import User


def test_stores_user_id():
    assert User(user_id=7).user_id == 7


@pytest.mark.parametrize("bad_id", [0, -1])
def test_rejects_non_positive_id(bad_id):
    with pytest.raises(ValidationError):
        User(user_id=bad_id)


def test_rejects_non_integer_id():
    with pytest.raises(ValidationError):
        User(user_id="abc")


def test_is_immutable():
    user = User(user_id=1)
    with pytest.raises(ValidationError):
        user.user_id = 2


def test_equal_by_value_and_hashable():
    assert User(user_id=1) == User(user_id=1)
    assert len({User(user_id=1), User(user_id=1), User(user_id=2)}) == 2
