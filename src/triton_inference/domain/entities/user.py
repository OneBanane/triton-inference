"""User entity: the person we recommend movies to."""

from pydantic import BaseModel, ConfigDict, field_validator


class User(BaseModel):
    """A MovieLens user.

    Expected contract (TDD scaffold, see tests/domain/test_user.py):
        - `user_id` is the raw MovieLens id and must be a positive integer
        - instances are immutable and hashable
    """

    model_config = ConfigDict(frozen=True)

    user_id: int

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, user_id: int) -> int:
        if user_id <= 0:
            raise ValueError("user_id must be a positive integer")

        return user_id
