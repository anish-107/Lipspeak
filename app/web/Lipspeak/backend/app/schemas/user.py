"""user.py"""


from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
)


class UserResponse(
    BaseModel,
):
    id: str

    username: str

    name: str

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )