from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    repository_id: str = Field(
        ...,
        min_length=1,
        description="Registered repository identifier.",
    )

    message: str = Field(
        ...,
        min_length=1,
        description="User message.",
    )

    thread_id: str = Field(
        ...,
        min_length=1,
        description="Conversation thread identifier.",
    )


class ChatResponse(BaseModel):
    thread_id: str
    answer: str