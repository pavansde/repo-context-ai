"""API routes."""

import json
import logging
import time
import uuid
from typing import Iterator

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from api.auth import require_api_key
from api.schemas import ChatRequest, ChatResponse
from core.config import Settings, get_settings
from core.exceptions import ConfigurationError
from core.repository_registry import (
    get_repository_registry,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)

logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(
    request: ChatRequest,
    http_request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
):
    """Execute a repository-aware agent request."""

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "Chat request started | "
        "request_id=%s repository_id=%s thread_id=%s",
        request_id,
        request.repository_id,
        request.thread_id,
    )

    registry = get_repository_registry(settings)

    if not registry.contains(request.repository_id):
        logger.warning(
            "Repository authorization failed | "
            "request_id=%s repository_id=%s",
            request_id,
            request.repository_id,
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Access to this repository "
                "is not authorized"
            ),
        )

    try:
        repository_path = registry.resolve(
            request.repository_id
        )

    except ConfigurationError as exc:
        logger.warning(
            "Repository resolution failed | "
            "request_id=%s repository_id=%s",
            request_id,
            request.repository_id,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    app_state = http_request.app.state.repo_context

    agent = app_state.create_agent(
        repository_path=repository_path,
        repository_id=request.repository_id,
    )

    config = {
        "configurable": {
            "thread_id": request.thread_id,
        }
    }

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=request.message
                    )
                ]
            },
            config=config,
        )

    except Exception:
        duration_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "Chat request failed | "
            "request_id=%s repository_id=%s "
            "thread_id=%s duration_ms=%.2f",
            request_id,
            request.repository_id,
            request.thread_id,
            duration_ms,
        )

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed",
        )

    answer = result["messages"][-1].content

    duration_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    logger.info(
        "Chat request completed | "
        "request_id=%s repository_id=%s "
        "thread_id=%s duration_ms=%.2f",
        request_id,
        request.repository_id,
        request.thread_id,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id

    return ChatResponse(
        thread_id=request.thread_id,
        answer=answer,
    )


@router.post(
    "/chat/stream",
    dependencies=[Depends(require_api_key)],
)
def chat_stream(
    request: ChatRequest,
    http_request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
):
    """Stream an agent response using Server-Sent Events."""

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        "Streaming chat request started | "
        "request_id=%s repository_id=%s thread_id=%s",
        request_id,
        request.repository_id,
        request.thread_id,
    )

    registry = get_repository_registry(settings)

    if not registry.contains(request.repository_id):
        logger.warning(
            "Streaming repository authorization failed | "
            "request_id=%s repository_id=%s",
            request_id,
            request.repository_id,
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Access to this repository "
                "is not authorized"
            ),
        )

    try:
        repository_path = registry.resolve(
            request.repository_id
        )

    except ConfigurationError as exc:
        logger.warning(
            "Streaming repository resolution failed | "
            "request_id=%s repository_id=%s",
            request_id,
            request.repository_id,
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    app_state = http_request.app.state.repo_context

    agent = app_state.create_agent(
        repository_path=repository_path,
        repository_id=request.repository_id,
    )

    config = {
        "configurable": {
            "thread_id": request.thread_id,
        }
    }

    def generate() -> Iterator[str]:
        try:
            for chunk, metadata in agent.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=request.message
                        )
                    ]
                },
                config=config,
                stream_mode="messages",
            ):
                if metadata.get(
                    "langgraph_node"
                ) != "agent":
                    continue

                content = chunk.content

                if not content:
                    continue

                if isinstance(content, str):
                    text = content

                else:
                    text_parts = []

                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        if block.get("type") == "text":
                            text_parts.append(
                                block.get(
                                    "text",
                                    "",
                                )
                            )

                    text = "".join(text_parts)

                if not text:
                    continue

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "token",
                            "content": text,
                        }
                    )
                    + "\n\n"
                )

            duration_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            logger.info(
                "Streaming chat request completed | "
                "request_id=%s repository_id=%s "
                "thread_id=%s duration_ms=%.2f",
                request_id,
                request.repository_id,
                request.thread_id,
                duration_ms,
            )

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "request_id": request_id,
                    }
                )
                + "\n\n"
            )

        except Exception:
            duration_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            logger.exception(
                "Streaming chat request failed | "
                "request_id=%s repository_id=%s "
                "thread_id=%s duration_ms=%.2f",
                request_id,
                request.repository_id,
                request.thread_id,
                duration_ms,
            )

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "error": (
                            "Agent execution failed"
                        ),
                        "request_id": request_id,
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )