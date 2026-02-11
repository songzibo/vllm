# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, WebSocket

from vllm.entrypoints.openai.video_realtime.connection import (
    VideoRealtimeConnection,
)
from vllm.entrypoints.openai.video_realtime.serving import (
    OpenAIServingVideoRealtime,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object

router = APIRouter()


@router.websocket("/v1/video_realtime")
async def video_realtime_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime video understanding.

    Protocol:
    1. Client connects to ws://host/v1/video_realtime
    2. Server sends session.created event
    3. Client optionally sends session.update with model/params
    4. Client sends input_video_buffer.commit when ready
    5. Client sends input_video_buffer.append events with base64 image frames
    6. Server processes and sends video_response.delta events
    7. Server sends video_response.done with final text + usage
    8. Repeat from step 5 for next clip
    9. Optionally, client sends input_video_buffer.commit with final=True
       to signal input is finished. Useful when streaming video files

    Video frames: base64-encoded image bytes (e.g., JPEG/PNG)
    """
    app = websocket.app
    serving = app.state.openai_serving_video_realtime

    connection = VideoRealtimeConnection(websocket, serving)
    await connection.handle_connection()


def attach_router(app: FastAPI):
    """Attach the video realtime router to the FastAPI app."""
    app.include_router(router)
    logger.info("Video realtime API router attached")


def init_video_realtime_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    state.openai_serving_video_realtime = (
        OpenAIServingVideoRealtime(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
        )
        if "video_realtime" in supported_tasks
        else None
    )
