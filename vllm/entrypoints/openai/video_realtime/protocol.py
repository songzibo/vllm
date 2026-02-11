# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.utils import random_uuid

# Client -> Server Events


class InputVideoBufferAppend(OpenAIBaseModel):
    """Append video frame to buffer"""

    type: Literal["input_video_buffer.append"] = "input_video_buffer.append"
    frame: str  # base64-encoded image bytes (e.g. JPEG/PNG)


class InputVideoBufferCommit(OpenAIBaseModel):
    """Process accumulated video buffer"""

    type: Literal["input_video_buffer.commit"] = "input_video_buffer.commit"
    final: bool = False


# Server -> Client Events
class SessionUpdate(OpenAIBaseModel):
    """Configure session parameters"""

    type: Literal["session.update"] = "session.update"
    model: str | None = None


class SessionCreated(OpenAIBaseModel):
    """Connection established notification"""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class VideoResponseDelta(OpenAIBaseModel):
    """Incremental response text"""

    type: Literal["video_response.delta"] = "video_response.delta"
    delta: str  # Incremental text


class VideoResponseDone(OpenAIBaseModel):
    """Final response with usage stats"""

    type: Literal["video_response.done"] = "video_response.done"
    text: str  # Complete response
    usage: UsageInfo | None = None


class ErrorEvent(OpenAIBaseModel):
    """Error notification"""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
