# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Protocol events for streaming video realtime API."""

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
    """Append video frame or chunk to buffer."""

    type: Literal["input_video_buffer.append"] = "input_video_buffer.append"
    video: str  # base64-encoded frame (e.g. JPEG) or video segment (e.g. MP4)
    format: str | None = None  # optional: "image/jpeg", "image/png", "video/mp4"


class InputVideoBufferCommit(OpenAIBaseModel):
    """Process accumulated video buffer."""

    type: Literal["input_video_buffer.commit"] = "input_video_buffer.commit"
    final: bool = False


# Server -> Client Events (shared types reused from realtime where applicable)
class SessionCreated(OpenAIBaseModel):
    """Connection established notification."""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class CompletionDelta(OpenAIBaseModel):
    """Incremental completion text."""

    type: Literal["completion.delta"] = "completion.delta"
    delta: str


class CompletionDone(OpenAIBaseModel):
    """Final completion with usage stats."""

    type: Literal["completion.done"] = "completion.done"
    text: str
    usage: UsageInfo | None = None


class ErrorEvent(OpenAIBaseModel):
    """Error notification."""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
