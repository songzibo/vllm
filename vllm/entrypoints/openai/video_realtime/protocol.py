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
    """Append one video frame to buffer (video is sent frame-by-frame)."""

    type: Literal["input_video_buffer.append"] = "input_video_buffer.append"
    video: str  # base64-encoded frame (e.g. JPEG)
    format: str | None = None  # optional: "image/jpeg", "image/png"


class InputVideoBufferCommit(OpenAIBaseModel):
    """Process accumulated video buffer."""

    type: Literal["input_video_buffer.commit"] = "input_video_buffer.commit"
    final: bool = False


# Server -> Client Events (shared types reused from realtime where applicable)
class InputVideoBufferWaterLevel(OpenAIBaseModel):
    """Server buffer water level so client can wait for capacity before sending."""

    queue_depth: int = 0
    """Number of batches currently in the server queue. Client should send only when queue_depth < max_queue_size."""
    max_queue_size: int = 1
    """Max batches the server accepts before applying backpressure."""
    buffer_frames: int = 0
    """Current number of frames in the append buffer (before commit)."""


class InputVideoBufferWaterLevelEvent(OpenAIBaseModel):
    """Standalone event sent by server after each commit so client can throttle by water level."""

    type: Literal["input_video_buffer.water_level"] = "input_video_buffer.water_level"
    queue_depth: int = 0
    max_queue_size: int = 1
    buffer_frames: int = 0


class SessionCreated(OpenAIBaseModel):
    """Connection established notification."""

    type: Literal["session.created"] = "session.created"
    id: str = Field(default_factory=lambda: f"sess-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    input_video_buffer: InputVideoBufferWaterLevel | None = None
    """Initial buffer capacity so client can throttle sends."""


class CompletionDelta(OpenAIBaseModel):
    """Incremental completion text."""

    type: Literal["completion.delta"] = "completion.delta"
    delta: str


class CompletionDone(OpenAIBaseModel):
    """Final completion with usage stats."""

    type: Literal["completion.done"] = "completion.done"
    text: str
    usage: UsageInfo | None = None
    input_video_buffer: InputVideoBufferWaterLevel | None = None
    """Current buffer water level so client can throttle next sends."""


class ErrorEvent(OpenAIBaseModel):
    """Error notification."""

    type: Literal["error"] = "error"
    error: str
    code: str | None = None
