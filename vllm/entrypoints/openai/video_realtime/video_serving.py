# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for streaming video input via WebSocket."""

import asyncio
from collections.abc import AsyncGenerator

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import StreamingInput, TextPrompt
from vllm.logger import init_logger

logger = init_logger(__name__)

# Default text prompt when client does not send one (video-only input).
DEFAULT_VIDEO_PROMPT = "Describe what you see in the video."


class OpenAIServingRealtimeVideo(OpenAIServing):
    """Realtime video understanding via WebSocket streaming.

    Transforms streamed video frames into StreamingInput objects with
    multi_modal_data for engine.generate().
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        log_error_stack: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.task_type = "realtime_video"
        logger.info(
            "OpenAIServingRealtimeVideo initialized for task: %s", self.task_type
        )

    async def stream_video_realtime(
        self,
        video_chunk_queue: asyncio.Queue[list | None],
        prompt_text: str = DEFAULT_VIDEO_PROMPT,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Turn queued video chunks into StreamingInput for engine.generate().

        Each chunk is a list of frames (PIL Images or compatible). One chunk
        is consumed per commit from the client.

        Args:
            video_chunk_queue: Queue of frame lists; None signals end of stream.
            prompt_text: Text prompt to use with each video chunk.

        Yields:
            StreamingInput with TextPrompt + multi_modal_data["video"].
        """
        while True:
            chunk = await video_chunk_queue.get()
            if chunk is None:
                break
            if not chunk:
                continue
            prompt: TextPrompt = TextPrompt(
                prompt=prompt_text,
                multi_modal_data={"video": chunk},
            )
            yield StreamingInput(prompt=prompt)
