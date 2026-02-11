# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Literal, cast

from PIL import Image

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import PromptType, StreamingInput
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsRealtimeVideo

logger = init_logger(__name__)


class OpenAIServingVideoRealtime(OpenAIServing):
    """Realtime video understanding service via WebSocket streaming.

    Provides streaming video-to-text generation by transforming video frames
    into StreamingInput objects that can be consumed by the engine.
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

        self.task_type: Literal["video_realtime"] = "video_realtime"

        logger.info(
            "OpenAIServingVideoRealtime initialized for task: %s",
            self.task_type,
        )

    @cached_property
    def model_cls(self) -> type[SupportsRealtimeVideo]:
        """Get the model class that supports realtime video."""
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsRealtimeVideo], model_cls)

    async def transcribe_realtime(
        self,
        video_stream: AsyncGenerator[Image.Image, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform video stream into StreamingInput for engine.generate().

        Args:
            video_stream: Async generator yielding PIL.Image frames
            input_stream: Queue containing context token IDs from previous
                generation outputs. Used for autoregressive multi-turn
                processing where each generation's output becomes the context
                for the next iteration.

        Yields:
            StreamingInput objects containing video prompts for the engine
        """

        # mypy is being stupid
        # TODO(Patrick) - fix this
        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_video(
                video_stream, input_stream, self.model_config
            ),
        )

        async for prompt in stream_input_iter:
            yield StreamingInput(prompt=prompt)
