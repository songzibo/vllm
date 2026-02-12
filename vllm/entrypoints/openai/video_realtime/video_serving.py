# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving layer for streaming video input via WebSocket."""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import StreamingInput, TextPrompt
from vllm.logger import init_logger

logger = init_logger(__name__)

# Default text prompt when client does not send one (video-only input).
DEFAULT_VIDEO_PROMPT = "Describe what you see in the video."

# Video placeholder required by the model so prompt replacement can inject video.
# Used by Qwen2-VL/Qwen3-VL; other models may use different placeholders (set
# prompt via session.update including the correct placeholder if needed).
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

# Qwen-style chat wrapper so the model generates a full assistant reply instead of
# stopping after one token (EOS). Without this, raw prompt has no "assistant" turn
# and the model may emit EOS immediately.
QWEN_CHAT_USER_PREFIX = "<|im_start|>user\n"
QWEN_CHAT_USER_SUFFIX = "<|im_end|>\n"
QWEN_CHAT_ASSISTANT_PREFIX = "<|im_start|>assistant\n"


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
        video_batch_queue: asyncio.Queue[list | None],
        prompt_text: str = DEFAULT_VIDEO_PROMPT,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Turn queued video batches into StreamingInput for engine.generate().

        One batch (one commit) → one StreamingInput. None in the queue signals EOS.
        Caller can advance this generator once per batch and run one generate per yield.
        """
        while True:
            batch = await video_batch_queue.get()
            if batch is None:
                break
            if not batch:
                yield StreamingInput(prompt=TextPrompt(prompt=prompt_text))
                continue
            num_frames = len(batch)
            frames_array = np.stack([np.array(img) for img in batch])
            fps = 1.0
            metadata = {
                "total_num_frames": num_frames,
                "fps": fps,
                "duration": num_frames / fps,
                "video_backend": "realtime_stream",
                "frames_indices": list(range(num_frames)),
                "do_sample_frames": True,
            }
            if VIDEO_PLACEHOLDER not in prompt_text:
                user_content = VIDEO_PLACEHOLDER + " " + prompt_text
            else:
                user_content = prompt_text
            effective_prompt = (
                QWEN_CHAT_USER_PREFIX
                + user_content
                + QWEN_CHAT_USER_SUFFIX
                + QWEN_CHAT_ASSISTANT_PREFIX
            )
            prompt = TextPrompt(
                prompt=effective_prompt,
                multi_modal_data={"video": (frames_array, metadata)},
            )
            yield StreamingInput(prompt=prompt)
