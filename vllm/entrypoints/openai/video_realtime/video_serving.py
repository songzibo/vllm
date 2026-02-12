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
            # Empty list = text-only turn (e.g. commit with no frames, or text after video).
            if not chunk:
                yield StreamingInput(
                    prompt=TextPrompt(prompt=prompt_text),
                )
                continue
            # Models like Qwen2-VL/Qwen3-VL require video metadata (fps, frames_indices, etc.).
            # Build (video_array, metadata) tuple; metadata format matches vllm video loaders.
            num_frames = len(chunk)
            frames_array = np.stack([np.array(img) for img in chunk])
            # Default fps=1 for streaming (no real timeline); duration = num_frames seconds.
            fps = 1.0
            metadata = {
                "total_num_frames": num_frames,
                "fps": fps,
                "duration": num_frames / fps,
                "video_backend": "realtime_stream",
                "frames_indices": list(range(num_frames)),
                "do_sample_frames": True,
            }
            # Prompt must contain the model's video placeholder for replacement.
            if VIDEO_PLACEHOLDER not in prompt_text:
                effective_prompt = VIDEO_PLACEHOLDER + " " + prompt_text
            else:
                effective_prompt = prompt_text
            prompt: TextPrompt = TextPrompt(
                prompt=effective_prompt,
                multi_modal_data={"video": (frames_array, metadata)},
            )
            yield StreamingInput(prompt=prompt)
