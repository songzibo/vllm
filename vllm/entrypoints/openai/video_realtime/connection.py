# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import io
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from uuid import uuid4

from PIL import Image
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm import envs
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.video_realtime.protocol import (
    ErrorEvent,
    InputVideoBufferAppend,
    InputVideoBufferCommit,
    SessionCreated,
    VideoResponseDelta,
    VideoResponseDone,
)
from vllm.entrypoints.openai.video_realtime.serving import (
    OpenAIServingVideoRealtime,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)


class VideoRealtimeConnection:
    """Manages WebSocket lifecycle and state for realtime video.

    This class handles:
    - WebSocket connection lifecycle (accept, receive, send, close)
    - Event routing (session.update, append, commit)
    - Video buffering via asyncio.Queue
    - Generation task management
    - Error handling and cleanup
    """

    def __init__(self, websocket: WebSocket, serving: OpenAIServingVideoRealtime):
        self.websocket = websocket
        self.connection_id = f"ws-{uuid4()}"
        self.serving = serving
        self.video_queue: asyncio.Queue[Image.Image | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None

        self._is_connected = False
        self._is_input_finished = False
        self._is_model_validated = False

        self._max_video_frames = envs.VLLM_MAX_VIDEO_CLIP_FRAMES

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket connection accepted: %s", self.connection_id)
        self._is_connected = True

        # Send session created event
        await self.send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError:
                    await self.send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self.send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self.cleanup()

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        if self.serving._is_model_supported(model):
            return None

        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def handle_event(self, event: dict):
        """Route events to handlers.

        Supported event types:
        - session.update: Configure model
        - input_video_buffer.append: Add video frame to queue
        - input_video_buffer.commit: Start video generation
        """
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            self._check_model(event["model"])
            self._is_model_validated = True
        elif event_type == "input_video_buffer.append":
            append_event = InputVideoBufferAppend(**event)
            try:
                frame_bytes = base64.b64decode(append_event.frame)
                frame_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

                if frame_image.size[0] == 0 or frame_image.size[1] == 0:
                    raise VLLMValidationError("Can't process empty video frame.")

                if self.video_queue.qsize() >= self._max_video_frames:
                    raise VLLMValidationError(
                        "Maximum video frames exceeded",
                        parameter="video_frames",
                        value=self.video_queue.qsize(),
                    )

                # Put video frame in queue
                self.video_queue.put_nowait(frame_image)

            except Exception as e:
                logger.error("Failed to decode video frame: %s", e)
                await self.send_error("Invalid video frame", "invalid_video")

        elif event_type == "input_video_buffer.commit":
            if not self._is_model_validated:
                err_msg = (
                    "Model not validated. Make sure to validate the"
                    " model by sending a session.update event."
                )
                await self.send_error(
                    err_msg,
                    "model_not_validated",
                )
                return

            commit_event = InputVideoBufferCommit(**event)
            # final signals that the video is finished
            if commit_event.final:
                self._is_input_finished = True
            else:
                await self.start_generation()
        else:
            await self.send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def video_stream_generator(self) -> AsyncGenerator[Image.Image, None]:
        """Generator that yields video frames from the queue."""
        while True:
            frame = await self.video_queue.get()
            if frame is None:  # Sentinel value to stop
                break
            yield frame

    async def start_generation(self):
        """Start the video generation task."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # Create video stream generator
        video_stream = self.video_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # Transform to StreamingInput generator
        streaming_input_gen = self.serving.transcribe_realtime(
            video_stream, input_stream
        )

        # Start generation task
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream)
        )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Run the generation and stream results back to the client.

        This method:
        1. Creates sampling parameters from session config
        2. Passes the streaming input generator to engine.generate()
        3. Streams video_response.delta events as text is generated
        4. Sends final video_response.done event with usage stats
        5. Feeds generated token IDs back to input_stream for next iteration
        6. Cleans up the video queue
        """
        request_id = f"vrt-{self.connection_id}-{uuid4()}"
        full_text = ""

        prompt_token_ids_len: int = 0
        completion_tokens_len: int = 0

        try:
            # Create sampling params
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            # Pass the streaming input generator to the engine
            # The engine will consume video frames as they arrive and
            # stream back responses incrementally
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Stream results back to client as they're generated
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)

                    delta = output.outputs[0].text
                    full_text += delta

                    # append output to input
                    input_stream.put_nowait(list(output.outputs[0].token_ids))
                    await self.send(VideoResponseDelta(delta=delta))

                    completion_tokens_len += len(output.outputs[0].token_ids)

                if not self._is_connected:
                    # finish because websocket connection was killed
                    break

                if self.video_queue.empty() and self._is_input_finished:
                    # finish because client signals that video input
                    # is finished
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )

            # Send final completion event
            await self.send(VideoResponseDone(text=full_text, usage=usage))

            # Clear queue for next clip
            while not self.video_queue.empty():
                self.video_queue.get_nowait()

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")

    async def send(
        self, event: SessionCreated | VideoResponseDelta | VideoResponseDone
    ):
        """Send event to client."""
        data = event.model_dump_json()
        await self.websocket.send_text(data)

    async def send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        error_event = ErrorEvent(error=message, code=code)
        await self.websocket.send_text(error_event.model_dump_json())

    async def cleanup(self):
        """Cleanup resources."""
        # Signal video stream to stop
        self.video_queue.put_nowait(None)

        # Cancel generation task if running
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

        logger.debug("Connection cleanup complete: %s", self.connection_id)
