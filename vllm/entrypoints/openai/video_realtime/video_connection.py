# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""WebSocket connection handler for streaming video input."""

import asyncio
import base64
import io
import json
from http import HTTPStatus
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.video_realtime.protocol import (
    CompletionDelta,
    CompletionDone,
    ErrorEvent,
    InputVideoBufferAppend,
    InputVideoBufferCommit,
    SessionCreated,
)
from vllm.entrypoints.openai.video_realtime.video_serving import (
    DEFAULT_VIDEO_PROMPT,
    OpenAIServingRealtimeVideo,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)

# Max frames per commit to avoid OOM (configurable via env if needed).
DEFAULT_MAX_FRAMES_PER_COMMIT = 64

def _decode_video_frame(payload_b64: str, fmt: str | None):
    """Decode base64 to a single frame (PIL Image) for multi_modal_data."""
    raw = base64.b64decode(payload_b64)
    if not raw:
        raise VLLMValidationError("Empty video frame data.")
    try:
        from PIL import Image
    except ImportError:
        raise VLLMValidationError(
            "PIL is required for streaming video. Install with: pip install Pillow"
        )
    stream = io.BytesIO(raw)
    img = Image.open(stream).convert("RGB")
    return img


class RealtimeVideoConnection:
    """Manages WebSocket lifecycle for realtime video understanding.

    - Session: session.update (model, optional prompt)
    - Append: input_video_buffer.append (base64 frame/chunk)
    - Commit: input_video_buffer.commit (process buffer, optional final)
      - With frames: process video + prompt. With no frames: text-only turn (prompt only).
      - So "text-only" or "text first then streaming video" are supported.
    - Server sends: completion.delta, completion.done, error
    """

    def __init__(
        self,
        websocket: WebSocket,
        serving: OpenAIServingRealtimeVideo,
        *,
        max_frames_per_commit: int = DEFAULT_MAX_FRAMES_PER_COMMIT,
    ):
        self.websocket = websocket
        self.connection_id = f"ws-video-{uuid4()}"
        self.serving = serving
        self._frame_buffer: list = []
        self._video_chunk_queue: asyncio.Queue[list | None] = asyncio.Queue()
        self.generation_task: asyncio.Task | None = None
        self._is_connected = False
        self._is_input_finished = False
        self._is_model_validated = False
        self._prompt_text = DEFAULT_VIDEO_PROMPT
        self._max_frames_per_commit = max_frames_per_commit

    async def handle_connection(self):
        """Main connection loop."""
        await self.websocket.accept()
        logger.debug("WebSocket (video) connection accepted: %s", self.connection_id)
        self._is_connected = True
        await self._send(SessionCreated())

        try:
            while True:
                message = await self.websocket.receive_text()
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON", "invalid_json")
                except Exception as e:
                    logger.exception("Error handling event: %s", e)
                    await self._send_error(str(e), "processing_error")
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected: %s", self.connection_id)
            self._is_connected = False
        except Exception as e:
            logger.exception("Unexpected error in connection: %s", e)
        finally:
            await self._cleanup()

    def _check_model(self, model: str | None) -> ErrorResponse | None:
        if self.serving._is_model_supported(model):
            return None
        return self.serving.create_error_response(
            message=f"The model `{model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def _handle_event(self, event: dict):
        """Route events to handlers."""
        event_type = event.get("type")
        if event_type == "session.update":
            logger.debug("Session updated: %s", event)
            self._check_model(event.get("model"))
            self._is_model_validated = True
            if "prompt" in event and event["prompt"]:
                self._prompt_text = event["prompt"]
        elif event_type == "input_video_buffer.append":
            append_evt = InputVideoBufferAppend(**event)
            try:
                frame = _decode_video_frame(append_evt.video, append_evt.format)
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) > self._max_frames_per_commit:
                    raise VLLMValidationError(
                        f"Too many frames in buffer (max {self._max_frames_per_commit}). "
                        "Send input_video_buffer.commit first."
                    )
            except Exception as e:
                logger.error("Failed to decode video frame: %s", e)
                await self._send_error("Invalid video data", "invalid_video")
        elif event_type == "input_video_buffer.commit":
            if not self._is_model_validated:
                await self._send_error(
                    "Model not validated. Send session.update with model first.",
                    "model_not_validated",
                )
                return
            commit_evt = InputVideoBufferCommit(**event)
            # Always enqueue: frames → video chunk; no frames → text-only chunk (empty list).
            # Then if final: mark input finished and send end-of-stream (None) to the consumer.
            if self._frame_buffer:
                self._video_chunk_queue.put_nowait(list(self._frame_buffer))
                self._frame_buffer = []
            else:
                self._video_chunk_queue.put_nowait([])
            if commit_evt.final:
                self._is_input_finished = True
                self._video_chunk_queue.put_nowait(None)
            if self.generation_task is None or self.generation_task.done():
                await self._start_generation()
        else:
            await self._send_error(f"Unknown event type: {event_type}", "unknown_event")

    async def _start_generation(self):
        """Start the generation task for the current video chunk queue."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        stream_gen = self.serving.stream_video_realtime(
            self._video_chunk_queue,
            prompt_text=self._prompt_text,
        )
        self.generation_task = asyncio.create_task(
            self._run_generation(stream_gen)
        )

    async def _run_generation(self, streaming_input_gen):
        """Run engine.generate with streaming input and send completion events."""
        request_id = f"rt-video-{self.connection_id}-{uuid4()}"
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0

        try:
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            sampling_params = SamplingParams.from_optional(
                temperature=0.0,
                max_tokens=1024,
                output_kind=RequestOutputKind.DELTA,
                skip_clone=True,
            )

            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            out_iter = 0
            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)
                    delta = output.outputs[0].text
                    n_tokens = len(output.outputs[0].token_ids)
                    full_text += delta
                    await self._send(CompletionDelta(delta=delta))
                    completion_tokens_len += n_tokens
                    logger.debug(
                        "[realtime_video] output iter=%s delta=%r n_tokens=%s "
                        "full_len=%s queue_empty=%s input_finished=%s",
                        out_iter,
                        delta,
                        n_tokens,
                        len(full_text),
                        self._video_chunk_queue.empty(),
                        self._is_input_finished,
                    )
                out_iter += 1
                if not self._is_connected:
                    logger.info("[realtime_video] break: connection closed")
                    break
                # Do NOT break when queue empty and input_finished: consume all
                # outputs from this generation (model may yield one delta per token).

            logger.info(
                "[realtime_video] gen done full_text_len=%s completion_tokens=%s",
                len(full_text),
                completion_tokens_len,
            )

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self._send(CompletionDone(text=full_text, usage=usage))

            while not self._video_chunk_queue.empty():
                try:
                    self._video_chunk_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in video generation: %s", e)
            await self._send_error(str(e), "processing_error")

    async def _send(self, event):
        """Send event to client."""
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(self, message: str, code: str | None = None):
        """Send error event to client."""
        await self.websocket.send_text(
            ErrorEvent(error=message, code=code).model_dump_json()
        )

    async def _cleanup(self):
        """Cleanup resources."""
        self._video_chunk_queue.put_nowait(None)
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
        logger.debug("Connection cleanup complete: %s", self.connection_id)
