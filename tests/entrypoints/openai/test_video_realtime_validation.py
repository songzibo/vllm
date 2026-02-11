# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import io
import json

import numpy as np
import pytest
import websockets
from PIL import Image

from vllm.assets.video import VideoAsset

from ...utils import RemoteOpenAIServer
from .conftest import add_attention_backend

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


def _frame_to_base64(frame: np.ndarray) -> str:
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_websocket_url(server: RemoteOpenAIServer) -> str:
    http_url = server.url_root
    ws_url = http_url.replace("http://", "ws://")
    return f"{ws_url}/v1/video_realtime"


async def receive_event(ws, timeout: float = 60.0) -> dict:
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(message)


async def send_event(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


@pytest.fixture
def video_frames() -> list[np.ndarray]:
    video_path = VideoAsset("baby_reading").video_path
    frames = VideoAsset("baby_reading", num_frames=4).np_ndarrays
    assert frames.shape[0] > 0, f"No frames loaded from {video_path}"
    return [frame for frame in frames]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.skip(reason="Video realtime streaming requires large multimodal model")
async def test_video_realtime_streaming(
    model_name, video_frames, rocm_aiter_fa_attention
):
    server_args = ["--enforce-eager", "--limit-mm-per-prompt", "video=1"]
    add_attention_backend(server_args, rocm_aiter_fa_attention)

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        ws_url = _get_websocket_url(remote_server)
        async with websockets.connect(ws_url) as ws:
            event = await receive_event(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await send_event(ws, {"type": "session.update", "model": model_name})
            await send_event(ws, {"type": "input_video_buffer.commit"})

            for frame in video_frames:
                await send_event(
                    ws,
                    {
                        "type": "input_video_buffer.append",
                        "frame": _frame_to_base64(frame),
                    },
                )

            await send_event(ws, {"type": "input_video_buffer.commit", "final": True})

            done_received = False
            while not done_received:
                event = await receive_event(ws, timeout=120.0)
                if event["type"] == "video_response.delta":
                    assert isinstance(event["delta"], str)
                elif event["type"] == "video_response.done":
                    done_received = True
                    assert "text" in event
                elif event["type"] == "error":
                    pytest.fail(f"Received error: {event}")