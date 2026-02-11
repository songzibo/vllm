# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM Video Realtime WebSocket API
to perform streaming video understanding by uploading video frames.

Before running this script, you must start the vLLM server with a
video-realtime-capable model, for example:

    vllm serve Qwen/Qwen3-VL-4B-Instruct --enforce-eager \
        --limit-mm-per-prompt video=1

Requirements:
- vllm with video support
- websockets
- opencv-python (for loading video assets)
- numpy
- pillow

The script:
1. Connects to the Video Realtime WebSocket endpoint
2. Loads a video asset and extracts frames
3. Sends frames to the server as base64-encoded JPEGs
4. Receives and prints model responses as they stream
"""

import argparse
import asyncio
import base64
import io
import json
from typing import Iterable

import numpy as np
import websockets
from PIL import Image

from vllm.assets.video import VideoAsset, video_to_ndarrays


def _frame_to_base64(frame: np.ndarray, image_format: str = "JPEG") -> str:
    """Encode a single frame (H, W, C) into base64 image bytes."""
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_video_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Load a video and return sampled frames as numpy arrays."""
    frames = video_to_ndarrays(video_path, num_frames=num_frames)
    return [frame for frame in frames]


async def realtime_video_understand(
    frames: Iterable[np.ndarray],
    host: str,
    port: int,
    model: str,
) -> None:
    """Connect to the Video Realtime API and stream video frames."""
    uri = f"ws://{host}:{port}/v1/video_realtime"

    async with websockets.connect(uri) as ws:
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            return

        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_video_buffer.commit"}))

        print("Sending video frames...")
        for frame in frames:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_video_buffer.append",
                        "frame": _frame_to_base64(frame),
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_video_buffer.commit", "final": True}))
        print("Frames sent. Waiting for response...\n")

        print("Response: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "video_response.delta":
                print(response["delta"], end="", flush=True)
            elif response["type"] == "video_response.done":
                print(f"\n\nFinal response: {response['text']}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                break
            elif response["type"] == "error":
                print(f"\nError: {response['error']}")
                break


def main(args: argparse.Namespace) -> None:
    if args.video_path:
        video_path = args.video_path
    else:
        video_path = VideoAsset("baby_reading").video_path
        print(f"No video path provided, using default: {video_path}")

    frames = load_video_frames(video_path, num_frames=args.num_frames)
    asyncio.run(realtime_video_understand(frames, args.host, args.port, args.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Realtime WebSocket Client"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model that is served and should be pinged.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to a video file to stream.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample from the video.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    main(parser.parse_args())