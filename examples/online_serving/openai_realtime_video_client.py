# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM Realtime WebSocket API to stream
video frames (PNG base64) for multimodal models such as Qwen3-VL.

Before running this script, you must start the vLLM server with a
realtime-capable multimodal model, for example:

    vllm serve Qwen/Qwen3-VL-7B --enforce-eager

Requirements:
- vllm with multimodal support
- websockets
- opencv-python
- numpy

The script:
1. Connects to the Realtime WebSocket endpoint
2. Reads a video file and extracts frames at a fixed FPS
3. Encodes each frame as PNG base64 and streams to the server
4. Receives and prints response as it streams
"""

import argparse
import asyncio
import base64
import json
from dataclasses import dataclass

import cv2
import numpy as np
import websockets


@dataclass
class FramePacket:
    data: bytes
    frame_idx: int
    timestamp_ms: int


def iter_video_frames(video_path: str, target_fps: int) -> list[FramePacket]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(source_fps / target_fps)), 1)

    frames: list[FramePacket] = []
    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            success, buffer = cv2.imencode(".png", frame)
            if not success:
                raise RuntimeError("Failed to encode frame as PNG")

            timestamp_ms = int(1000 * (frame_idx / source_fps))
            frames.append(
                FramePacket(
                    data=buffer.tobytes(),
                    frame_idx=sampled_idx,
                    timestamp_ms=timestamp_ms,
                )
            )
            sampled_idx += 1

        frame_idx += 1

    cap.release()
    return frames


async def realtime_video_stream(
    video_path: str,
    host: str,
    port: int,
    model: str,
    fps: int,
    commit_every: int,
):
    uri = f"ws://{host}:{port}/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            return

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Start streaming
        await ws.send(json.dumps({"type": "input_video_buffer.commit"}))

        print(f"Loading video from: {video_path}")
        frames = iter_video_frames(video_path, fps)
        print(f"Sending {len(frames)} frames at ~{fps} fps...")

        for idx, packet in enumerate(frames):
            payload = {
                "type": "input_video_buffer.append",
                "frame": base64.b64encode(packet.data).decode("utf-8"),
                "frame_idx": packet.frame_idx,
                "timestamp_ms": packet.timestamp_ms,
            }
            await ws.send(json.dumps(payload))

            if commit_every > 0 and (idx + 1) % commit_every == 0:
                await ws.send(json.dumps({"type": "input_video_buffer.commit"}))

        await ws.send(json.dumps({"type": "input_video_buffer.commit", "final": True}))
        print("Video frames sent. Waiting for response...\n")

        print("Response: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "response.delta":
                print(response["delta"], end="", flush=True)
            elif response["type"] == "response.done":
                print(f"\n\nFinal response: {response['text']}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                break
            elif response["type"] == "error":
                print(f"\nError: {response['error']}")
                break


def main(args):
    asyncio.run(
        realtime_video_stream(
            args.video_path,
            args.host,
            args.port,
            args.model,
            args.fps,
            args.commit_every,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime WebSocket Video Streaming Client"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-7B",
        help="Model that is served and should be pinged.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file to stream.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Target FPS for frame sampling (default: 5)",
    )
    parser.add_argument(
        "--commit_every",
        type=int,
        default=8,
        help="Commit every N frames (default: 8)",
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
    args = parser.parse_args()
    main(args)