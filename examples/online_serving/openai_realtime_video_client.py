# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo client for the vLLM Realtime Video WebSocket API.

Sends video frames (from a file or image) and prints streamed completion.

Before running, start vLLM with a vision model that supports video, e.g.:

    vllm serve Qwen2.5-VL-7B-Instruct --enforce-eager

Requirements:
- vllm (with vision)
- websockets
- Pillow
- opencv-python (optional, for video files)

Usage:
  # From a video file (extracts frames)
  python openai_realtime_video_client.py --video_path /path/to/video.mp4

  # From a single image
  python openai_realtime_video_client.py --image_path /path/to/image.jpg

  # Custom prompt and model
  python openai_realtime_video_client.py --image_path frame.jpg --prompt "What is in this image?" --model Qwen2.5-VL-7B-Instruct
"""

import argparse
import asyncio
import base64
import io
import json

import websockets

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


def image_to_base64_jpeg(image_path: str, quality: int = 85) -> str:
    """Read image file and return base64-encoded JPEG."""
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def video_frames_to_base64_jpeg(
    video_path: str, max_frames: int = 32, quality: int = 85
) -> list[str]:
    """Read video file and return list of base64-encoded JPEG frames."""
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for video. Install with: pip install opencv-python"
        )
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    cap.release()
    return frames


async def run_realtime_video(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    image_path: str | None,
    video_path: str | None,
    max_frames: int,
):
    uri = f"ws://{host}:{port}/v1/realtime_video"

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        if msg.get("type") == "error":
            print(f"Error: {msg.get('error', msg)}")
            return
        if msg.get("type") != "session.created":
            print(f"Unexpected: {msg}")
            return
        print(f"Session created: {msg.get('id', '')}")

        payload = {"type": "session.update", "model": model}
        if prompt:
            payload["prompt"] = prompt
        await ws.send(json.dumps(payload))

        if image_path:
            print(f"Loading image: {image_path}")
            b64 = image_to_base64_jpeg(image_path)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_video_buffer.append",
                        "video": b64,
                        "format": "image/jpeg",
                    }
                )
            )
            await ws.send(json.dumps({"type": "input_video_buffer.commit", "final": True}))
        elif video_path:
            print(f"Loading video: {video_path} (max {max_frames} frames)")
            frames_b64 = video_frames_to_base64_jpeg(video_path, max_frames=max_frames)
            print(f"Sending {len(frames_b64)} frames...")
            for b64 in frames_b64:
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_video_buffer.append",
                            "video": b64,
                            "format": "image/jpeg",
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_video_buffer.commit", "final": True}))
        else:
            print("Provide --image_path or --video_path")
            return

        print("Waiting for completion...\n")
        while True:
            response = json.loads(await ws.recv())
            t = response.get("type")
            if t == "completion.delta":
                print(response.get("delta", ""), end="", flush=True)
            elif t == "completion.done":
                print(f"\n\nDone. Text: {response.get('text', '')}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                break
            elif t == "error":
                print(f"\nError: {response.get('error', response)}")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Video WebSocket client for vLLM"
    )
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.image_path and not args.video_path:
        parser.error("Provide at least one of --image_path or --video_path")

    asyncio.run(
        run_realtime_video(
            args.host,
            args.port,
            args.model,
            args.prompt,
            args.image_path,
            args.video_path,
            args.max_frames,
        )
    )


if __name__ == "__main__":
    main()
