# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo client for the vLLM Realtime Video WebSocket API.

Follows the protocol defined in vllm/entrypoints/openai/video_realtime/api_router.py:
connect, session.update, then loop: get water level (queue_depth < max_queue_size),
send one batch (appends + commit), recv completion.delta/done; use final=True on last batch.

Before running, start vLLM with a vision model that supports video, e.g.:

    vllm serve Qwen2.5-VL-7B-Instruct --enforce-eager

Requirements:
- vllm (with vision)
- websockets
- Pillow
- opencv-python (optional, for video files)

Usage:
  # From a video file (sends all frames by default)
  python openai_realtime_video_client.py --video-path /path/to/video.mp4

  # Send every 25th frame (e.g. 1 frame per second for 25fps video)
  python openai_realtime_video_client.py --video-path /path/to/video.mp4 --frame-interval 25

  # Limit to 32 frames, every 10th frame
  python openai_realtime_video_client.py --video-path /path/to/video.mp4 --max-frames 32 --frame-interval 10

  # From a single image
  python openai_realtime_video_client.py --image-path /path/to/image.jpg

  # Custom prompt and model
  python openai_realtime_video_client.py --image-path frame.jpg --prompt "What is in this image?" --model Qwen2.5-VL-7B-Instruct

Troubleshooting (no visible result):
  - Server accepts at most 64 frames per commit. If you send more, you get an error.
    Use --max-frames 64 or --frame-interval to send fewer frames.
  - Long videos take time to encode; wait after "Waiting for completion...".
  - Run with --verbose to print every message from the server (e.g. to see errors).
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
    video_path: str,
    max_frames: int | None = None,
    frame_interval: int = 1,
    quality: int = 85,
) -> list[str]:
    """Read video file and return list of base64-encoded JPEG frames.

    Args:
        video_path: Path to the video file.
        max_frames: Max number of frames to send; None = no limit (send all sampled).
        frame_interval: Send every Nth frame (1 = every frame, 25 = every 25th frame).
        quality: JPEG quality for encoding.
    """
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for video. Install with: pip install opencv-python"
        )
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")
    if frame_interval < 1:
        raise ValueError("frame_interval must be >= 1")
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        frame_idx += 1
    cap.release()
    return frames


async def run_realtime_video(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    image_path: str | None,
    video_path: str | None,
    max_frames: int | None,
    frame_interval: int,
    batch_size: int,
):
    """Streaming video: one commit per batch; send rhythm is controlled by server water level (backpressure), not delay."""
    uri = f"ws://{host}:{port}/v1/realtime_video"

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        if msg.get("type") == "error":
            print(f"Error: {msg.get('error', msg)}")
            return
        if msg.get("type") != "session.created":
            print(f"Unexpected: {msg}")
            return
        initial_water = msg.get("input_video_buffer") or {}
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
                    err_msg = response.get("error", response.get("message", response))
                    print(f"\nError: {err_msg}", flush=True)
                    if response.get("code"):
                        print(f"Code: {response['code']}", flush=True)
                    break
                else:
                    print(f"[Received type={t!r}] {response}", flush=True)

        elif video_path:
            limit_str = f"max {max_frames} frames" if max_frames is not None else "all frames"
            print(f"Loading video: {video_path} ({limit_str}, every {frame_interval} frame(s))")
            frames_b64 = video_frames_to_base64_jpeg(
                video_path,
                max_frames=max_frames,
                frame_interval=frame_interval,
            )
            num_batches = (len(frames_b64) + batch_size - 1) // batch_size if batch_size else 0
            print(
                f"Sending {len(frames_b64)} frames in {num_batches} batch(es) "
                f"(batch_size={batch_size}); send when water level allows (queue_depth < max_queue_size)."
            )
            if max_frames is not None and len(frames_b64) > max_frames:
                print(
                    "Warning: server accepts at most 64 frames per commit. "
                    "Use --max-frames 64 or --frame-interval to send fewer.",
                    flush=True,
                )

            # Single loop: get water level (from last recv), decide whether to send; else recv.
            # Protocol: 4) get water level, decide; 5–6) send one batch (appends + commit); 7) recv until completion.done; 8) repeat.
            queue_depth = 0
            max_queue_size = initial_water.get("max_queue_size", 3)
            batch_index = 0
            received_done_count = 0
            err: str | None = None

            while received_done_count < num_batches and err is None:
                # 获取水位，判断是否需要发送
                if batch_index < num_batches and queue_depth < max_queue_size:
                    # 5–6) 发送一个批次：append 若干帧 + commit
                    batch = frames_b64[batch_index * batch_size : (batch_index + 1) * batch_size]
                    for b64 in batch:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "input_video_buffer.append",
                                    "video": b64,
                                    "format": "image/jpeg",
                                }
                            )
                        )
                    is_final = batch_index == num_batches - 1
                    await ws.send(
                        json.dumps({"type": "input_video_buffer.commit", "final": is_final})
                    )
                    batch_index += 1
                    queue_depth += 1  # optimistic until server sends next water level
                    continue
                # 7) 收一条消息，更新水位或处理 completion/error
                response = json.loads(await ws.recv())
                t = response.get("type")
                if t == "completion.delta":
                    print(response.get("delta", ""), end="", flush=True)
                elif t == "completion.done":
                    print(f"\n\n[Batch {received_done_count + 1}] {response.get('text', '')}")
                    if response.get("usage"):
                        print(f"Usage: {response['usage']}")
                    received_done_count += 1
                    buf = response.get("input_video_buffer")
                    if buf is not None:
                        queue_depth = buf.get("queue_depth", queue_depth)
                        max_queue_size = buf.get("max_queue_size", max_queue_size)
                elif t == "input_video_buffer.water_level":
                    queue_depth = response.get("queue_depth", queue_depth)
                    max_queue_size = response.get("max_queue_size", max_queue_size)
                elif t == "error":
                    err = response.get("error", response.get("message", str(response)))
                    print(f"\nError: {err}", flush=True)
                    if response.get("code"):
                        print(f"Code: {response['code']}", flush=True)
                else:
                    print(f"[Received type={t!r}] {response}", flush=True)
        else:
            print("Provide --image-path or --video-path")
            return


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Video WebSocket client for vLLM"
    )
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to send from video; default None = send all (after sampling).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Send every Nth frame (1=every frame, 25=every 25th frame). Default: 1.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Frames per batch; one commit per batch. Send rhythm is controlled by server water level (backpressure).",
    )
    args = parser.parse_args()

    if not args.image_path and not args.video_path:
        parser.error("Provide at least one of --image-path or --video-path")

    asyncio.run(
        run_realtime_video(
            args.host,
            args.port,
            args.model,
            args.prompt,
            args.image_path,
            args.video_path,
            args.max_frames,
            args.frame_interval,
            args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
