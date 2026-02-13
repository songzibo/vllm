# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Async/streaming version of the vLLM Realtime Video WebSocket client.

Uses a generator to yield video frames one-by-one instead of loading the entire
video into memory. This reduces memory consumption for large videos.

Uses sync generator + run_in_executor to pull frames in batches. At most batch_size
frames in memory at any time. No producer task or queue - simpler loop.

Usage: Same as openai_realtime_video_client.py, e.g.
  python openai_realtime_video_client_async.py --video-path /path/to/video.mp4
  python openai_realtime_video_client_async.py --video-path /path/to/video.mp4 --max-frames 32 --frame-interval 10

  # Debug: print logs to stderr
  python openai_realtime_video_client_async.py --video-path /path/to/video.mp4 -v
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


def video_frames_to_base64_jpeg_generator(
    video_path: str,
    max_frames: int | None = None,
    frame_interval: int = 1,
    quality: int = 85,
):
    """Generator that yields base64-encoded JPEG frames one at a time.

    Reduces memory usage by not loading the entire video into memory.
    Yields (frame_b64, frame_index) tuples; frame_index is 0-based within
    the yielded sequence (useful for logging).

    Args:
        video_path: Path to the video file.
        max_frames: Max number of frames to yield; None = no limit. Caller may pass -1 for no limit.
        frame_interval: Yield every Nth frame (1 = every frame, 25 = every 25th frame).
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
    if max_frames is not None and max_frames < 0:
        max_frames = None

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    yielded_count = 0
    try:
        while True:
            if max_frames is not None and yielded_count >= max_frames:
                break
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                yield b64, yielded_count
                yielded_count += 1
            frame_idx += 1
    finally:
        cap.release()


def _pull_batch_from_generator(gen, size: int) -> tuple[list[str], bool]:
    """Sync: pull up to size frames from generator. Returns (batch, exhausted)."""
    batch = []
    for _ in range(size):
        try:
            b64, _ = next(gen)
            batch.append(b64)
        except StopIteration:
            return batch, True
    return batch, False


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        import sys
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


def _handle_server_message(
    response: dict,
    received_done_count: int,
    queue_depth: int,
    max_queue_size: int,
    err: str | None,
) -> tuple[int, int, int, str | None]:
    """Process server message, return (received_done_count, queue_depth, max_queue_size, err)."""
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
    return received_done_count, queue_depth, max_queue_size, err


async def run_realtime_video(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    video_path: str,
    max_frames: int | None,
    frame_interval: int,
    batch_size: int,
    verbose: bool = False,
):
    """Streaming video: consume frames from generator, send in batches with backpressure."""
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

        # Video streaming
        limit_str = f"max {max_frames} frames" if max_frames is not None else "all frames"
        print(
            f"Streaming video: {video_path} ({limit_str}, every {frame_interval} frame(s))"
        )
        print(
            f"Using generator: batch_size={batch_size}, send when queue_depth < max_queue_size."
        )
        if max_frames is not None and max_frames > 64:
            print(
                "Warning: server accepts at most 64 frames per commit. "
                "Use --max-frames 64 or --frame-interval to send fewer.",
                flush=True,
            )

        frame_gen = video_frames_to_base64_jpeg_generator(
            video_path,
            max_frames=max_frames,
            frame_interval=frame_interval,
        )
        loop = asyncio.get_event_loop()
        queue_depth = 0
        max_queue_size = initial_water.get("max_queue_size", 3)
        batches_sent = 0
        received_done_count = 0
        err: str | None = None
        batch_buffer: list[str] = []
        gen_exhausted = False

        while err is None:
            # 1. Fill batch from generator (non-blocking via executor)
            if not gen_exhausted and len(batch_buffer) < batch_size:
                batch, exhausted = await loop.run_in_executor(
                    None, _pull_batch_from_generator, frame_gen, batch_size
                )
                batch_buffer.extend(batch)
                gen_exhausted = exhausted
                _log(verbose, f"pulled {len(batch)} frames, exhausted={exhausted}")

            # 2. Send batch if ready and server has capacity
            has_batch = len(batch_buffer) > 0
            can_send = has_batch and queue_depth < max_queue_size
            batch_full_or_done = has_batch and (
                len(batch_buffer) >= batch_size or gen_exhausted
            )

            if batch_full_or_done and can_send:
                to_send = batch_buffer[:batch_size]
                batch_buffer = batch_buffer[batch_size:]
                is_final = gen_exhausted and len(batch_buffer) == 0

                for b64 in to_send:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_video_buffer.append",
                                "video": b64,
                                "format": "image/jpeg",
                            }
                        )
                    )
                await ws.send(
                    json.dumps({"type": "input_video_buffer.commit", "final": is_final})
                )
                batches_sent += 1
                queue_depth += 1
                _log(verbose, f"sent batch {batches_sent} (is_final={is_final})")
                continue

            # 3. Done: no more frames and all completions received
            if gen_exhausted and not has_batch and received_done_count >= batches_sent:
                break

            # 4. Wait for server message (water_level, completion.done, etc.)
            response = json.loads(await ws.recv())
            (received_done_count, queue_depth, max_queue_size, err) = _handle_server_message(
                response, received_done_count, queue_depth, max_queue_size, err
            )

        print(f"\nStreaming complete: sent {batches_sent} batch(es), received {received_done_count} completion(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Video WebSocket client (async/streaming) for vLLM"
    )
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to send from video. Default or -1 = send entire video.",
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
        help="Frames per batch; one commit per batch. Send rhythm is controlled by server water level.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print debug logs to stderr.",
    )
    args = parser.parse_args()

    max_frames = None if args.max_frames in (None, -1) else args.max_frames

    asyncio.run(
        run_realtime_video(
            args.host,
            args.port,
            args.model,
            args.prompt,
            args.video_path,
            max_frames,
            args.frame_interval,
            args.batch_size,
            args.verbose,
        )
    )


if __name__ == "__main__":
    main()
