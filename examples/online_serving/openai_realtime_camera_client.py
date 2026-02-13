# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demo client for the vLLM Realtime Video WebSocket API using live camera.

Reads frames from the camera, samples every Nth frame (--frame-interval),
and sends to the server. When the client-side frame queue is full, old
frames are discarded and newer frames are kept (drop-oldest policy).

Before running, start vLLM with a vision model that supports video, e.g.:

    vllm serve Qwen2.5-VL-7B-Instruct --enforce-eager

Requirements:
- websockets
- Pillow
- opencv-python
- gradio (for --gradio UI)

Usage:
  # Default camera (device 0), every frame
  python openai_realtime_camera_client.py

  # Every 10th frame (e.g. ~3 fps for 30fps camera)
  python openai_realtime_camera_client.py --frame-interval 10

  # Custom prompt and camera
  python openai_realtime_camera_client.py --prompt "Describe what you see." --camera-id 0

  # Limit queue size (drop old when full)
  python openai_realtime_camera_client.py --queue-size 32 --frame-interval 5

  # Gradio UI (image + text side by side)
  python openai_realtime_camera_client.py --gradio

Press Ctrl+C to stop.
"""

import argparse
import asyncio
import base64
import collections
import io
import json
import threading

import websockets

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

# Shared state
frame_queue: collections.deque | None = None
is_capturing = False
dropped_count = 0
latest_display_frame = None  # RGB numpy array for Gradio
response_text = ""  # Model output for Gradio

# Config for Gradio (set before launch)
_gradio_config: dict | None = None


def _append_response(s: str) -> None:
    """Append to response_text for Gradio display."""
    global response_text
    response_text += s


def frame_to_base64_jpeg(bgr, quality: int = 85) -> str:
    """Convert BGR frame to base64-encoded JPEG."""
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def camera_capture_loop(
    camera_id: int,
    frame_interval: int,
    queue: "collections.deque[str]",
    quality: int,
    update_display_frame: bool = False,
) -> None:
    """Capture frames from camera, sample, and append to queue. Drops oldest when full."""
    global dropped_count, latest_display_frame
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required. Install with: pip install opencv-python"
        )
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    frame_idx = 0
    try:
        while is_capturing:
            ret, bgr = cap.read()
            if not ret:
                break
            if update_display_frame:
                latest_display_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if frame_idx % frame_interval == 0:
                b64 = frame_to_base64_jpeg(bgr, quality=quality)
                # deque(maxlen=N): when full, append() automatically drops leftmost (oldest)
                if len(queue) >= queue.maxlen:
                    dropped_count += 1
                queue.append(b64)
            frame_idx += 1
    finally:
        cap.release()


async def run_realtime_camera(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
    update_display_frame: bool = False,
):
    """Stream camera frames to realtime video WebSocket."""
    global frame_queue, is_capturing

    frame_queue = collections.deque(maxlen=queue_size)
    is_capturing = True

    uri = f"ws://{host}:{port}/v1/realtime_video"
    capture_thread = threading.Thread(
        target=camera_capture_loop,
        args=(camera_id, frame_interval, frame_queue, quality, update_display_frame),
        daemon=True,
    )
    capture_thread.start()

    print(
        f"Camera {camera_id}: frame_interval={frame_interval}, "
        f"batch_size={batch_size}, queue_size={queue_size} (drop oldest when full)"
    )
    print("Press Ctrl+C to stop.\n")

    try:
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

            queue_depth = 0
            max_queue_size = initial_water.get("max_queue_size", 3)
            received_done_count = 0
            err: str | None = None

            while err is None:
                # Collect up to batch_size frames from queue
                batch: list[str] = []
                for _ in range(batch_size):
                    try:
                        batch.append(frame_queue.popleft())
                    except IndexError:
                        break

                # Send when we have frames and server has capacity
                if batch and queue_depth < max_queue_size:
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
                    # For live camera, we never send final=True until we stop
                    await ws.send(
                        json.dumps({"type": "input_video_buffer.commit", "final": False})
                    )
                    queue_depth += 1
                    continue

                if batch:
                    # Put batch back for next iteration (we couldn't send yet)
                    for b64 in reversed(batch):
                        frame_queue.appendleft(b64)

                # When queue is empty, don't block on recv (server won't send until we send first)
                if len(frame_queue) == 0:
                    await asyncio.sleep(0.05)
                    continue

                # Receive message
                response = json.loads(await ws.recv())
                t = response.get("type")
                if t == "completion.delta":
                    delta = response.get("delta", "")
                    if update_display_frame:
                        _append_response(delta)
                    else:
                        print(delta, end="", flush=True)
                elif t == "completion.done":
                    text = response.get("text", "")
                    if update_display_frame:
                        _append_response(f"\n\n[Batch {received_done_count + 1}] {text}")
                        if response.get("usage"):
                            _append_response(f"\nUsage: {response['usage']}")
                    else:
                        print(f"\n\n[Batch {received_done_count + 1}] {text}")
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
                    err_str = f"\nError: {err}"
                    if response.get("code"):
                        err_str += f"\nCode: {response['code']}"
                    if update_display_frame:
                        _append_response(err_str)
                    else:
                        print(err_str, flush=True)
                else:
                    print(f"[Received type={t!r}] {response}", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        is_capturing = False
        if dropped_count > 0:
            print(f"\nDropped {dropped_count} frame(s) (queue was full).", flush=True)


def websocket_handler(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
) -> None:
    """Run WebSocket + camera in event loop (for Gradio background thread)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_realtime_camera(
                host, port, model, prompt,
                camera_id, frame_interval, batch_size, queue_size, quality,
                update_display_frame=True,
            )
        )
    except Exception as e:
        print(f"WebSocket error: {e}")


def start_camera_service() -> tuple:
    """Start the camera + WebSocket service (Gradio Start button)."""
    global response_text, _gradio_config
    if _gradio_config is None:
        return gr.update(), gr.update()
    response_text = ""
    cfg = _gradio_config
    thread = threading.Thread(
        target=websocket_handler,
        args=(
            cfg["host"], cfg["port"], cfg["model"], cfg["prompt"],
            cfg["camera_id"], cfg["frame_interval"], cfg["batch_size"],
            cfg["queue_size"], cfg["quality"],
        ),
        daemon=True,
    )
    thread.start()
    return gr.update(interactive=False), gr.update(interactive=True)


def stop_camera_service() -> tuple:
    """Stop the camera + WebSocket service (Gradio Stop button)."""
    global is_capturing
    is_capturing = False
    return gr.update(interactive=True), gr.update(interactive=False)


def get_latest_display() -> tuple:
    """Return latest frame and response text for Gradio periodic update."""
    global latest_display_frame, response_text
    img = latest_display_frame
    return img, response_text


def create_gradio_demo(
    host: str,
    port: int,
    model: str,
    prompt: str | None,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int = 85,
) -> "gr.Blocks":
    """Create Gradio interface with image and text side by side."""
    global _gradio_config
    _gradio_config = {
        "host": host,
        "port": port,
        "model": model,
        "prompt": prompt,
        "camera_id": camera_id,
        "frame_interval": frame_interval,
        "batch_size": batch_size,
        "queue_size": queue_size,
        "quality": quality,
    }
    with gr.Blocks(title="Real-time Camera Vision") as demo:
        gr.Markdown("# Real-time Camera Vision")
        gr.Markdown("Click **Start** to capture from camera and stream to the vision model.")

        with gr.Row():
            image_out = gr.Image(label="Camera", height=400)
            text_out = gr.Textbox(label="Model Response", lines=12, max_lines=20)

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", interactive=False)

        start_btn.click(
            start_camera_service,
            inputs=[],
            outputs=[start_btn, stop_btn],
        )
        stop_btn.click(
            stop_camera_service,
            outputs=[start_btn, stop_btn],
        )

        # Periodic update: refresh image and text every 100ms
        demo.load(
            get_latest_display,
            outputs=[image_out, text_out],
            every=0.1,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Video WebSocket client for vLLM (live camera)"
    )
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index. Default: 0.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Send every Nth frame (1=every frame, 10=every 10th frame). Default: 1.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Frames per batch; one commit per batch. Send rhythm is controlled by server water level (backpressure).",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Max frames in client queue. When full, oldest frames are dropped to make room for newer ones. Default: 64.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG encoding quality (1-100). Default: 85.",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio UI (image + text side by side).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link (use with --gradio).",
    )
    args = parser.parse_args()

    if args.frame_interval < 1:
        parser.error("--frame-interval must be >= 1")

    if args.gradio:
        if gr is None:
            raise RuntimeError("gradio is required for --gradio. Install with: pip install gradio")
        demo = create_gradio_demo(
            args.host, args.port, args.model, args.prompt,
            args.camera_id, args.frame_interval, args.batch_size, args.queue_size,
            args.quality,
        )
        demo.launch(share=args.share)
        return

    try:
        asyncio.run(
            run_realtime_camera(
                args.host,
                args.port,
                args.model,
                args.prompt,
                args.camera_id,
                args.frame_interval,
                args.batch_size,
                args.queue_size,
                args.quality,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)


if __name__ == "__main__":
    main()
