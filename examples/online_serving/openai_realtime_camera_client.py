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

  # Gradio UI: set all parameters in the interface, send new prompts anytime
  python openai_realtime_camera_client.py --gradio

  # Gradio with CLI defaults (override in UI)
  python openai_realtime_camera_client.py --gradio --port 8000 --camera-id 0

Press Ctrl+C to stop.
"""

import argparse
import asyncio
import base64
import collections
import io
import json
import os
import queue
import sys
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

    # cv2.setLogLevel exists in OpenCV 4.8+; Gradio may call it
    if not hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel = lambda _: None  # no-op for older OpenCV
except ImportError:
    cv2 = None

# Shared state
frame_queue: collections.deque | None = None
is_capturing = False
dropped_count = 0
latest_display_frame = None  # RGB numpy array for Gradio
response_text = ""  # Model output for Gradio

# For Gradio: prompts to send (session.update) when user sends new prompt
_prompt_queue: queue.Queue | None = None


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
    # On Windows, CAP_DSHOW often works better for webcams
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
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
    prompt_queue: queue.Queue | None = None,
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
        try:
            async with websockets.connect(uri, open_timeout=10) as ws:
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

                # Background task: when user sends new prompt via UI, send session.update + empty commit
                prompt_task: asyncio.Task | None = None
                if prompt_queue is not None:

                    async def prompt_sender() -> None:
                        while True:
                            try:
                                new_prompt = prompt_queue.get_nowait()
                                await ws.send(
                                    json.dumps({
                                        "type": "session.update",
                                        "model": model,
                                        "prompt": new_prompt or "",
                                    })
                                )
                                await ws.send(
                                    json.dumps({"type": "input_video_buffer.commit", "final": False})
                                )
                                if update_display_frame:
                                    _append_response(f"\n[Prompt updated] {new_prompt}\n")
                                else:
                                    print(f"\n[Prompt updated] {new_prompt}", flush=True)
                            except queue.Empty:
                                pass
                            await asyncio.sleep(0.05)

                    prompt_task = asyncio.create_task(prompt_sender())

                queue_depth = 0
                max_queue_size = initial_water.get("max_queue_size", 3)
                received_done_count = 0
                err: str | None = None

                # Qwen2-VL/Qwen3-VL require at least 2 frames per batch; different models have
                # different frame requirements. Client-side: only send when we have ≥2 frames,
                # avoiding single-frame server errors and staying compatible with various models.
                MIN_VIDEO_FRAMES = 2

                try:
                    while err is None:
                        # Collect up to batch_size frames from queue
                        batch: list[str] = []
                        for _ in range(batch_size):
                            try:
                                batch.append(frame_queue.popleft())
                            except IndexError:
                                break

                        # Send when we have enough frames and server has capacity
                        if (
                            len(batch) >= MIN_VIDEO_FRAMES
                            and queue_depth < max_queue_size
                        ):
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
                            # Put batch back for next iteration (couldn't send yet)
                            for b64 in reversed(batch):
                                frame_queue.appendleft(b64)

                        # When queue is empty, don't block on recv
                        if len(frame_queue) == 0:
                            await asyncio.sleep(0.05)
                            continue

                        # Receive message
                        if prompt_queue is not None:
                            try:
                                msg_bytes = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                        else:
                            msg_bytes = await ws.recv()
                        response = json.loads(msg_bytes)
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
                finally:
                    if prompt_task is not None and not prompt_task.done():
                        prompt_task.cancel()
                        try:
                            await prompt_task
                        except asyncio.CancelledError:
                            pass
        except Exception as conn_err:
            err_msg = f"Connection error: {conn_err}\nIs the vLLM server running at {uri}?"
            if update_display_frame:
                _append_response(err_msg)
            else:
                print(err_msg)
            # Keep camera running for preview so user can see it works
            while is_capturing:
                await asyncio.sleep(0.2)
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
                host,
                port,
                model,
                prompt,
                camera_id,
                frame_interval,
                batch_size,
                queue_size,
                quality,
                prompt_queue=_prompt_queue,
                update_display_frame=True,
            )
        )
    except Exception as e:
        print(f"WebSocket error: {e}")


def send_prompt(prompt: str) -> tuple:
    """Gradio callback: enqueue new prompt for session.update."""
    global _prompt_queue
    if _prompt_queue is not None and prompt and prompt.strip():
        _prompt_queue.put(prompt.strip())
    return gr.update(value="")  # Clear the input


def start_camera_service(
    host: str,
    port: int,
    model: str,
    prompt: str,
    camera_id: int,
    frame_interval: int,
    batch_size: int,
    queue_size: int,
    quality: int,
) -> tuple:
    """Start the camera + WebSocket service (Gradio Start button)."""
    global response_text
    response_text = ""
    thread = threading.Thread(
        target=websocket_handler,
        args=(
            host or "localhost",
            int(port) if port else 8000,
            model or "Qwen2.5-VL-7B-Instruct",
            prompt.strip() if prompt else None,
            int(camera_id) if camera_id is not None else 0,
            int(frame_interval) if frame_interval else 1,
            int(batch_size) if batch_size else 16,
            int(queue_size) if queue_size else 64,
            int(quality) if quality else 85,
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
    host: str = "localhost",
    port: int = 8000,
    model: str = "Qwen2.5-VL-7B-Instruct",
    prompt: str | None = None,
    camera_id: int = 0,
    frame_interval: int = 1,
    batch_size: int = 16,
    queue_size: int = 64,
    quality: int = 85,
) -> "gr.Blocks":
    """Create Gradio interface with parameter inputs and prompt send."""
    # Inline JS: stick to bottom unless user scrolls up; when at bottom, keep auto-scrolling
    scroll_js = """
    <script>
    (function(){
      var userScrolledUp = false;
      function isAtBottom(ta){
        return ta.scrollHeight - ta.scrollTop - ta.clientHeight < 10;
      }
      function run(){
        var el = document.getElementById("model-response-output");
        if(!el) return;
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(!ta) return;
        if(!userScrolledUp || isAtBottom(ta)) ta.scrollTop = ta.scrollHeight;
      }
      function onScroll(){
        var el = document.getElementById("model-response-output");
        if(!el) return;
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(!ta) return;
        userScrolledUp = !isAtBottom(ta);
      }
      function start(){
        var el = document.getElementById("model-response-output");
        if(!el){ setTimeout(start, 200); return; }
        var ta = el.tagName==="TEXTAREA" ? el : el.querySelector("textarea");
        if(ta) ta.addEventListener("scroll", onScroll, {passive:true});
        setInterval(run, 100);
      }
      if(document.readyState==="loading")
        document.addEventListener("DOMContentLoaded", start);
      else
        setTimeout(start, 500);
    })();
    </script>
    """
    with gr.Blocks(title="Real-time Camera Vision", head=scroll_js) as demo:
        gr.Markdown("# Real-time Camera Vision")
        gr.Markdown(
            "Set parameters below (or use defaults from command line), click **Start**. "
            "You can send new prompts at any time; each sends a `session.update` to the server."
        )
        with gr.Row():
            with gr.Column(scale=1):
                host_in = gr.Textbox(label="Host", value=host or "localhost")
                port_in = gr.Number(label="Port", value=port or 8000, precision=0)
                model_in = gr.Textbox(
                    label="Model",
                    value=model or "Qwen2.5-VL-7B-Instruct",
                )
                prompt_in = gr.Textbox(
                    label="Initial Prompt (optional)",
                    value=prompt or "",
                    placeholder="Describe what you see.",
                    lines=2,
                )
                camera_id_in = gr.Number(
                    label="Camera ID",
                    value=camera_id,
                    precision=0,
                )
                frame_interval_in = gr.Number(
                    label="Frame Interval",
                    value=frame_interval,
                    precision=0,
                )
                batch_size_in = gr.Number(
                    label="Batch Size",
                    value=batch_size,
                    precision=0,
                )
                queue_size_in = gr.Number(
                    label="Queue Size",
                    value=queue_size,
                    precision=0,
                )
                quality_in = gr.Number(
                    label="Quality (1-100)",
                    value=quality,
                    precision=0,
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False)
            with gr.Column(scale=1):
                image_out = gr.Image(label="Camera", height=400)
                text_out = gr.Textbox(
                    label="Model Response",
                    lines=12,
                    max_lines=25,
                    elem_id="model-response-output",
                )
                gr.Markdown("### Send new prompt (session.update)")
                with gr.Row():
                    new_prompt_in = gr.Textbox(
                        label="New Prompt",
                        placeholder="Type and click Send to update prompt",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", scale=1)

        start_btn.click(
            start_camera_service,
            inputs=[
                host_in,
                port_in,
                model_in,
                prompt_in,
                camera_id_in,
                frame_interval_in,
                batch_size_in,
                queue_size_in,
                quality_in,
            ],
            outputs=[start_btn, stop_btn],
        )
        stop_btn.click(
            stop_camera_service,
            outputs=[start_btn, stop_btn],
        )
        send_btn.click(
            send_prompt,
            inputs=[new_prompt_in],
            outputs=[new_prompt_in],
        )
        # Gradio 6+: use Timer instead of demo.load(every=...)
        timer = gr.Timer(value=0.1)  # Refresh every 100ms
        timer.tick(
            get_latest_display,
            outputs=[image_out, text_out],
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
        global _prompt_queue
        _prompt_queue = queue.Queue()
        demo = create_gradio_demo(
            host=args.host,
            port=args.port,
            model=args.model,
            prompt=args.prompt,
            camera_id=args.camera_id,
            frame_interval=args.frame_interval,
            batch_size=args.batch_size,
            queue_size=args.queue_size,
            quality=args.quality,
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
