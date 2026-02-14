# 流式视频输入完整方案 (Streaming Video Input Design)

基于当前代码的流式视频理解方案说明：协议、数据模型、服务端/客户端行为与背压机制。

---

## 1. 概述与目标

- **目标**：支持客户端按「批」持续推送视频帧，服务端对**每一批**做一次多模态理解并流式返回文本，实现**低延迟、实时**的视频理解，且单次请求不超出 `max_model_len`。
- **约束**：发送节奏由服务端**水位（water level）**驱动，避免固定延迟或队列溢出；每批独立推理，不合并多批为一次长请求。

---

## 2. 相关文件说明

| 路径 | 说明 |
|------|------|
| **vllm/entrypoints/openai/video_realtime/api_router.py** | 注册 WebSocket 路由 `GET /v1/realtime_video`，在连接建立后创建 `RealtimeVideoConnection` 并调用 `handle_connection()`；提供 `attach_router()` 与 `init_realtime_video_state()` 供 FastAPI 挂载路由与初始化 `OpenAIServingRealtimeVideo`。文档注释中写有完整 7 步协议说明。 |
| **vllm/entrypoints/openai/video_realtime/video_connection.py** | 单连接处理器：维护 `_frame_buffer`（未 commit 的帧）、有界队列 `_video_batch_queue`、prompt 与 model 校验状态；处理 `session.update`、`input_video_buffer.append`、`input_video_buffer.commit`，在首次 commit 时启动 `_run_generation_loop()`，按批从队列取数据并调用引擎生成，向客户端发送 `completion.delta` / `completion.done` 及水位事件。 |
| **vllm/entrypoints/openai/video_realtime/video_serving.py** | 流式视频的 serving 层：实现 `stream_video_realtime(video_batch_queue, prompt_text)` 异步生成器，从队列逐个取 batch（遇 `None` 结束），将每个 batch 转为 `StreamingInput`（空 batch 为纯文本，非空为 Qwen 风格 prompt + `multi_modal_data["video"]`），供 connection 层每批调用一次 `engine_client.generate()`。 |
| **vllm/entrypoints/openai/video_realtime/protocol.py** | 协议事件与 Pydantic 模型：客户端事件 `InputVideoBufferAppend`、`InputVideoBufferCommit`；服务端事件 `SessionCreated`、`InputVideoBufferWaterLevel`、`InputVideoBufferWaterLevelEvent`、`CompletionDelta`、`CompletionDone`、`ErrorEvent`。水位结构包含 `queue_depth`、`max_queue_size`、`buffer_frames`。 |
| **vllm/entrypoints/openai/video_realtime/__init__.py** | 包初始化文件（当前无导出）。 |
| **examples/online_serving/openai_realtime_video_client.py** | 示例客户端：连接 `ws://host/v1/realtime_video`，发送 `session.update` 后按协议循环——根据水位判断是否发送下一批（多次 append + 一次 commit），并接收 `completion.delta` / `completion.done` / `input_video_buffer.water_level`；支持从本地图片或视频文件读取帧并 base64 发送。 |

---

## 3. 数据模型

| 概念 | 含义 | 代码对应 |
|------|------|-----------|
| **帧 (Frame)** | 单张图像，base64 编码（如 JPEG） | 一次 `input_video_buffer.append`；服务端解码为 PIL Image 存入 `_frame_buffer` |
| **批 (Batch)** | 一次 commit 包含的所有帧 | `list` of PIL Images；服务端 `_frame_buffer` 在 commit 时整体取出，作为队列的**一个元素** |
| **队列 (Queue)** | 有界队列，元素 = 一个 batch 或 EOS | `_video_batch_queue: asyncio.Queue[list \| None]`；每个单元 = 一个 batch；`None` 表示流结束 |

关系：**多帧 → 1 个 batch → 队列中的 1 个单元**。队列可容纳多个 batch（如 `maxsize=4`），客户端根据水位决定是否继续发送下一批。

---

## 4. WebSocket 协议

- **端点**：`ws://host/v1/realtime_video`
- **流程**：
  1. 客户端连接。
  2. 服务端发送 `session.created`，并在 `input_video_buffer` 中携带初始水位：`queue_depth=0`，`max_queue_size`（如 3，即 `queue maxsize - 1`，为 EOS 预留一槽）。
  3. 客户端发送 `session.update`，携带 `model` 及可选 `prompt`。
  4. **获取水位、判断是否发送**：客户端根据当前 `queue_depth` 与 `max_queue_size`，仅当 `queue_depth < max_queue_size` 时发送下一批。
  5. **发送一个批次**：多次 `input_video_buffer.append`（每帧一条），再发一条 `input_video_buffer.commit`（可选 `final: true` 表示最后一批）。
  6. 服务端对该批做一次推理，先流式发送 `completion.delta`，再发送 `completion.done`，并在 `completion.done` 的 `input_video_buffer` 中带上最新水位；每次 commit 后还会发一条 `input_video_buffer.water_level`。
  7. 重复步骤 4～6，最后一批 commit 时置 `final: true`。

- **文本-only / 先文本后视频**：可先 `session.update` 带 prompt，再发**空** commit（buffer 中无帧）做一次纯文本轮；或先 append 若干帧再 commit 做「文本 + 视频」理解。

---

## 5. 服务端架构

### 5.1 模块划分

- **api_router.py**：注册 WebSocket 路由 `/v1/realtime_video`，创建 `RealtimeVideoConnection` 并交给其 `handle_connection()`。
- **video_connection.py**：单连接生命周期与事件处理。
  - **状态**：`_frame_buffer`（当前未 commit 的帧）、`_video_batch_queue`（batch 队列）、`_prompt_text`、`_is_model_validated` 等。
  - **事件**：`session.update` → 校验 model、更新 prompt；`input_video_buffer.append` → 解码帧并追加到 `_frame_buffer`（超过 `max_frames_per_commit` 则报错）；`input_video_buffer.commit` → 将当前 buffer 打成**一个 batch** 放入队列，若 `final` 则再放入 `None`（EOS），并发送 `input_video_buffer.water_level`，必要时启动 generation 任务。
- **video_serving.py**：将「队列中的 batch」转为引擎可消费的 `StreamingInput`。
  - **stream_video_realtime(queue, prompt_text)**：异步生成器，从队列中逐个取 batch（遇 `None` 结束），每个 batch 构造一个 `StreamingInput`（空 batch 为纯文本；非空为 Qwen 风格 prompt + `multi_modal_data["video"]`），yield 给调用方。
- **protocol.py**：定义所有事件类型（客户端/服务端）及水位结构 `InputVideoBufferWaterLevel`（`queue_depth`, `max_queue_size`, `buffer_frames`）。

### 5.2 推理流程（每批一次 generate）

- Connection 在**首次**收到 commit 时启动一个长期任务 `_run_generation_loop()`。
- 该循环消费 `stream_video_realtime(...)`：每次 `__anext__()` 从队列取一个 batch（阻塞直到有数据或 EOS），得到**一个** `StreamingInput`。
- 对该 `StreamingInput` 调用一次 `engine_client.generate(prompt=one_input(), ...)`，即**一次请求只包含当前 batch**，避免超长上下文。
- 流式消费 generate 的输出，向客户端发送 `completion.delta`；本批结束后发送 `completion.done`，并在其中附带当前 `InputVideoBufferWaterLevel`（`queue_depth=qsize()`, `max_queue_size=maxsize-1`, `buffer_frames=len(_frame_buffer)`）。
- 循环直到生成器结束（遇到队列中的 `None`）。

### 5.3 背压与水位

- **服务端**：队列有界（如 `maxsize=4`）。commit 时 `await _video_batch_queue.put(batch)`，队列满时阻塞，从而对「过快发送」的客户端形成背压。
- **对外水位**：配置项 `video_batch_queue_maxsize` 表示「允许客户端在途的 batch 数」；物理队列容量为 `maxsize + 1`，多出的一槽专用于 EOS 的 `None`，因此 `put(None)` 不会因满而阻塞。对外发送的 `max_queue_size` 即为该配置值（不再减 1）。
- 每次 commit 后、每次 `completion.done` 时都会向客户端发送最新水位，供客户端判断是否可发下一批。

---

## 6. 客户端行为（示例：openai_realtime_video_client.py）

- **单循环、水位驱动**：不再使用 `batch_delay` 或 `asyncio.Condition`。循环内：
  - **若** `batch_index < num_batches` 且 `queue_depth < max_queue_size`：发送当前批（多次 append + 一次 commit），`batch_index += 1`，本地乐观更新 `queue_depth += 1`，然后 `continue`。
  - **否则**：`await ws.recv()`，根据消息类型更新 `queue_depth` / `max_queue_size`（来自 `completion.done.input_video_buffer` 或 `input_video_buffer.water_level`），并处理 `completion.delta` / `completion.done` / `error`。
- 循环条件：`received_done_count < num_batches && !err`，保证在收到所有批的完成事件或出错时退出。

---

## 7. 水位字段含义（协议层）

| 字段 | 含义 | 用途 |
|------|------|------|
| **queue_depth** | 当前排队中的 batch 数量（`qsize()`） | 客户端仅当 `queue_depth < max_queue_size` 时发送下一批 |
| **max_queue_size** | 允许在途的 batch 上限（服务端为 `maxsize - 1`） | 与 queue_depth 共同决定背压 |
| **buffer_frames** | 当前 append 缓冲区中未 commit 的帧数 | 状态/调试，可选用于限流或展示 |

---

## 8. 端到端时序（单批简化）

1. Client 连接 → Server 发送 `session.created`（含初始水位）。
2. Client 发送 `session.update`（model, prompt）。
3. Client 判断 `queue_depth < max_queue_size` → 发送一批：N 条 append + 1 条 commit。
4. Server 将 N 帧打成 batch 入队，发送 `input_video_buffer.water_level`；若 generation 未启动则启动 `_run_generation_loop`。
5. 循环从队列取到该 batch，构造 `StreamingInput`，调用一次 `engine.generate()`，流式回 `completion.delta`，最后发送 `completion.done`（含新水位）。
6. Client 收到 `completion.done`，更新水位，若还有批次则回到步骤 3；最后一批 commit 带 `final: true`，服务端在队列中放入 `None`，generation 循环结束。

---

## 9. 配置与限制

- **每批最大帧数**：`DEFAULT_MAX_FRAMES_PER_COMMIT = 64`，防止单批过大 OOM。
- **队列容量**：`DEFAULT_VIDEO_BATCH_QUEUE_MAXSIZE = 4` 表示允许客户端在途的 batch 数；物理队列为 `maxsize + 1`（多一槽给 EOS），客户端可见 `max_queue_size = 4`。
- **模型与占位符**：当前 serving 层按 Qwen-VL 风格拼接 prompt 与占位符；其他模型需在 `session.update` 中提供合适 prompt/占位符。

---

## 10. 小结

- **流式输入**：帧级 append，按批 commit；队列中每单元 = 一个 batch。
- **实时理解**：每批一次 `engine.generate()`，单请求不超长，延迟与批大小相关。
- **背压**：有界队列 + 水位协议，客户端根据 `queue_depth` / `max_queue_size` 决定发送节奏，无需固定 delay。
- **协议与实现**：完整流程见 `api_router.py` 的 WebSocket 注释；服务端逻辑见 `video_connection.py` 与 `video_serving.py`，客户端示例见 `examples/online_serving/openai_realtime_video_client.py`。

### 用例说明

**1. 启动 vLLM 服务（带视觉模型）**

```bash
vllm serve /home/user/10T/user/weights/Qwen3-VL-2B-Instruct \
  --served-model-name qwenvl \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.7
```

- 使用 Qwen3-VL-2B-Instruct 等支持视频的视觉模型；`--max-model-len` 需满足单批 prompt 长度。

**2. 运行流式视频客户端**

```bash
python examples/online_serving/openai_realtime_video_client.py \
  --video-path /home/user/10T/user/video_benchmark/Video-Bench/Driving-decision-making/4.mp4 \
  --model qwenvl \
  --max-frames 64 \
  --frame-interval 25 \
  --batch-size 4
```

- `--video-path`：本地视频文件。
- `--model`：与服务端 `--served-model-name` 一致（如 `qwenvl`）。
- `--max-frames 64`：最多发送 64 帧（与每批最大帧数一致，避免服务端报错）。
- `--frame-interval 25`：每 25 帧取 1 帧（例如 25fps 视频约 1 帧/秒）。
- `--batch-size 4`：每 4 帧为一批，每批触发一次 commit 与一次服务端理解；发送节奏由服务端水位背压控制。
