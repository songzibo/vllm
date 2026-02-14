# 流式视频输入完整方案 (Streaming Video Input Design)

---

## 1. 概述与目标

- **目标**：支持客户端按「批」持续推送视频帧，服务端对**每一批**做一次多模态理解并流式返回文本，实现**低延迟、实时**的视频理解。


**流式视频特性说明**：
- **流式持续输入**：支持视频帧按批持续推流输入，而非原有视频一次加载到内存再处理；客户端按水位背压节奏发送，服务端每批独立推理，实现实时理解。
- **提示词动态更新**：根据用户输入提示词，实时改变视频理解任务；客户端可随时发送 `session.update` 更新 prompt，后续批次将采用新提示词，无需重连。

---

## 2. 整体方案与数据流向

### 2.1 整体架构（客户端 + 服务端）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  客户端（三种实现）                                                           │
│  • openai_realtime_video_client.py        （短视频，一次性加载）              │
│  • openai_realtime_video_client_async.py  （大视频，生成器逐帧）              │
│  • openai_realtime_camera_client.py      （实时摄像头，Gradio）               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ WebSocket ws://host/v1/realtime_video
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  服务端                                                                      │
│  api_router → RealtimeVideoConnection (video_connection.py)                 │
│       → OpenAIServingRealtimeVideo (video_serving.py)                        │
│       → engine_client.generate()                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **客户端**：从视频文件或摄像头获取帧，按批 base64 编码后，经 WebSocket 发送；根据服务端水位决定是否继续发送，避免队列溢出。
- **服务端**：接收帧、打成 batch 入队，每批调用一次 `engine.generate()` 做多模态理解，流式返回文本；同时下发水位供客户端背压。

### 2.2 数据流向：客户端 → 服务端（发送）

| 阶段 | 客户端发送 | 服务端处理 |
|------|------------|------------|
| 连接 | 建立 WebSocket 连接 | 接受连接 |
| 会话 | `session.update`（model、可选 prompt） | 校验 model、更新 `_prompt_text` |
| 逐帧 | `input_video_buffer.append`（base64 帧，每帧一条） | 解码为 PIL Image，追加到 `_frame_buffer` |
| 提交批次 | `input_video_buffer.commit`（可选 `final: true`） | 将 buffer 打成 batch 放入 `_video_batch_queue`；若 `final` 则放入 `None`（EOS）；发送 `input_video_buffer.water_level` |
| 循环 | 当 `queue_depth < max_queue_size` 时继续发下一批 | 首次 commit 时启动 `_run_generation_loop`，从队列取 batch、构造 `StreamingInput`、调用 `engine.generate()` |

数据形态：**帧（base64 JPEG）→ append 多条 → commit 一次 → 服务端 batch（PIL Image 列表）→ 队列中的一个单元**。

### 2.3 数据流向：服务端 → 客户端（接收）

| 阶段 | 服务端发送 | 客户端处理 |
|------|------------|------------|
| 会话创建 | `session.created`（含 `input_video_buffer`：`queue_depth`、`max_queue_size`） | 解析初始水位，后续根据 `queue_depth < max_queue_size` 决定是否发下一批 |
| 水位更新 | `input_video_buffer.water_level`（每次 commit 后） | 更新本地 `queue_depth`、`max_queue_size` |
| 流式输出 | `completion.delta`（每次生成一个 token） | 累计并展示文本 |
| 批次完成 | `completion.done`（含 `text`、`usage`、`input_video_buffer` 最新水位） | 输出完整文本，更新水位，若还有批则继续发送 |
| 异常 | `error`（message、code） | 打印错误并退出 |

数据形态：**水位（背压控制） + 流式文本（completion.delta/done）**。

### 2.4 相关文件

| 路径 | 说明 |
|------|------|
| **video_realtime/api_router.py** | 注册 WebSocket 路由，创建 `RealtimeVideoConnection` |
| **video_realtime/video_connection.py** | 单连接处理：帧缓冲、队列、prompt、事件路由、generation 循环 |
| **video_realtime/video_serving.py** | `stream_video_realtime`：队列 → `StreamingInput`，支持 `prompt_getter` |
| **video_realtime/protocol.py** | 协议事件与 Pydantic 模型 |
| **openai_realtime_video_client.py** | 同步视频客户端（短视频） |
| **openai_realtime_video_client_async.py** | 异步视频客户端（大视频，生成器） |
| **openai_realtime_camera_client.py** | 实时摄像头客户端（Gradio） |

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
  2. 服务端发送 `session.created`，并在 `input_video_buffer` 中携带初始水位：`queue_depth=0`，`max_queue_size`（如 4，即允许客户端在途的 batch 数；物理队列容量多一槽用于 EOS）。
  3. 客户端发送 `session.update`，携带 `model` 及可选 `prompt`。
  4. **获取水位、判断是否发送**：客户端根据当前 `queue_depth` 与 `max_queue_size`，仅当 `queue_depth < max_queue_size` 时发送下一批。
  5. **发送一个批次**：多次 `input_video_buffer.append`（每帧一条），再发一条 `input_video_buffer.commit`（可选 `final: true` 表示最后一批）。
  6. 服务端对该批做一次推理，先流式发送 `completion.delta`，再发送 `completion.done`，并在 `completion.done` 的 `input_video_buffer` 中带上最新水位；每次 commit 后还会发一条 `input_video_buffer.water_level`。
  7. 重复步骤 4～6，最后一批 commit 时置 `final: true`。

- **文本-only / 先文本后视频**：可先 `session.update` 带 prompt，再发**空** commit（buffer 中无帧）做一次纯文本轮；或先 append 若干帧再 commit 做「文本 + 视频」理解。
- **prompt 动态更新**：客户端可在任意时刻发送 `session.update` 并携带新 `prompt`；服务端通过 `prompt_getter` 在每批处理时读取当前 prompt，后续批次将使用新 prompt，无需重连。

---

## 5. 服务端架构

### 5.1 模块划分

- **api_router.py**：注册 WebSocket 路由 `/v1/realtime_video`，创建 `RealtimeVideoConnection` 并交给其 `handle_connection()`。
- **video_connection.py**：单连接生命周期与事件处理。
  - **状态**：`_frame_buffer`（当前未 commit 的帧）、`_video_batch_queue`（batch 队列）、`_prompt_text`、`_is_model_validated` 等。
  - **事件**：`session.update` → 校验 model、更新 prompt；`input_video_buffer.append` → 解码帧并追加到 `_frame_buffer`（超过 `max_frames_per_commit` 则报错）；`input_video_buffer.commit` → 将当前 buffer 打成**一个 batch** 放入队列，若 `final` 则再放入 `None`（EOS），并发送 `input_video_buffer.water_level`，必要时启动 generation 任务。
- **video_serving.py**：将「队列中的 batch」转为引擎可消费的 `StreamingInput`。
  - **stream_video_realtime(queue, prompt_text=..., prompt_getter=...)**：异步生成器，从队列中逐个取 batch（遇 `None` 结束），每个 batch 构造一个 `StreamingInput`（空 batch 为纯文本；非空为 Qwen 风格 prompt + `multi_modal_data["video"]`），yield 给调用方。若提供 `prompt_getter`，则每批处理时调用以获取当前 prompt，使 `session.update` 的 prompt 在运行中的 generation 循环中生效。
- **protocol.py**：定义所有事件类型（客户端/服务端）及水位结构 `InputVideoBufferWaterLevel`（`queue_depth`, `max_queue_size`, `buffer_frames`）。

### 5.2 推理流程（每批一次 generate）

- Connection 在**首次**收到 commit 时启动一个长期任务 `_run_generation_loop()`。
- 该循环消费 `stream_video_realtime(..., prompt_getter=lambda: self._prompt_text)`：每次 `__anext__()` 从队列取一个 batch（阻塞直到有数据或 EOS），得到**一个** `StreamingInput`。每批处理时通过 `prompt_getter` 读取当前 prompt，因此 `session.update` 的 prompt 变更会在后续批次生效。
- 对该 `StreamingInput` 调用一次 `engine_client.generate(prompt=one_input(), ...)`，即**一次请求只包含当前 batch**，避免超长上下文。
- 流式消费 generate 的输出，向客户端发送 `completion.delta`；本批结束后发送 `completion.done`，并在其中附带当前 `InputVideoBufferWaterLevel`（`queue_depth=qsize()`, `max_queue_size=video_batch_queue_maxsize`, `buffer_frames=len(_frame_buffer)`）。
- 循环直到生成器结束（遇到队列中的 `None`）。

### 5.3 背压与水位

- **服务端**：队列有界（如 `maxsize=4`）。commit 时 `await _video_batch_queue.put(batch)`，队列满时阻塞，从而对「过快发送」的客户端形成背压。
- **对外水位**：配置项 `video_batch_queue_maxsize` 表示「允许客户端在途的 batch 数」；物理队列容量为 `maxsize + 1`，多出的一槽专用于 EOS 的 `None`，因此 `put(None)` 不会因满而阻塞。对外发送的 `max_queue_size` 即为该配置值（不再减 1）。
- 每次 commit 后、每次 `completion.done` 时都会向客户端发送最新水位，供客户端判断是否可发下一批。

---

## 6. 客户端实现与使用方式

### 6.1 共通行为（所有客户端）

- **单循环、水位驱动**：循环内根据 `queue_depth < max_queue_size` 决定是否发送下一批；否则 `await ws.recv()` 等待服务端消息并更新水位。
- **协议一致**：均连接 `ws://host/v1/realtime_video`，发送 `session.update` 后按 append + commit 循环，接收 `completion.delta` / `completion.done` / `input_video_buffer.water_level`。

### 6.2 三种客户端对比

| 客户端 | 适用场景 | 帧来源 | 特点 |
|--------|----------|--------|------|
| **openai_realtime_video_client.py** | 短视频文件 | 本地视频，一次性加载到内存 | 实现简单，适合小视频；大视频会 OOM |
| **openai_realtime_video_client_async.py** | 大视频文件 | 本地视频，生成器逐帧读取 | 使用 `video_frames_to_base64_jpeg_generator` + `run_in_executor` 批量拉取，内存占用低；支持 `-v` 调试 |
| **openai_realtime_camera_client.py** | 实时摄像头 | OpenCV 摄像头，帧队列（满则 drop-oldest） | Gradio UI、界面内配置、`session.update` 动态改 prompt、Model Response 智能滚动 |

---

## 7. 水位字段含义（协议层）

| 字段 | 含义 | 用途 |
|------|------|------|
| **queue_depth** | 当前排队中的 batch 数量（`qsize()`） | 客户端仅当 `queue_depth < max_queue_size` 时发送下一批 |
| **max_queue_size** | 允许在途的 batch 上限（即 `video_batch_queue_maxsize`） | 与 queue_depth 共同决定背压 |
| **buffer_frames** | 当前 append 缓冲区中未 commit 的帧数 | 状态/调试，可选用于限流或展示 |

---

## 8. WebSocket 保活机制（流式视频特需）

### 8.1 问题与原因

- **现象**：长时间 camera 流或大视频推理时，出现 `keepalive ping timeout; no close frame received`，连接被关闭。
- **原因**：uvicorn 底层使用 websockets 库的协议级 ping/pong 保活。服务端每隔 `ws_ping_interval` 发送 ping，若在 `ws_ping_timeout` 内未收到客户端的 pong，则关闭连接。单批视频推理（`engine.generate`）可能耗时 30 秒乃至更久，期间无应用层消息，原 uvicorn 默认 20 秒超时过短。

### 8.2 服务端修改

- **cli_args.py**（FrontendArgs）：新增 `--ws-ping-interval`、`--ws-ping-timeout`，默认 60.0 秒（原 uvicorn 默认 20.0）。
- **api_server.py**：`run_server_worker` 中将 `args.ws_ping_interval`、`args.ws_ping_timeout` 传入 `serve_http`，最终到达 uvicorn.Config。
- **推荐**：长时间 camera 流可设为 `--ws-ping-interval 120 --ws-ping-timeout 120` 或更高。

### 8.3 断开时的优雅处理（video_connection.py）

- **_send**：捕获 `WebSocketDisconnect`、`ClientDisconnected`，设置 `_is_connected = False`，仅以 `logger.debug` 记录，避免 ERROR 级日志。
- **推理循环**：对上述异常单独处理，仅打 debug 日志；真实错误才打 exception 并尝试 `_send_error`；若 `_send_error` 时连接已断，则静默忽略。

---

## 9. 端到端时序（单批简化）

1. Client 连接 → Server 发送 `session.created`（含初始水位）。
2. Client 发送 `session.update`（model, prompt）。
3. Client 判断 `queue_depth < max_queue_size` → 发送一批：N 条 append + 1 条 commit。
4. Server 将 N 帧打成 batch 入队，发送 `input_video_buffer.water_level`；若 generation 未启动则启动 `_run_generation_loop`。
5. 循环从队列取到该 batch，构造 `StreamingInput`，调用一次 `engine.generate()`，流式回 `completion.delta`，最后发送 `completion.done`（含新水位）。
6. Client 收到 `completion.done`，更新水位，若还有批次则回到步骤 3；最后一批 commit 带 `final: true`，服务端在队列中放入 `None`，generation 循环结束。

---

## 10. 配置与限制

- **每批最大帧数**：`DEFAULT_MAX_FRAMES_PER_COMMIT = 64`，防止单批过大 OOM。
- **队列容量**：`DEFAULT_VIDEO_BATCH_QUEUE_MAXSIZE = 4` 表示允许客户端在途的 batch 数；物理队列为 `maxsize + 1`（多一槽给 EOS），客户端可见 `max_queue_size = 4`。
- **模型与占位符**：当前 serving 层按 Qwen-VL 风格拼接 prompt 与占位符；其他模型需在 `session.update` 中提供合适 prompt/占位符。
- **WebSocket keepalive**：详见第 8 节。流式视频特需，默认 60 秒；长时间流建议 120 秒或更高。

---

## 11. 小结

- **流式输入**：帧级 append，按批 commit；队列中每单元 = 一个 batch。
- **实时理解**：每批一次 `engine.generate()`，单请求不超长，延迟与批大小相关。
- **背压**：有界队列 + 水位协议，客户端根据 `queue_depth` / `max_queue_size` 决定发送节奏，无需固定 delay。
- **协议与实现**：完整流程见 `api_router.py` 的 WebSocket 注释；服务端逻辑见 `video_connection.py` 与 `video_serving.py`。客户端示例：`openai_realtime_video_client.py`（短视频）、`openai_realtime_video_client_async.py`（大视频/低内存）、`openai_realtime_camera_client.py`（实时摄像头 + Gradio）。

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
- 长时间 camera 流建议加上：`--ws-ping-interval 120 --ws-ping-timeout 120`，避免 keepalive 超时断开。

**2. 运行流式视频客户端（三种方式）**

**方式一：短视频文件（一次性加载到内存）**

```bash
python examples/online_serving/openai_realtime_video_client.py \
  --host <vllm_hostip> \
  --port <vllm_port> \
  --model qwenvl \
  --video-path /path/to/video.mp4 \
  --max-frames 64 \
  --frame-interval 25 \
  --batch-size 4
```

**方式二：大视频文件（生成器逐帧，低内存）**

```bash
python examples/online_serving/openai_realtime_video_client_async.py \
  --host <vllm_hostip> \
  --port <vllm_port> \
  --model qwenvl \
  --video-path /path/to/large.mp4 \
  --max-frames 64 \
  --frame-interval 25 \
  --batch-size 16
```

**方式三：实时摄像头 + Gradio 界面**

```bash
python examples/online_serving/openai_realtime_camera_client.py \
  --host <vllm_hostip> \
  --port <vllm_port> \
  --model qwenvl \
  --batch-size 32 \
  --gradio
```

**参数说明**：`--host` / `--port` 为 vLLM 服务地址；`--model` 与服务端 `--served-model-name` 一致。视频方式需 `--video-path`；`--max-frames 64` 与每批最大帧数一致；`--frame-interval` 为采样间隔（如 25 表示每 25 帧取 1 帧）；`--batch-size` 为每批帧数，发送节奏由服务端水位背压控制。
