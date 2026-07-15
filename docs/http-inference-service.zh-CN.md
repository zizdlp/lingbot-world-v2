# LingBot World V2 HTTP 推理服务使用手册

本文档描述 `server.py` 和 `client.py` 提供的持久化 HTTP 推理服务，包括参数归属、默认值、输入文件、接口协议、任务生命周期和常见操作。

## 1. 参数归属

持久化 server 启动后会在 8 个 GPU 进程中加载一次模型。需要重新加载模型或重建分布式状态的参数必须由 server 管理；只影响单次生成结果的参数由 client 管理。

### Server 管理

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--task` | `i2v-A14B` | server 实际加载的任务类型 |
| `--ckpt-dir` | `lingbot-world-v2-14b-causal-fast` | 模型权重目录 |
| `--action-path` | `examples/03` | 默认输入目录 |
| `--default-image` | 默认输入目录中的 `image.*` | 默认起始图 |
| `--default-prompt` | `prompt.txt` 或内置 prompt | 默认文本提示词 |
| `--chunk-size` | `4` | 因果推理的时间块大小 |
| `--local-attn-size` | `18` | KV cache 的局部注意力窗口，`-1` 表示不限制 |
| `--sink-size` | `6` | KV cache 中保留的 attention sink 长度 |
| `--ulysses-size` | `8` | Ulysses 序列并行规模，必须与 `WORLD_SIZE` 一致 |
| `--dit-fsdp` | 启用 | 是否对 DiT 使用 FSDP |
| `--t5-fsdp` | 启用 | 是否对 T5 使用 FSDP |
| `--max-attention-size` | 无限制 | 单次注意力可使用的最大 KV 长度 |
| `--max-frame-num` | `361` | client 可请求的最大帧数 |
| `--max-upload-mb` | `32` | 每个上传文件的最大 MiB 数 |
| `--max-queue-size` | `32` | 等待队列的最大任务数 |
| `--retention-hours` | `168` | 成功或失败任务的保留时间 |
| `--data-dir` | `/mnt/data/lingbot-world-service` | 数据库和任务输入快照目录 |
| `--output-dir` | `/mnt/outputs/lingbot-world-v2` | server 生成视频的持久化目录 |

关闭 FSDP 时使用 `--no-dit-fsdp` 或 `--no-t5-fsdp`。模型已经启动后，client 不能覆盖这些参数。

### Client 管理

| 参数 | 省略后的行为 | 含义 |
| --- | --- | --- |
| `--task` | `i2v-A14B` | 声明期望任务；必须与 server 已加载任务一致 |
| `--input-dir` | 使用 server 默认输入 | 一次读取完整样例目录 |
| `--prompt` | 使用输入目录或 server 默认值 | 直接传入 prompt |
| `--prompt-file` | 使用输入目录或 server 默认值 | 从文本文件读取 prompt |
| `--image` | 使用输入目录或 server 默认值 | 起始图覆盖 |
| `--action-path` | 使用 server 默认输入目录 | 包含 `.npy` 输入的 client 本地目录 |
| `--poses` | 逐级回退 | 单独覆盖 `poses.npy` |
| `--intrinsics` | 逐级回退 | 单独覆盖 `intrinsics.npy` |
| `--action` | 逐级回退 | 单独覆盖 `action.npy` |
| `--wasd-action` | 逐级回退 | 单独覆盖 `wasd_action.npy` |
| `--ijkl-action` | 逐级回退 | 单独覆盖 `ijkl_action.npy` |
| `--size` | server 的 `--size` | 输出最大像素面积，可选值见 capabilities |
| `--frame-num` | server 的 `--frame-num` | 请求生成帧数 |
| `--seed` | `42` | 非负随机种子 |
| `--sample-shift` | server 的 `--sample-shift` | diffusion noise schedule shift |
| `--timesteps-index` | server 默认时间步 | causal-fast 使用的采样时间步索引 |
| `--request-id` | 自动生成 UUID | 幂等请求标识 |

`--prompt` 和 `--prompt-file` 互斥。client 本地下载、轮询相关参数不会影响推理结果。

## 2. 七项输入

完整样例目录例如 `examples/05`：

```text
examples/05/
├── prompt.txt
├── image.jpg
├── poses.npy
├── intrinsics.npy
├── action.npy
├── wasd_action.npy
└── ijkl_action.npy
```

| 文件 | 格式 | 含义 | 当前 camera checkpoint 是否消费 |
| --- | --- | --- | --- |
| `prompt.txt` | UTF-8 文本 | 画面内容与动态描述 | 是 |
| `image.jpg` | JPG/JPEG/PNG/WebP | 视频首帧和视觉条件 | 是 |
| `poses.npy` | `(frames, 4, 4)` | OpenCV 坐标系下的相机位姿 | 是 |
| `intrinsics.npy` | `(frames, 4)` | 每帧相机内参 `fx, fy, cx, cy` | 是 |
| `action.npy` | `(frames, 4)` | 原始动作序列 | 否 |
| `wasd_action.npy` | `(frames, 4)` | WASD 动作条件序列 | 否 |
| `ijkl_action.npy` | `(frames, 4)` | IJKL 动作条件序列 | 否 |

三个 action 文件会被 client 上传、server 校验并保存到任务输入目录。当前 `causal_fast` camera checkpoint 的控制维度固定为相机 Plucker 特征，因此实际生成只消费 `poses.npy` 和 `intrinsics.npy`。上传 action 文件不会改变当前 checkpoint 的输出。

## 3. 输入覆盖优先级

Client 端先解析本地输入：

1. 单文件参数，例如 `--poses` 或 `--image`
2. `--action-path`，只用于查找五个 `.npy` 文件
3. `--input-dir`，用于查找全部七项输入
4. 没有找到时不上传该项，由 server 提供默认值

Server 收到请求后逐项合并：

1. 使用 client 上传值
2. 使用 `--action-path` 默认目录中的文件
3. `poses.npy` 和 `intrinsics.npy` 仍然缺失时拒绝请求
4. 三个 action 文件缺失时标记为 `unavailable`，不阻塞 camera 推理

解析后的 prompt、image、参数和数组会形成任务快照。后续修改 server 默认目录不会改变已经入队的任务。

## 4. 启动 Server

推荐使用 8 GPU 启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FSDP_SHARDING_STRATEGY=FULL_SHARD \
NCCL_DEBUG=WARN \
torchrun --nproc_per_node=8 server.py \
  --task i2v-A14B \
  --ckpt-dir lingbot-world-v2-14b-causal-fast \
  --action-path examples/03 \
  --size '480*832' \
  --frame-num 361 \
  --max-frame-num 361 \
  --chunk-size 4 \
  --local-attn-size 18 \
  --sink-size 6 \
  --ulysses-size 8 \
  --dit-fsdp \
  --t5-fsdp \
  --sample-shift 10.0 \
  --timesteps-index '0,250,500,750' \
  --data-dir /mnt/data/lingbot-world-service \
  --output-dir /mnt/outputs/lingbot-world-v2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-queue-size 32 \
  --retention-hours 168
```

也可以直接运行：

```bash
make start_server
```

`--ckpt-dir` 是 server 文件系统中的权重路径，不应由远程 client 提供。

## 5. 查询 Server 能力

在提交任务前查询 server 已加载配置、允许尺寸和默认值：

```bash
python client.py \
  --server http://127.0.0.1:8000 \
  --show-capabilities
```

或直接访问：

```bash
curl http://127.0.0.1:8000/v1/capabilities
```

响应包含：

- 当前 `task`
- 可选尺寸和逐任务默认值
- server 默认输入是否可用
- checkpoint、FSDP、Ulysses、chunk 和 attention 配置
- 单文件上传限制

## 6. 使用 Client

### 使用全部 Server 默认值

```bash
python client.py --server http://127.0.0.1:8000
```

该命令使用 server 默认 prompt、image、输入数组和采样参数，等待任务完成并下载到 `output/<job-id>.mp4`。

### 一次提交完整样例目录

```bash
python client.py \
  --server http://127.0.0.1:8000 \
  --task i2v-A14B \
  --input-dir examples/05 \
  --size '480*832' \
  --frame-num 361 \
  --seed 42 \
  --sample-shift 10.0 \
  --timesteps-index '0,250,500,750' \
  --output-dir output
```

`--input-dir examples/05` 会读取该目录下存在的全部七项输入。

### 单独覆盖部分输入

```bash
python client.py \
  --server http://127.0.0.1:8000 \
  --prompt-file /path/to/prompt.txt \
  --image /path/to/image.png \
  --poses /path/to/poses.npy \
  --seed 1234
```

此例只上传 prompt、image 和 poses。intrinsics 及其他数组由 server 默认目录补齐。

### 只提交任务，不等待下载

```bash
python client.py \
  --server http://127.0.0.1:8000 \
  --input-dir examples/05 \
  --no-wait
```

client 会输出任务状态 URL。稍后可使用该 URL 查询任务。

### 幂等重试

```bash
python client.py \
  --server http://127.0.0.1:8000 \
  --request-id render-scene-001 \
  --input-dir examples/05
```

相同 `request_id` 再次提交时，server 返回原任务，不会重复入队。重试时应保持请求内容一致。

## 7. HTTP API

### `GET /health`

返回服务状态、GPU worker 数、当前运行任务和队列计数。

```bash
curl http://127.0.0.1:8000/health
```

### `GET /v1/capabilities`

返回 server 固定参数、逐任务参数默认值和默认输入可用性。

### `POST /v1/jobs`

创建任务。所有字段均可省略，省略后使用 server 默认值。

```json
{
  "request_id": "render-scene-001",
  "task": "i2v-A14B",
  "prompt": "A first-person flight through a dense jungle.",
  "size": "480*832",
  "frame_num": 361,
  "seed": 42,
  "sample_shift": 10.0,
  "timesteps_index": [0, 250, 500, 750],
  "image_name": "image.jpg",
  "image_base64": "<base64>",
  "trajectory": {
    "poses_base64": "<base64>",
    "intrinsics_base64": "<base64>",
    "action_base64": "<base64>",
    "wasd_action_base64": "<base64>",
    "ijkl_action_base64": "<base64>"
  }
}
```

新任务返回 HTTP `202 Accepted`。相同 `request_id` 已存在时返回 HTTP `200 OK` 和原任务。

任务响应示例：

```json
{
  "id": "8f3a2c7d1e4b5a60",
  "request_id": "render-scene-001",
  "status": "queued",
  "queue_position": 1,
  "task": "i2v-A14B",
  "prompt": "A first-person flight through a dense jungle.",
  "size": "480*832",
  "frame_num": 361,
  "seed": 42,
  "sample_shift": 10.0,
  "timesteps_index": [0, 250, 500, 750],
  "input_sources": {
    "prompt": "client",
    "image": "client",
    "poses.npy": "client",
    "intrinsics.npy": "client",
    "action.npy": "client",
    "wasd_action.npy": "client",
    "ijkl_action.npy": "client"
  },
  "trajectory_source": "uploaded",
  "attempts": 0,
  "created_at": "2026-07-15T08:00:00Z",
  "started_at": null,
  "finished_at": null,
  "error": null,
  "result_path": null,
  "video_url": null
}
```

`input_sources` 的值：

| 值 | 含义 |
| --- | --- |
| `client` | 由当前请求上传 |
| `default` | 来自 server 默认输入 |
| `unavailable` | 可选输入不存在 |

`trajectory_source` 可能为 `default`、`mixed` 或 `uploaded`。

### `GET /v1/jobs?limit=20`

按创建顺序倒序列出任务。`limit` 范围为 1 到 100。

### `GET /v1/jobs/<job-id>`

查询单个任务。状态可能为：

| 状态 | 含义 |
| --- | --- |
| `queued` | 正在等待 GPU worker |
| `running` | 正在生成 |
| `succeeded` | 已生成，可下载视频 |
| `failed` | 生成失败，查看 `error` |

### `GET /v1/jobs/<job-id>/video`

成功任务返回 `video/mp4`。任务未完成时返回 HTTP `409 Conflict`。

```bash
curl --fail \
  --output result.mp4 \
  http://127.0.0.1:8000/v1/jobs/<job-id>/video
```

## 8. 状态码和错误

| 状态码 | 场景 |
| --- | --- |
| `200` | 查询成功，或幂等请求命中原任务 |
| `202` | 新任务已创建并进入队列 |
| `400` | 参数、Base64、图片或数组格式不合法 |
| `404` | 接口或任务不存在 |
| `409` | 视频尚未就绪 |
| `429` | 等待队列已满 |
| `503` | 服务正在停止，不再接收任务 |
| `500` | server 内部错误 |

错误响应统一为：

```json
{
  "error": "error description"
}
```

## 9. 持久化和清理

任务输入快照：

```text
<data-dir>/jobs/YYYY/MM/DD/<job-id>/
├── input.<jpg|jpeg|png|webp>
├── request.json
└── inputs/
    ├── poses.npy
    ├── intrinsics.npy
    ├── action.npy
    ├── wasd_action.npy
    └── ijkl_action.npy
```

可选 action 文件不存在时不会出现在快照中。

SQLite 数据库：

```text
<data-dir>/jobs.sqlite3
```

生成视频：

```text
<output-dir>/YYYY/MM/DD/<job-id>.mp4
```

日期目录使用 UTC。成功或失败任务超过 `--retention-hours` 后，其输入快照、输出视频和数据库记录会一起删除。运行中断的任务在 server 重启后会根据输入快照重新排队。

## 10. 稳定使用建议

1. 生产环境启动前调用 `/v1/capabilities`，确认 task、尺寸和最大帧数。
2. 为业务任务生成稳定且唯一的 `request_id`，网络重试时复用该值。
3. 将 `--data-dir` 和 `--output-dir` 放在持久化磁盘。
4. 不要向 client 暴露 `--ckpt-dir` 或 server 文件系统路径。
5. `poses.npy` 和 `intrinsics.npy` 必须具有相同帧数，并至少覆盖一个完整 chunk。
6. `frame_num` 最好使用 `4n+1`。模型会向下对齐并受 poses 长度及 chunk 大小限制，实际帧数可能小于请求值。
7. 提交高分辨率或长视频前确认 GPU 显存和队列容量。
8. 当前 action 文件仅作为已校验、可持久化的接口输入；在 action-control checkpoint 和对应模型通路接入前，不应把它们视为已生效控制。
