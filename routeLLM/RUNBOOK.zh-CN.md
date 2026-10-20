# RouteLLM 本地复现手册

## 已固定的源码状态

| 项目 | 值 |
| --- | --- |
| 上游仓库 | `https://github.com/lm-sys/RouteLLM.git` |
| 分支 | `main` |
| 提交 | `0b64fdafe049e596a3f5657c219329f24af24198` (`create /health endpoint for health checks`) |
| 本地目录 | `D:\router\routeLLM` |
| Python 基础解释器 | `D:\Anaconda\envs\pytorch\python.exe`（Python 3.10.11） |
| PyTorch | `2.8.0+cu126` |
| PyTorch 内置 CUDA | `12.6` |
| 已验证 GPU | `NVIDIA GeForce RTX 3050 Laptop GPU`（compute capability `8.6`） |

源码本身不包含 Git 子模块。安装声明位于 `pyproject.toml`：运行时依赖包括 PyTorch、Transformers、Datasets、OpenAI 与 LiteLLM；`serve` extra 增加 FastAPI/Uvicorn，`eval` extra 增加 Matplotlib、SGLang、Tiktoken。

当前环境已实测 `torch.cuda.is_available()` 为 `True`，且可在 `cuda:0` 上创建张量。以下状态可以随时复核：

```powershell
& 'D:\Anaconda\envs\pytorch\python.exe' -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

> CUDA 由当前 PyTorch wheel 自带；不要仅因系统安装了 CUDA Toolkit 就重装 PyTorch。若上面的命令仍显示 `True`，RouteLLM 的 PyTorch 路径可使用 GPU。

## 1. 创建环境并安装代码

安装脚本直接使用指定的 Anaconda `pytorch` 环境，以复用其中已安装的 CUDA 版 PyTorch，避免重复下载或意外替换该大型依赖。脚本本身不会安装、卸载或改写 `torch`。请从仓库根目录执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

默认仅安装服务所需的 `serve` extra。该步骤会从 PyPI 下载除既有 PyTorch 之外的 Python 依赖，但**不会**下载 Hugging Face 数据集或 RouteLLM 路由器权重。安装完成后，将实际解析到的包版本写入 `.repro\requirements.freeze.txt`，便于复现同一环境。

若确实需要评估功能，再显式执行：

```powershell
.\scripts\bootstrap.ps1 -IncludeEvaluation
```

> 仓库没有发布锁定依赖文件；因此本地复现以此源码提交和首次安装生成的 freeze 文件为准。若需要在另一台机器重建同一环境，可先安装 `pip install -r .repro\requirements.freeze.txt`，再执行 `pip install -e ".[serve,eval]" --no-deps`。

> `pyproject.toml` 的上游声明要求 `numpy<2`，而当前环境按本地要求固定为 `numpy 2.2.6`。因此 `pip check` 会报告这一项版本声明不匹配；重新运行 `bootstrap.ps1` 会把 NumPy 降回 `<2`。在决定是否重跑安装脚本前，先确认是否要保持 NumPy 2.2.6。

## 2. 不下载数据或权重的本地冒烟检查

```powershell
.\scripts\smoke.ps1
```

它只实例化 `random` 路由器并检查两个阈值方向的路由结果；不会发起网络请求，也不会调用任何模型服务。

## 3. 推荐的 `mf` 服务流程

`mf` 是 README 推荐的路由器，也是本机 GPU 的首选路径。其源码会在 CUDA 可用时将矩阵分解路由器自动移动到 `cuda:0`；但每条请求仍要调用 OpenAI Embeddings，所以整体延迟也受网络和 OpenAI API 影响。先在当前 PowerShell 会话提供密钥；不要把密钥写进配置文件或提交到 Git：

```powershell
$env:OPENAI_API_KEY = 'sk-...'
# 以下是强/弱模型供应商之一的示例。也可改用任意 LiteLLM 支持的提供商。
$env:ANYSCALE_API_KEY = '...'

& 'D:\Anaconda\envs\pytorch\python.exe' -m routellm.openai_server `
  --routers mf `
  --config config.example.yaml `
  --strong-model gpt-4-1106-preview `
  --weak-model anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1
```

首次以 `mf` 启动时会下载相应预训练路由器权重到 Hugging Face 缓存。服务启动后可检查：

```powershell
Invoke-RestMethod http://127.0.0.1:6060/health
```

客户端请求中的 `model` 采用 `router-mf-<threshold>` 格式，例如 `router-mf-0.11593`。真正生成回复仍会消耗所选强/弱模型供应商的 API 配额；此外，`mf` 本身固定使用 OpenAI 的 `text-embedding-3-small`，即便强/弱模型是其他供应商也需要 OpenAI API 密钥与相应费用。

### OpenAI 兼容网关的模型别名

MF 路由器的评分模型名必须与其内置训练表完全一致，例如 `gpt-3.5-turbo-0125` 和 `chatglm3-6b`。某些 OpenAI 兼容网关则要求向 LiteLLM 发送带供应商前缀的名称，例如 `openai/chatglm3-6b`。服务提供两个可选参数来分离这两者：

```powershell
--strong-model gpt-3.5-turbo-0125 `
--weak-model chatglm3-6b `
--strong-model-backend openai/gpt-3.5-turbo-0125 `
--weak-model-backend openai/chatglm3-6b
```

未设置 `--*-model-backend` 时，行为与上游版本完全一致：评分名也会作为实际 API 请求名。

## 4. 按当前 CUDA 环境选择路由器

| 目标 | 推荐项 | GPU 作用 | 外部资源 / 限制 |
| --- | --- | --- | --- |
| 验证安装与阈值语义 | `random` | 不使用 GPU | 无下载、无模型服务、无真实路由质量 |
| 首次真实部署 | `mf` | 自动使用 `cuda:0` 运行轻量路由器 | 首次下载 `routellm/mf_gpt4_augmented`；每次路由需 OpenAI Embeddings |
| 比较传统分类器 | `bert` | 可运行，但不是首选 GPU 路径 | 首次下载 BERT 权重和 tokenizer |
| 本地大模型路由分类 | `causal_llm` | 源码强制使用 CUDA | 需下载路由器权重和默认 Llama 3 8B；显存需求明显高于 `mf`，先检查可用显存后再尝试 |
| 离线偏好数据相似度路由 | `sw_ranking` | GPU 并非主要收益点 | 需要下载多个 Arena 偏好与 embedding 数据集 |

建议顺序是：先运行离线 `random` 冒烟检查，再在同意下载权重并配置密钥后运行 `mf`。不要先安装 `eval` extra 或下载 `causal_llm` 的 8B 基座模型；它们并不是完成首个服务验证所需的资源。

### 本地 Qwen3 弱模型 + DeepSeek Flash 强模型

服务端可为强、弱模型分别设置 LiteLLM 的模型名、地址和密钥。下面假设本地 Ollama 已启动并已拉取 `qwen3:1.7b`；`DEEPSEEK_FLASH_BASE_URL` 和 `DEEPSEEK_FLASH_API_KEY` 应替换为实际 DeepSeek Flash（或其兼容网关）提供的 OpenAI 兼容地址与密钥。密钥只保留在环境变量中：

```powershell
$env:EMBEDDING_API_KEY = '...' # 提供 Embeddings API 的 OpenAI 兼容服务
$env:EMBEDDING_BASE_URL = 'https://<your-embedding-compatible-endpoint>/v1'
$env:DEEPSEEK_FLASH_API_KEY = '...'
$env:DEEPSEEK_FLASH_BASE_URL = 'https://<your-deepseek-compatible-endpoint>/v1'

& 'D:\Anaconda\envs\pytorch\python.exe' -m routellm.openai_server `
  --routers mf `
  --config config.example.yaml `
  --strong-model gpt-4-1106-preview `
  --weak-model mixtral-8x7b-instruct-v0.1 `
  --strong-model-backend deepseek/deepseek-flash `
  --strong-base-url $env:DEEPSEEK_FLASH_BASE_URL `
  --strong-api-key $env:DEEPSEEK_FLASH_API_KEY `
  --embedding-base-url $env:EMBEDDING_BASE_URL `
  --embedding-api-key $env:EMBEDDING_API_KEY `
  --embedding-model text-embedding-3-small `
  --weak-model-backend ollama/qwen3:1.7b `
  --weak-base-url http://127.0.0.1:11434 `
  --weak-api-key ollama
```

`mf` 的预训练打分表不包含 `deepseek-flash` 或 `qwen3:1.7b`。因此 `--strong-model` 与 `--weak-model` 必须保留为其支持的评分锚点（上例为 GPT-4/Mixtral），而 `--*-model-backend` 才是实际调用的 DeepSeek/Qwen 模型。`--embedding-*` 专用于路由打分，必须指向支持 Embeddings API 的兼容服务；所用模型须返回 1536 维向量，才能匹配该 MF 权重的输入尺寸。这个分数只能作为迁移后的初始路由信号，须以你的真实请求集重新校准阈值并检查质量。

服务端会自动丢弃目标后端不支持的可选 OpenAI 参数。例如 Ollama 的 `qwen3:1.7b` 不支持 `presence_penalty`，但客户端即使按 OpenAI 默认值发送该字段，也不会使请求失败。

## 5. 同步到 Mercury-s-repo 的 `routeLLM/` 子目录

发布仓库的 `main` 分支将 RouteLLM 放在 `routeLLM/` 子目录。日常修改请在本仓库完成并提交，然后运行：

```powershell
.\scripts\sync_to_mercury_repo.ps1 -Message "Describe this RouteLLM update"
```

脚本默认使用 `D:\router\routeLLM-publish-snapshot` 作为发布仓库，只镜像当前 `HEAD` 的已提交文件到其 `routeLLM/`。它会拒绝源仓库或发布仓库存在未提交改动的情况；`/MIR` 的删除范围也严格限于发布仓库的 `routeLLM/`，不会触及其他项目。若发布仓库迁移到其他路径，可明确传入：

```powershell
.\scripts\sync_to_mercury_repo.ps1 `
  -PublishRepository 'D:\path\to\Mercury-s-repo' `
  -Message "Describe this RouteLLM update"
```

## 6. 阈值校准与评估

在已同意下载外部资源后，校准 50% 强模型调用比例：

```powershell
& 'D:\Anaconda\envs\pytorch\python.exe' -m routellm.calibrate_threshold `
  --task calibrate --routers mf --strong-model-pct 0.5 `
  --config config.example.yaml
```

评估示例（仓库内带有部分缓存结果；所选路由器仍可能触发权重/数据下载）：

```powershell
& 'D:\Anaconda\envs\pytorch\python.exe' -m routellm.evals.evaluate `
  --routers random mf --benchmark gsm8k --config config.example.yaml
```

## 需要确认后才下载的外部资源

| 使用场景 | 将下载的资源 |
| --- | --- |
| `mf` 路由 | Hugging Face：`routellm/mf_gpt4_augmented` 预训练权重 |
| `bert` 路由 | Hugging Face：`routellm/bert_gpt4_augmented` 预训练权重与 tokenizer |
| `causal_llm` 路由 | Hugging Face：`routellm/causal_llm_gpt4_augmented`，以及默认 `meta-llama/Meta-Llama-3-8B` 基座模型 |
| `sw_ranking` 路由 | Hugging Face 数据集：`lmsys/lmsys-arena-human-preference-55k`、`routellm/gpt4_judge_battles` 及两个 embedding 数据集 |
| `calibrate_threshold` | Hugging Face 数据集：`routellm/lmsys-arena-human-preference-55k-thresholds` |
| 重新生成阈值 | `lmsys/lmsys-arena-human-preference-55k`，并可能调用 OpenAI Embeddings |

当前没有下载上述数据集或预训练权重。


启动服务

$env:EMBEDDING_BASE_URL = 'https://api.zhizengzeng.com/vi'
$env:EMBEDDING_API_KEY = ''
$env:DEEPSEEK_FLASH_API_KEY = ''
$env:DEEPSEEK_FLASH_BASE_URL = 'https://api.zhizengzeng.com/vi'

& 'D:\Anaconda\envs\pytorch\python.exe' -m routellm.openai_server `
  --routers mf `
  --config config.example.yaml `
  --strong-model gpt-4-1106-preview `
  --weak-model mixtral-8x7b-instruct-v0.1 `
  --strong-model-backend deepseek/deepseek-flash `
  --strong-base-url $env:DEEPSEEK_FLASH_BASE_URL `
  --strong-api-key $env:DEEPSEEK_FLASH_API_KEY `
  --embedding-base-url $env:EMBEDDING_BASE_URL `
  --embedding-api-key $env:EMBEDDING_API_KEY `
  --embedding-model text-embedding-3-small `
  --weak-model-backend ollama/qwen3:1.7b `
  --weak-base-url http://127.0.0.1:11434 `
  --weak-api-key ollama


查看服务是否运行
Invoke-RestMethod http://127.0.0.1:6060/health


提问
$body = @{
  model = 'router-mf-0.5'
  messages = @(
    @{ role = 'user'; content = '5+3=？' }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:6060/v1/chat/completions' `
  -ContentType 'application/json' `
  -Body $body




查看回答
$response = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:6060/v1/chat/completions' `
  -ContentType 'application/json; charset=utf-8' `
  -Body $body

$response.model
$response.choices[0].message.content
