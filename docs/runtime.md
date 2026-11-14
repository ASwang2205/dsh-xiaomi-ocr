# 运行时准备

插件本身只有几十 KB，真正干活的是一个 Python 进程。它需要：

| 依赖 | 说明 |
| --- | --- |
| Python 3.13 | 3.11+ 也可，但权重是用 3.13 验证的 |
| torch | 与显卡匹配的构建（本机为 `2.12.0+rocm7.14`） |
| torchvision | 供 transformers 的图像解码 |
| transformers ≥ 5 | 提供 `Qwen3_5` / `Qwen3VLProcessor` |
| Pillow | 图片读取与缩放 |
| pypdfium2 | 仅 PDF 需要，插件会自动装到私有 `pylibs` |

## 方案 A：复用已有环境（推荐，零重复下载）

如果机器上已经有 ComfyUI（桌面版或整合包），它的 venv 通常就带 ROCm/CUDA 版 torch。
把安装根目录填进设置页「运行时 → 额外搜索目录」，插件会按 ComfyUI 的目录布局找解释器：

```
<root>\
└─ <install>\
   └─ ComfyUI\
      └─ .venv\Scripts\python.exe
```

例如 root 填 `D:\Comfy-Desktop\ComfyUI-Installs`，就能匹配
`D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\.venv\Scripts\python.exe`。

验证：

```powershell
& "<venv>\Scripts\python.exe" -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

只要 `torch.cuda.is_available()` 为 `True`（ROCm 也走这个开关），就可以直接用。
插件不会往这个环境里装任何东西 —— PDF 依赖走 `--target` 装到
`%USERPROFILE%\.dsh\xiaomi-ocr\pylibs`。

## 方案 B：建独立 venv

```powershell
pwsh scripts/setup-runtime.ps1 -Python "C:\Python313\python.exe"
```

脚本会在 `%USERPROFILE%\.dsh\xiaomi-ocr\venv` 建环境，并按索引装 torch/torchvision，
再装 `transformers pillow pypdfium2`。NVIDIA 用户把 `-TorchIndex` 换成
`https://download.pytorch.org/whl/cu124`；纯 CPU 用 `https://download.pytorch.org/whl/cpu`。

装完在设置页把「Python 路径」指向 `%USERPROFILE%\.dsh\xiaomi-ocr\venv\Scripts\python.exe`
（或者留空，插件会自动优先用这个路径）。

## 模型权重

```powershell
python -c "from huggingface_hub import snapshot_download as s; s('xiaomi-research/Xiaomi-OCR-0', local_dir=r'%USERPROFILE%\.dsh\xiaomi-ocr\models\Xiaomi-OCR-0', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','processor_config.json','chat_template.jinja'])"
```

约 1.7 GB。必须的文件只有 6 个；`example_pics/` 是上游样例图，可选。

## 精度与速度

服务启动时按设备自动选精度：

| 设备 | dtype | 备注 |
| --- | --- | --- |
| ROCm（`torch.version.hip` 存在） | `float16` | 实测比 bf16 快一倍（26–30 vs 12.9 tok/s） |
| CUDA | `bfloat16` | NVIDIA 上 bf16 通常更快更稳 |
| CPU | `float32` | 能跑，但慢到只适合验证链路 |

可用 `--dtype` 显式覆盖（服务脚本参数）。

## 显存与像素预算

显存主要由**视觉塔**决定，而且随 patch 数平方增长，跟语言模型那 0.8B 关系不大：

| 输入 | 视觉 token | 峰值显存 | 备注 |
| --- | --- | --- | --- |
| 932×1644 收据（1.5 MP） | ~1500 | 2–3 GB | 宽松 |
| 1428×2520 A4 扫描（3.6 MP） | 3576 | **22.4 GB** | 16 GB 卡必 OOM |

所以：

- 默认 `--max-pixels 2200000`（A4@150 dpi 正好 2.2 MP）。再大先降采样。
- 万一仍 OOM，服务把像素预算减半重试（最多两次），结果里带 `max_pixels` / `retries`。
- 需要细腻小字时，优先用 `region` 只裁那一块（裁剪会按 `--min-side` 放大），而不是整页拉高 DPI。

## 端口与生命周期

- 默认 `127.0.0.1:18765`，`GET /health` 可探活。
- 插件在首次 OCR 时自动拉起，`--idle-exit 1800` 表示空闲 30 分钟后自行退出。
- 设置页「停止」调用 `POST /shutdown`；失败时回退到 `taskkill /PID <pid> /T /F`。
- 手动调试：`python runtime/server.py --model <dir> --port 18765 --log-file <path>`。

## 换成别的后端

服务只实现 OpenAI 兼容协议，所以设置页把后端切成「外部端点」后，可以指向：

- **Ollama**：`http://127.0.0.1:11434/v1` + 模型名（例如 `longwayxu/xiaomi-ocr-0:bf16`）
- **llama.cpp server**：`http://127.0.0.1:8080/v1` + GGUF（`Xiaomi-OCR-0.mmproj-bf16.gguf`）
- **SGLang / vLLM**：各自的 OpenAI 兼容入口
- 任何云端多模态端点（此时图片会离开本机 —— 与插件主打的离线目标相反）
