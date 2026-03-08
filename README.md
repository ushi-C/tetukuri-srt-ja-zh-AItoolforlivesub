<div align="center">

#  tetukuri-srt-ja-zh-AItoolforlivesub

**直播字幕手搓轴-AI翻校工具**

*Faster-Whisper large-v3 × GPT 智能校对 × 弹幕辅助 × 日中双语 SRT 输出*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ushi-C/tetukuri-srt-ja-zh-AItoolforlivesub/blob/main/notebooks/pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

</div>

---

## ✨ 功能特性

| 步骤 | 模块 | 说明 |
|------|------|------|
| **Step 1** | ASR（人工干预模式） | 以参考 SRT 时间轴为基准，用 ffmpeg 逐段切片后交给 Whisper large-v3 重新识别 |
| **Step 2** | 弹幕清洗 | 解析 YouTube 直播弹幕 JSON，过滤非汉字内容与重复弹幕 |
| **Step 3** | 智能校对 | GPT 结合弹幕上下文纠正 ASR 错误|
| **Step 4** | 并发翻译 | 多线程调用 GPT 将日语逐行翻译为中文，输出双语 SRT |

---

## 🚀 快速开始

### 方式 A · Google Colab（推荐，无需本地环境）

1. 点击标题里面嵌的 **Open in Colab** 按钮
2. Runtime → Change runtime type → **T4 GPU**
3. 按顺序执行 Cell 1 → 5

### 方式 B · 本地运行

```bash
# 克隆项目
git clone https://github.com/ushi-C/tetukuri-srt-ja-zh-AItoolforlivesub.git
cd tetukuri-srt-ja-zh-AItoolforlivesub

# 安装依赖
pip install -r requirements.txt

# 系统需预装 ffmpeg
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

---

## 📁 项目结构

```
tetukuri-srt-ja-zh-AItoolforlivesub/
├── notebooks/
│   └── pipeline.ipynb      # Colab 一键运行入口
├── src/
│   └── pipeline.py         # ASR / 校对 / 翻译
├── docs/
│   └── example_output.srt  # 输出示例
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📋 使用说明

### 需要准备的文件

| 文件 | 必须 | 说明 |
|------|------|------|
| 音频文件 | ✅ | `.mp3 / .wav / .m4a / .flac`，直播录音或 YouTube 下载 |
| 参考 SRT | ✅ | 时间轴来源（你手打的轴**注意，要输出可识别的轴，轴不能为空，空内容可以全选填入一个数字，保证时间轴存在**）|
| 弹幕 JSON | ⬜ 可选 | YouTube 弹幕，用于校对 ASR 错误，格式为 `replayChatItemAction` |

### 输出

- `{音频文件名}_bilingual.srt` — 日中双语 SRT 字幕文件

```
1
00:00:01,234 --> 00:00:04,567
今日もみんなに会えて嬉しい！
今天也很高兴能见到大家！
```

---

## ⚙️ 配置参数

在 `src/pipeline.py` 顶部修改：

```python
WHISPER_MODEL_SIZE   = "large-v3"    # 模型大小 large-v3
LANGUAGE             = "ja"          # 源语言
MAX_WORKERS          = 4             # 翻译并发线程数
MAX_CHARS_PER_CHUNK  = 3000          # 单次翻译最大字符数
PROOFREAD_BATCH_SIZE = 100           # 单批校对字幕条数
```

在 Colab Cell 3 中修改 API 配置：

```python
OPENAI_BASE_URL = "https://api.openai.com/v1"    # 官方 API
MODEL_NAME      = "gpt-5.2"                       # 推荐模型
```

---

## 🔑 API Key 获取

- **OpenAI 官方**：https://platform.openai.com/api-keys
- **第三方代理**（国内可用）：自行搜索可靠的 OpenAI 代理服务

> ⚠️ API Key 仅在运行时通过 `getpass` 输入，**不会**写入代码或提交到 Git。

---

## 📦 依赖说明

| 包 | 版本 | 用途 |
|----|------|------|
| `faster-whisper` | ≥1.0.0 | Whisper 推理（CTranslate2 加速）|
| `openai` | ≥1.0.0 | GPT 校对 & 翻译 API |
| `rapidfuzz` | ≥3.0.0 | 弹幕相似度去重 |
| `tenacity` | ≥8.0.0 | API 调用失败自动重试 |
| `tqdm` | ≥4.65.0 | ASR 进度条 |
| `ffmpeg` | 系统级 | 音频切片 |

> 首次运行时，`faster-whisper` 会自动从 HuggingFace 下载 `large-v3` 模型（约 **3 GB**）。


---

## 🤝 Contributing

欢迎 PR 和 Issue！  


---

## 📄 License

[MIT](LICENSE) © 2026 ushi-C
