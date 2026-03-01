"""
ASR → 校对 → 翻译 → 双语 SRT 输出
"""

import json
import re
import os
import gc
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ──────────────────────────────────────────────
# 全局配置
# ──────────────────────────────────────────────
WHISPER_MODEL_SIZE   = "large-v3"
LANGUAGE             = "ja"
TEMP_DIR             = "temp_clips"

MAX_WORKERS          = 4
RETRY_MAX_ATTEMPTS   = 3
MAX_CHARS_PER_CHUNK  = 3000
PROOFREAD_BATCH_SIZE = 100

TRANSLATE_SYSTEM_PROMPT = "执行字幕翻译任务：将日语翻译为中文。"
TRANSLATE_USER_TEMPLATE = (
    "请逐行将日语翻译为中文。根据上下文语境纠正突兀之处，"
    "人名和自造词保留日语原文，必须严格保持并输出所有 ID。\n"
    "格式：[ID] 中文翻译\n\n{input_block}"
)


# ──────────────────────────────────────────────
# Token 统计
# ──────────────────────────────────────────────
class TokenCounter:
    def __init__(self): self.total_tokens = 0
    def add(self, text: str): self.total_tokens += int(len(text) * 1.3)

usage_stats = TokenCounter()


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def time_to_seconds(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    parts = str(t).strip().split(":")
    try:
        if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return int(parts[0])*60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0

def format_srt_time(seconds: float) -> str:
    ms       = int(abs(seconds % 1) * 1000)
    full_sec = int(abs(seconds))
    m, s = divmod(full_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ──────────────────────────────────────────────
# Step 1 · 人工干预模式 ASR（基于参考 SRT 时间轴）
# ──────────────────────────────────────────────
def parse_srt_blocks(path: str) -> List[dict]:
    """解析 SRT，返回 [{index, start, end, original_text}] 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = []
    time_pat = re.compile(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})')
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split("\n")
        idx = next((i for i, l in enumerate(lines) if time_pat.search(l)), -1)
        if idx == -1: continue
        m = time_pat.search(lines[idx])
        blocks.append({
            "index":         len(blocks) + 1,
            "start":         m.group(1),
            "end":           m.group(2),
            "original_text": "\n".join(lines[idx+1:]).strip(),
        })
    return blocks

def _srt_ts_to_sec(t: str) -> float:
    t = t.replace(",", ".")
    h, m, s = t.split(":")
    return float(h)*3600 + float(m)*60 + float(s)

def _split_audio_block(audio_path: str, block: dict, idx: int) -> Optional[str]:
    start = _srt_ts_to_sec(block["start"])
    end   = _srt_ts_to_sec(block["end"])
    dur   = end - start
    if dur < 0.3: return None
    out = os.path.join(TEMP_DIR, f"clip_{idx:04d}.wav")
    cmd = ["ffmpeg","-y","-ss",str(start),"-i",audio_path,"-t",str(dur),
           "-ac","1","-ar","16000","-c:a","pcm_s16le","-avoid_negative_ts","make_zero",out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.exists(out) and os.path.getsize(out) > 2048: return out
    except subprocess.CalledProcessError:
        pass
    return None

def _transcribe_clip(model: WhisperModel, clip_path: str) -> str:
    try:
        segs, _ = model.transcribe(clip_path, language=LANGUAGE, beam_size=5)
        return "".join(s.text for s in segs).strip()
    except Exception:
        return ""

def run_asr_from_srt(audio_path: str, srt_path: str) -> List[dict]:
    """
    以参考 SRT 时间轴为基准，逐段切片后交给 Whisper 识别。
    返回: [{"start": float, "end": float, "text": str}, ...]
    """
    print("🎧 [Step 1/4] 按 SRT 时间轴切片并 ASR 识别...")
    blocks = parse_srt_blocks(srt_path)
    print(f"   📋 共 {len(blocks)} 个时间段")

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"   🤖 加载模型 {WHISPER_MODEL_SIZE} ({device})")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)

    os.makedirs(TEMP_DIR, exist_ok=True)
    result = []

    for i, block in enumerate(tqdm(blocks, desc="   识别进度")):
        clip = _split_audio_block(audio_path, block, i)
        if clip:
            txt = _transcribe_clip(model, clip)
            try: os.remove(clip)
            except OSError: pass
        else:
            txt = ""
        result.append({
            "start": _srt_ts_to_sec(block["start"]),
            "end":   _srt_ts_to_sec(block["end"]),
            "text":  txt or block["original_text"],
        })

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"   ✅ ASR 完成，共 {len(result)} 条")
    return result


# ──────────────────────────────────────────────
# Step 2 · 弹幕清洗
# ──────────────────────────────────────────────
def run_danmaku_cleaning(file_name: str) -> List[dict]:
    print("🧹 [Step 2/4] 清洗参考弹幕...")
    buckets, clean_res = defaultdict(list), []
    KANJI = re.compile(r"[\u4E00-\u9FFF]")
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                data  = json.loads(line)
                items = data.get("replayChatItemAction", {}).get("actions", [])
                for a in items:
                    renderer = (a.get("addChatItemAction", {})
                                  .get("item", {})
                                  .get("liveChatTextMessageRenderer"))
                    if not renderer: continue
                    ts  = renderer.get("timestampText", {}).get("simpleText", "0:00")
                    msg = "".join(r.get("text","") for r in renderer.get("message",{}).get("runs",[]))
                    if len(msg) < 2 or not KANJI.search(msg): continue
                    key = (msg[0], len(msg)//2)
                    if any(fuzz.ratio(msg, old) >= 80 for old in buckets[key]): continue
                    buckets[key].append(msg)
                    clean_res.append({"_sec": time_to_seconds(ts), "text": msg})
    except Exception as e:
        print(f"   ⚠️ 弹幕清洗出错: {e}")
    return sorted(clean_res, key=lambda x: x["_sec"])


# ──────────────────────────────────────────────
# Step 3 · 智能校对
# ──────────────────────────────────────────────
def extract_mapping(content: str) -> Dict[str, str]:
    mapping = {}
    for line in content.splitlines():
        m = re.search(r"(S\d+)", line)
        if m:
            sid       = m.group(1)
            text_part = re.split(r"S\d+[\s\]:：]*", line, maxsplit=1)[-1]
            mapping[sid] = text_part.strip()
    return mapping

@retry(stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
       wait=wait_exponential(multiplier=2, min=4, max=15))
def call_llm_api(client: OpenAI, messages: List[dict], temp: float = 0.2) -> str:
    usage_stats.add(str(messages))
    resp    = client.chat.completions.create(model=client._model, messages=messages, temperature=temp)
    content = resp.choices[0].message.content
    usage_stats.add(content)
    return content

def run_smart_proofread(client: OpenAI, asr_data: List[dict],
                        danmu_data: List[dict], bg_params: dict) -> List[dict]:
    print("📡 [Step 3/4] 智能校对...")
    ctx     = f"Host: {bg_params['host_info']} | Title: {bg_params['stream_title']}"
    final   = []
    total   = len(asr_data)
    matched = 0

    for i in range(0, total, PROOFREAD_BATCH_SIZE):
        batch      = asr_data[i : i + PROOFREAD_BATCH_SIZE]
        w_s, w_e   = max(0, batch[0]["start"] - 15), batch[-1]["end"] + 15
        relevant   = [d for d in danmu_data if w_s <= d["_sec"] <= w_e]
        dm_in      = "\n".join(f"{d['_sec']:.1f}s: {d['text']}" for d in relevant)
        asr_in     = "\n".join(f"[S{i+idx+1:05d}] {s['text']}" for idx, s in enumerate(batch))
        messages   = [
            {"role": "system", "content": (
                f"执行日语 ASR 文本校对任务。校对背景：{ctx}。"
                "依据 [Host] 确定讲话人背景，依据 [Title] 确定话题起始背景。"
                "根据同期参考弹幕修正 ASR 中的错误。\n"
                "【约束】1.保留 [Sxxxxx] 标签格式。2.无需修改则原样返回。3.禁止输出解释。")},
            {"role": "user", "content": f"[参考弹幕]\n{dm_in}\n\n[待校对ASR]\n{asr_in}"},
        ]
        try:
            mapping = extract_mapping(call_llm_api(client, messages))
            for idx, s in enumerate(batch):
                tid      = f"S{i+idx+1:05d}"
                res_text = mapping.get(tid, s["text"])
                if res_text != s["text"]: matched += 1
                final.append({"start": s["start"], "end": s["end"], "ja": res_text})
        except Exception:
            for s in batch: final.append({"start": s["start"], "end": s["end"], "ja": s["text"]})

    print(f"   ✅ 校对完成，订正 {matched} 处")
    return final


# ──────────────────────────────────────────────
# Step 4 · 并发翻译
# ──────────────────────────────────────────────
def _translate_worker(client: OpenAI, chunk: List[Tuple[str, str]], idx: int, total: int) -> Dict[str, str]:
    input_block = "\n".join([f"[{sid}] {txt}" for sid, txt in chunk])
    messages = [
        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
        {"role": "user", "content": TRANSLATE_USER_TEMPLATE.format(input_block=input_block)},
    ]
    try:
        content = call_llm_api(client, messages)
        mapping = extract_mapping(content)

        # 检测解析失败的 ID，逐条单独重试
        missing = [(sid, txt) for sid, txt in chunk if sid not in mapping]
        if missing:
            print(f"   ⚠️ chunk {idx}/{total}: {len(missing)} 条解析失败，逐条重试...")
            for sid, txt in missing:
                try:
                    single = call_llm_api(client, [
                        {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"只输出中文译文，不要任何其他内容：{txt}"},
                    ])
                    mapping[sid] = single.strip()
                except Exception:
                    mapping[sid] = txt  # 保留日语原文

        return mapping

    except Exception as e:
        print(f"   ❌ chunk {idx}/{total} 整体失败: {e}，保留原文")
        return {sid: txt for sid, txt in chunk}  # 整体失败时保留原文，不返回空字典


def run_parallel_translation(client: OpenAI, segments: List[dict]) -> List[dict]:
    print(f"🚀 [Step 4/4] 启动并发翻译 (并发: {MAX_WORKERS})...")
    items = [(f"S{i+1:05d}", s["ja"]) for i, s in enumerate(segments)]

    chunks, cur_chunk, cur_len = [], [], 0
    for sid, txt in items:
        line = f"[{sid}] {txt}"
        if cur_chunk and cur_len + len(line) > MAX_CHARS_PER_CHUNK:
            chunks.append(cur_chunk)
            cur_chunk, cur_len = [], 0
        cur_chunk.append((sid, txt))
        cur_len += len(line)
    if cur_chunk:
        chunks.append(cur_chunk)

    all_zh: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_translate_worker, client, c, i + 1, len(chunks)): i
            for i, c in enumerate(chunks)
        }
        for f in as_completed(futures):
            all_zh.update(f.result())

    # 仍缺失的 ID 用日语原文填充
    failed = 0
    for i, s in enumerate(segments):
        sid = f"S{i+1:05d}"
        s["zh"] = all_zh.get(sid) or s["ja"]
        if not all_zh.get(sid):
            failed += 1

    if failed:
        print(f"   ⚠️ 最终仍有 {failed} 条未翻译，已用日语原文填充")
    else:
        print("   ✅ 全部翻译完成")
    return segments


# ──────────────────────────────────────────────
# SRT 写出
# ──────────────────────────────────────────────
def write_bilingual_srt(final_data: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(final_data, 1):
            f.write(f"{i}\n{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n"
                    f"{s['ja']}\n{s['zh']}\n\n")
    print(f"   💾 已输出：{output_path}")
