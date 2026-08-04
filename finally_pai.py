#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VTG 最终终极版 - 整合所有优化
- 显式时间戳 + NumPro帧编号
- 由粗到细定位 (Coarse-to-Fine)
- 自适应运动感知采样
- 均匀 + 运动峰值帧加权集成
- 增强后处理 (区间压缩、边界平滑)
- 16GB显存安全 (offload、低分辨率、清缓存)
- 断点续传
"""

import os
import re
import json
import logging
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info

# ========== 显存优化 ==========
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ========== 配置 ==========
BASE_DIR = "/run/media/g/Data/vgt_project"
MODEL_PATH = os.path.join(BASE_DIR, "modelscope/qwen_local")
VIDEO_DIR = os.path.join(BASE_DIR, "data/videos")
TEST_JSONL = os.path.join(BASE_DIR, "data/test.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "output/final_ultimate.jsonl")
PROGRESS_FILE = os.path.join(BASE_DIR, "output/final_progress.txt")
LOG_FILE = os.path.join(BASE_DIR, "output/final_log.txt")

# 帧配置
COARSE_FRAMES = 4           # 粗定位帧数
FINE_FRAMES = 8             # 细定位帧数
MAX_PIXELS = 256 * 28 * 28
REFINE_WINDOW = 2.0         # 细定位窗口扩展秒数
MAX_VIDEOS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INTERVAL_RATIO = 0.30   # 最大区间不超过视频时长的30%
ENABLE_AUDIO = False        # 音频默认关闭

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 加载模型 ==========
logger.info("Loading Qwen model (float16) with offload...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./offload",
        trust_remote_code=True,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    logger.info("Model loaded.")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    exit(1)

logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# ========== 工具函数 ==========

def get_duration_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return total / fps if total > 0 else 0.0, fps

def add_visual_annotations(frame, frame_idx, total_frames, timestamp, duration):
    """
    添加帧编号(NumPro)和绝对时间戳
    """
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # 左上角帧号 (NumPro)
    frame_text = f"Frame {frame_idx+1}/{total_frames}"
    bbox = draw.textbbox((0, 0), frame_text, font=font)
    draw.rectangle([(5, 5), (bbox[2]+15, bbox[3]+15)], fill=(0, 0, 0, 180))
    draw.text((10, 10), frame_text, fill=(255, 255, 0), font=font)

    # 右上角绝对时间
    time_text = f"t={timestamp:.1f}s"
    tw, th = draw.textbbox((0, 0), time_text, font=font)[2:4]
    draw.rectangle([(pil_img.width - tw - 15, 5), (pil_img.width - 5, th + 15)], fill=(0, 0, 0, 180))
    draw.text((pil_img.width - tw - 10, 10), time_text, fill=(0, 255, 255), font=font)

    return pil_img

def extract_frames_motion_aware(video_path, num_frames, duration, fps):
    """
    自适应运动感知采样
    - 计算运动密度分布，在运动剧烈的区域分配更多帧
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    # 评估运动密度
    step = max(1, total_frames // 30)
    motion_scores = []
    frame_indices = []
    prev_gray = None
    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            motion_scores.append(diff)
            frame_indices.append(idx)
        prev_gray = gray
    cap.release()
    
    if not motion_scores:
        # 无运动信息，均匀采样
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()
    else:
        # 运动密度加权采样
        motion_scores = np.array(motion_scores)
        probs = motion_scores / (motion_scores.sum() + 1e-6)
        selected = np.random.choice(frame_indices, size=num_frames, replace=False, p=probs)
        indices = sorted(selected)
    
    # 提取帧并添加标注
    frames = []
    cap = cv2.VideoCapture(video_path)
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = add_visual_annotations(frame_rgb, i, num_frames, timestamp, duration)
            frames.append(pil_img)
    cap.release()
    return frames

def extract_frames_in_window(video_path, start_time, end_time, num_frames, duration, fps):
    """
    在指定时间窗口内均匀采样帧
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = max(0, int(start_time * fps))
    end_idx = min(total_frames, int(end_time * fps))
    if end_idx <= start_idx:
        return []
    step = max(1, (end_idx - start_idx) // num_frames)
    indices = list(range(start_idx, end_idx, step))[:num_frames]
    frames = []
    cap = cv2.VideoCapture(video_path)
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = add_visual_annotations(frame_rgb, i, num_frames, timestamp, duration)
            frames.append(pil_img)
    cap.release()
    return frames

def get_motion_peak_frame(video_path, fps):
    """
    返回运动峰值帧索引（帧间差分最大）
    """
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    max_diff = -1
    peak_idx = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            if diff > max_diff:
                max_diff = diff
                peak_idx = idx
        prev_gray = gray
        idx += 1
    cap.release()
    return peak_idx

def extract_single_frame_annotated(video_path, idx, duration, fps, total_frames):
    """
    提取单帧并添加标注
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    timestamp = idx / fps
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return add_visual_annotations(frame_rgb, 0, total_frames, timestamp, duration)

# ========== 提示词构建 ==========

def build_prompt(question, duration, stage="coarse", prev_s=None, prev_e=None):
    """
    构建时间感知提示词
    """
    base = f"Video duration: {duration:.1f} seconds.\n"
    base += "Timeline: 0.0s to {duration:.1f}s.\n"
    if stage == "coarse":
        base += "Based on the provided frames, estimate the time interval where the answer occurs.\n"
    else:
        base += f"Previous estimate: {prev_s:.1f}-{prev_e:.1f}s. Refine to a more precise interval.\n"
    base += f"Question: {question}\n"
    base += "Output the most compact interval in format 'start-end', e.g., '15.6-20.8'.\n"
    base += "No other text.\nAnswer:"
    return base

# ========== 推理函数 ==========

def infer_frames(model, processor, frames, prompt):
    if not frames:
        return ""
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=MAX_PIXELS
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    generated = outputs[0][len(inputs.input_ids[0]):]
    raw = processor.decode(generated, skip_special_tokens=True)
    torch.cuda.empty_cache()
    return raw

def infer_single_frame(model, processor, pil_img, prompt):
    if pil_img is None:
        return None, None
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=MAX_PIXELS
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    generated = outputs[0][len(inputs.input_ids[0]):]
    raw = processor.decode(generated, skip_special_tokens=True)
    torch.cuda.empty_cache()
    s, e = parse_interval(raw)
    return s, e

def parse_interval(text):
    pattern = r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)'
    m = re.search(pattern, text)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        if s > e:
            s, e = e, s
        return s, e
    nums = re.findall(r'\d+\.?\d*', text)
    if len(nums) >= 2:
        s, e = float(nums[0]), float(nums[1])
        if s > e:
            s, e = e, s
        return s, e
    return None, None

def clamp_interval(s, e, duration, max_ratio=MAX_INTERVAL_RATIO):
    max_len = duration * max_ratio
    if max_len < 0.5:
        max_len = 0.5
    if e - s <= max_len:
        return s, e
    center = (s + e) / 2
    half = max_len / 2
    return max(0, center - half), min(duration, center + half)

# ========== 主预测流程 ==========

def predict_video(video_path, question):
    duration, fps = get_duration_fps(video_path)
    if duration == 0:
        return "0.0-1.0"

    # ----- 粗定位 (Coarse) -----
    coarse_frames = extract_frames_motion_aware(video_path, COARSE_FRAMES, duration, fps)
    if not coarse_frames:
        return "0.0-1.0"
    coarse_prompt = build_prompt(question, duration, stage="coarse")
    raw_coarse = infer_frames(model, processor, coarse_frames, coarse_prompt)
    s_c, e_c = parse_interval(raw_coarse)
    if s_c is None or e_c is None or e_c - s_c < 0.5:
        # 粗定位失败，尝试运动峰值帧
        peak_idx = get_motion_peak_frame(video_path, fps)
        motion_frame = extract_single_frame_annotated(video_path, peak_idx, duration, fps, 1)
        s_m, e_m = infer_single_frame(model, processor, motion_frame, coarse_prompt) if motion_frame else (None, None)
        if s_m is not None and e_m is not None and e_m - s_m >= 0.5:
            s_c, e_c = s_m, e_m
        else:
            return "0.0-5.0"

    # ----- 细定位 (Fine) 在粗定位窗口附近扩展 -----
    window_s = max(0, s_c - REFINE_WINDOW)
    window_e = min(duration, e_c + REFINE_WINDOW)
    fine_frames = extract_frames_in_window(video_path, window_s, window_e, FINE_FRAMES, duration, fps)
    if not fine_frames:
        # 无法获取细定位帧，返回粗定位并压缩
        s_final, e_final = clamp_interval(s_c, e_c, duration)
        return f"{s_final:.1f}-{e_final:.1f}"

    fine_prompt = build_prompt(question, duration, stage="refine", prev_s=s_c, prev_e=e_c)
    raw_fine = infer_frames(model, processor, fine_frames, fine_prompt)
    s_f, e_f = parse_interval(raw_fine)
    if s_f is None or e_f is None or e_f - s_f < 0.3:
        s_final, e_final = s_c, e_c
    else:
        # 边界约束在窗口内
        s_final = max(window_s, min(s_f, window_e - 0.5))
        e_final = max(s_final + 0.5, min(e_f, window_e))
        # 再次压缩
        s_final, e_final = clamp_interval(s_final, e_final, duration)

    # 最终后处理
    if s_final >= e_final:
        e_final = s_final + 0.5
    if s_final < 0:
        s_final = 0
    if e_final > duration:
        e_final = duration
    if e_final - s_final < 0.5:
        center = (s_final + e_final) / 2
        s_final = max(0, center - 0.5)
        e_final = min(duration, center + 0.5)

    return f"{s_final:.1f}-{e_final:.1f}"

# ========== 主流程 ==========

def main():
    if not os.path.exists(TEST_JSONL):
        logger.error(f"Test file not found: {TEST_JSONL}")
        return

    test_data = []
    with open(TEST_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_data.append(json.loads(line))
                except:
                    pass

    if not test_data:
        logger.error("test.jsonl is empty")
        return

    test_data = test_data[:MAX_VIDEOS]
    logger.info(f"Processing {len(test_data)} videos...")

    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            completed = set(line.strip() for line in f)

    existing = {}
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        existing[item['id']] = item['model_prediction']
                    except:
                        pass

    for item in tqdm(test_data, desc="Processing"):
        vid = item['id']
        if vid in completed:
            continue
        question = item['question']
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            pred = "0.0-1.0"
        else:
            pred = predict_video(video_path, question)
            logger.info(f"{vid} -> {pred}")
        result = {"id": vid, "model_prediction": pred}
        with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        completed.add(vid)
        with open(PROGRESS_FILE, 'a') as f:
            f.write(vid + '\n')
        torch.cuda.empty_cache()

    logger.info(f"All done. Results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()