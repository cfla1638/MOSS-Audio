"""
MOSS-Audio 监督微调脚本，支持 LoRA 和全参数训练。

阅读路线：JSONL 对话 -> Dataset 提取 Mel、拼接 token 和标签 ->
默认 collator 增加批维 -> MossAudioModel.forward -> 答案部分的下一 token 损失。

维度名称：audio_samples 为波形采样点数，mel_bins 为 Mel 频率通道数（128），
audio_frames 为有效 Mel 帧数，audio_tokens 为三次卷积下采样后的长度；
sequence_length 为截断/补齐前的对话 token 总数，max_sequence_length 对应 max_len，
batch_size 为训练样本数。每个训练样本仅使用解析到的第一段音频。
retained_audio_frames 为截断后保留的 Mel 帧数；未发生截断时等于 audio_frames。

用法：
    # LoRA
    accelerate launch finetune.py \
        --model_dir ./weights/moss-audio \
        --data_path train.jsonl \
        --output_dir ./output \
        --use_lora

    # 全参数训练
    accelerate launch finetune.py \
        --model_dir ./weights/moss-audio \
        --data_path train.jsonl \
        --output_dir ./output

数据格式（JSONL，每行一个样本）：
    {"conversation": [
        {"role": "user", "message_type": "audio", "content": "/path/to/audio.wav"},
        {"role": "user", "message_type": "text",  "content": "Transcribe the audio."},
        {"role": "assistant", "message_type": "text", "content": "Hello world."}
    ]}
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
import transformers
from transformers import Trainer
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

from src.configuration_moss_audio import MossAudioConfig
from src.modeling_moss_audio import MossAudioModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen3 特殊 token ID：必须与所加载模型的 tokenizer 对应。
AUDIO_TOKEN_ID = 151654       # <|AUDIO|>
AUDIO_START_ID = 151669       # <|audio_bos|>
AUDIO_END_ID = 151670         # <|audio_eos|>
IM_START_ID = 151644          # <|im_start|>
IM_END_ID = 151645            # <|im_end|>

# 预先编码的对话边界与系统提示词，省去逐样本重复编码；这些是 Python ID 列表。
SYSTEM_IDS = [IM_START_ID, 8948, 198, 2610, 525, 264, 10950, 17847, 13, IM_END_ID, 198]
USER_AUDIO_START_IDS = [IM_START_ID, 872, 198, AUDIO_START_ID]
AUDIO_END_NL_IDS = [AUDIO_END_ID, 198]
TURN_BOUNDARY_IDS = [IM_END_ID, 198, IM_START_ID, 77091, 198]  # <|im_end|>\n<|im_start|>assistant\n


# ---------------------------------------------------------------------------
# 参数：HfArgumentParser 将三个 dataclass 转为命令行参数。
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    # 指定 checkpoint 和注意力后端；真正的模型结构来自 checkpoint 配置。
    model_dir: str = field(metadata={"help": "Path to MOSS-Audio model directory."})
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArguments:
    # max_len 限制整个多模态对话的 token 数，不是音频采样点数或 Mel 帧数。
    data_path: str = field(metadata={"help": "Path to training JSONL file."})
    eval_data_path: Optional[str] = field(default=None)
    max_len: int = field(default=8192)
    prompt_default: str = field(default="")


@dataclass
class FinetuneArguments(transformers.TrainingArguments):
    # 继承批大小、学习率等通用参数；LoRA rank 是低秩更新的中间维度。
    # LoRA 不改变被适配层的输入/输出张量形状。
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=2)
    lora_on_audio_encoder: bool = field(default=False)
    optim: str = field(default="adamw_torch_fused")


# ---------------------------------------------------------------------------
# Mel 提取：与推理端使用相同类型的 Whisper 特征提取内核。
# ---------------------------------------------------------------------------

_whisper_fe: Optional[WhisperFeatureExtractor] = None


def extract_mel(audio_path: str, sr: int = 16000) -> torch.Tensor:
    # 返回 [mel_bins, audio_frames]、bfloat16；缓存 extractor 以避免逐样本重新创建。
    global _whisper_fe
    if _whisper_fe is None:
        _whisper_fe = WhisperFeatureExtractor(
            feature_size=128, sampling_rate=sr, hop_length=160, n_fft=400,
        )
    wav, _ = librosa.load(audio_path, sr=sr)
    # librosa 默认转单声道并重采样：wav 为 [audio_samples]。
    # 增加批维 [1, audio_samples] -> 特征 [1, mel_bins, audio_frames]。
    feats = _whisper_fe._np_extract_fbank_features(wav[None, ...], device="cpu")
    # feats[0] 去掉批维；这里直接调用内核，不做公开接口的固定时长填充/截断。
    return torch.from_numpy(feats[0]).to(torch.bfloat16)


def _compute_audio_tokens(mel_len: int) -> int:
    """三次 stride=2 卷积：每次向上取整减半，最终得到 ceil(audio_frames / 8)。"""
    # 默认 16kHz、hop=160 对应约 100 帧/秒，下采样后约 12.5 个音频 token/秒。
    for _ in range(3):
        mel_len = (mel_len - 1) // 2 + 1
    return mel_len


# ---------------------------------------------------------------------------
# Dataset：单样本构造。这里返回的张量尚未添加训练批维。
# ---------------------------------------------------------------------------

class MossAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], tokenizer, max_len: int, prompt_default: str = ""):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_default = prompt_default

    def __len__(self):
        return len(self.data)

    def _parse(self, conversation):
        # 只取第一条音频消息；所有 user 文本和 assistant 文本分别合并。
        # 因而这里得到的是一组“音频 + 问题 + 答案”，并未保留多轮交替结构。
        audio_path = prompt = answer = None
        prompt_parts, answer_parts = [], []
        for msg in conversation:
            mt = msg.get("message_type")
            if mt == "audio" and audio_path is None:
                audio_path = msg["content"]
            elif mt == "text":
                if msg["role"] == "user":
                    prompt_parts.append(msg.get("content", ""))
                elif msg["role"] == "assistant":
                    answer_parts.append(msg.get("content", ""))
        prompt = "\n".join(prompt_parts).strip() or self.prompt_default
        answer = "\n".join(answer_parts).strip()
        return audio_path, prompt, answer

    def __getitem__(self, idx):
        obj = self.data[idx]
        audio_path, prompt, answer = self._parse(obj["conversation"])
        if audio_path is None:
            raise ValueError(f"No audio in sample {idx}")

        mel = extract_mel(audio_path)
        # mel: [mel_bins, audio_frames]；n_tokens 是整数长度，不是张量。
        n_tokens = _compute_audio_tokens(mel.shape[-1])

        # 构造 Qwen3 对话序列，音频占位符随后在模型中被连续特征替换。
        # 当前脚本不插入 processing_moss_audio.py 推理侧可选的时间数字标记。
        audio_ids = [AUDIO_TOKEN_ID] * n_tokens
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = (
            SYSTEM_IDS
            + USER_AUDIO_START_IDS
            + audio_ids
            + AUDIO_END_NL_IDS
            + prompt_ids
            + TURN_BOUNDARY_IDS
            + answer_ids
            + [IM_END_ID]
        )
        labels = (
            # labels 与 input_ids 等长；系统、用户、音频和答案前缀均标 -100。
            # 只有答案及其结束符参与监督，标签的错位在模型 forward 内完成。
            [-100] * (len(SYSTEM_IDS) + len(USER_AUDIO_START_IDS) + len(audio_ids)
                      + len(AUDIO_END_NL_IDS) + len(prompt_ids) + len(TURN_BOUNDARY_IDS))
            + answer_ids + [IM_END_ID]
        )
        audio_mask = (
            # bool 列表，同样长为 sequence_length；只有音频占位符位置为 True。
            # 音频起止符仍使用普通词嵌入，不参与连续特征替换。
            [False] * len(SYSTEM_IDS)
            + [False] * len(USER_AUDIO_START_IDS)
            + [tid == AUDIO_TOKEN_ID for tid in audio_ids]
            + [False] * (len(AUDIO_END_NL_IDS) + len(prompt_ids) + len(TURN_BOUNDARY_IDS)
                         + len(answer_ids) + 1)
        )

        # 三个列表同步保留前 max_sequence_length 个位置，维持 token/标签/mask 对齐。
        # 当前策略可能截掉全部答案；没有额外保证截断后仍存在有效监督标签。
        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]
        audio_mask = audio_mask[: self.max_len]

        # 如果截断进入音频区域，也要裁剪 Mel，保证编码器输出数量等于占位符数量。
        actual_audio_tokens = sum(1 for m in audio_mask if m)
        if actual_audio_tokens < n_tokens:
            keep_frames = actual_audio_tokens * 8  # 每个保留 token 对应最多 8 帧输入。
            # [mel_bins, audio_frames] -> [mel_bins, retained_audio_frames]。
            mel = mel[:, :keep_frames]

        seq_len = len(input_ids)
        pad_len = self.max_len - seq_len

        # 补齐后 input_ids/labels/attention_mask/audio_input_mask 都为 [max_sequence_length]。
        # input_ids/labels/attention_mask 是 int64；audio_input_mask 是 bool。
        # attention_mask 的 1 表示真实序列、0 表示 padding；与是否监督该 token 无关。
        # audio_data_seqlens 是零维 int64 标量 []，数值为裁剪后 Mel 的有效帧数。
        return {
            "input_ids": torch.tensor(input_ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
            "audio_data": mel,                                    # [mel_bins, retained_audio_frames]
            "audio_data_seqlens": torch.tensor(mel.shape[-1], dtype=torch.long),
            "audio_input_mask": torch.tensor(audio_mask + [False] * pad_len, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# 训练入口：解析参数、加载模型、可选 LoRA、构建数据集并交给 Trainer。
# ---------------------------------------------------------------------------

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, FinetuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---- 模型 ----
    # 在顶层和语言模型配置中设置注意力后端；这里没有直接改 audio_config。
    config = MossAudioConfig.from_pretrained(model_args.model_dir)
    for cfg in [config, getattr(config, "language_config", None)]:
        if cfg is not None:
            cfg._attn_implementation = model_args.attn_implementation

    model = MossAudioModel.from_pretrained(
        # 模型加载 dtype 在此固定为 bfloat16；训练精度选项由 TrainingArguments 另行管理。
        model_args.model_dir, config=config, dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_dir, trust_remote_code=True)

    # ---- LoRA ----
    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        target = (
            # 默认匹配语言模型各层注意力和 MLP 的投影矩阵。
            r"language_model\.layers\.\d+\.(self_attn|mlp)\."
            r"(q_proj|k_proj|v_proj|o_proj|up_proj|gate_proj|down_proj)"
        )
        if training_args.lora_on_audio_encoder:
            # 可选扩展到音频编码器的 q/k/v；不包含卷积、模态适配器或 lm_head。
            target += r"|audio_encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"

        model = get_peft_model(
            # PEFT 注入低秩增量并默认冻结基座权重；未启用 LoRA 时保留全参训练路径。
            model,
            LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=target,
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    # ---- 数据 ----
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    train_data = load_jsonl(data_args.data_path)
    train_dataset = MossAudioDataset(train_data, tokenizer, data_args.max_len, data_args.prompt_default)
    logger.info("Train samples: %d", len(train_dataset))

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = MossAudioDataset(
            load_jsonl(data_args.eval_data_path), tokenizer, data_args.max_len, data_args.prompt_default,
        )
        logger.info("Eval samples: %d", len(eval_dataset))

    # ---- 训练 ----
    # 未传自定义 collator，默认按字段堆叠单样本张量，增加 batch_size 轴：
    # input_ids/labels/attention_mask/audio_input_mask -> [batch_size, max_sequence_length]；
    # audio_data -> [batch_size, mel_bins, retained_audio_frames]；
    # audio_data_seqlens -> [batch_size]。
    # Mel 未跨样本补齐，因此不同长度不能直接堆叠，文档要求默认批大小为 1。
    # 本训练数据每样本一段音频，因此这里 audio_count = batch_size；
    # 不要把这个对应关系推广到推理端一个提示词含多段音频的情况。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    # Trainer 调用模型 forward 获取标量 loss 并负责反向传播、梯度累积和优化更新。
    # 保存训练状态及模型；LoRA 模式通常保存适配器权重，不是合并后的完整基座。
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
