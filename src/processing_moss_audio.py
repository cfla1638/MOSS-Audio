# 阅读路线：波形 -> Mel 特征 -> 音频占位 token -> 对话序列 -> 模型输入。
# 本文件的形状名称：
#   audio_samples：单段波形的采样点数；mel_bins：Mel 频率通道数（默认 128）。
#   audio_frames：单段 Mel 的有效帧数；padded_audio_frames：多段中最大的帧数。
#   audio_count：一个提示词中的音频段数，不是文本 batch_size。
#   audio_tokens：单段音频经过三次下采样后的长度。
#   sequence_length：文本、音频占位符、时间标记及对话边界的总 token 数。
# 当前 processor 每次构造一个文本样本，input_ids 的 batch_size 固定为 1。
import importlib.util
import os
import re
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, BatchEncoding


@dataclass
class MelConfig:
    # 默认每 160 个采样点取一帧：16000 / 160 = 100 帧/秒；不是 100 个音频 token/秒。
    mel_sr: int = 16000
    mel_dim: int = 128
    mel_n_fft: int = 400
    mel_hop_length: int = 160
    mel_dtype: torch.dtype = torch.bfloat16
    use_whisper_feature_extractor: bool = True


def load_chat_template(template_path: str, mossflux_path: str = None) -> List:
    # 加载 Python 模板模块中的分段定义，供后面逐段拼接 token 使用。
    if mossflux_path is None:
        template_dir = os.path.dirname(os.path.abspath(template_path))
        current = template_dir
        while current and os.path.basename(current) != "mossLite":
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if os.path.basename(current) == "mossLite":
            mossflux_path = os.path.join(current, "mossflux")

    if mossflux_path and mossflux_path not in sys.path:
        sys.path.insert(0, mossflux_path)

    spec = importlib.util.spec_from_file_location("chat_template_module", template_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chat_template_module"] = module
    spec.loader.exec_module(module)
    return module.chat_template


class MossAudioProcessor:
    # 正则匹配的是原始提示词中的整段音频占位区域，随后按实际音频长度展开。
    _AUDIO_SPAN_RE = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")
    _auto_class = None

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__
        cls._auto_class = auto_class

    def __init__(
        self,
        tokenizer,
        *,
        mel_config: Optional[MelConfig] = None,
        template_path: Optional[str] = None,
        enable_time_marker: bool = True,
        audio_token_id: int = 151654,
        audio_start_id: int = 151669,
        audio_end_id: int = 151670,
    ):
        self._base_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.audio_token_id = int(audio_token_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.chat_template = (
            None if template_path is None else load_chat_template(template_path)
        )
        self.custom_texts = {}
        self.enable_time_marker = bool(enable_time_marker)
        self.config = mel_config or MelConfig()
        self._whisper_feature_extractor = None

        alias_map = {
            "<|AUDIO|>": self.audio_token_id,
            "<|audio_bos|>": self.audio_start_id,
            "<|audio_eos|>": self.audio_end_id,
        }
        orig_convert_tokens_to_ids = self.tokenizer.convert_tokens_to_ids

        # 只覆盖这三个音频符号的 ID 映射；普通文本仍调用原 tokenizer。
        def _patched_convert_tokens_to_ids(tokenizer_self, tokens):
            if isinstance(tokens, (list, tuple)):
                converted = [
                    _patched_convert_tokens_to_ids(tokenizer_self, token)
                    for token in tokens
                ]
                return converted if isinstance(tokens, list) else tuple(converted)
            if isinstance(tokens, str) and tokens in alias_map:
                return alias_map[tokens]
            return orig_convert_tokens_to_ids(tokens)

        self.tokenizer.convert_tokens_to_ids = types.MethodType(
            _patched_convert_tokens_to_ids, self.tokenizer
        )

        self._digit_token_ids = {
            "0": 15,
            "1": 16,
            "2": 17,
            "3": 18,
            "4": 19,
            "5": 20,
            "6": 21,
            "7": 22,
            "8": 23,
            "9": 24,
        }
        self.audio_tokens_per_second = 12.5
        # 100 帧/秒经三次 stride=2 下采样约为 12.5 token/秒，每 25 个插入秒数。
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        self.model_input_names = [
            "input_ids",
            "attention_mask",
            "audio_data",
            "audio_data_seqlens",
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 此处加载 tokenizer；Mel 参数使用传入配置或 MelConfig 默认值。
        # 注意本工厂方法默认关闭时间标记，和 __init__ 默认值不同。
        tokenizer_kwargs = {}
        for key in ["cache_dir", "revision", "token", "local_files_only"]:
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=False,
            **tokenizer_kwargs,
        )

        mel_config = kwargs.pop("mel_config", None)
        template_path = kwargs.pop("template_path", None)
        enable_time_marker = kwargs.pop("enable_time_marker", False)
        audio_token_id = kwargs.pop("audio_token_id", 151654)
        audio_start_id = kwargs.pop("audio_start_id", 151669)
        audio_end_id = kwargs.pop("audio_end_id", 151670)

        return cls(
            tokenizer,
            mel_config=mel_config,
            template_path=template_path,
            enable_time_marker=enable_time_marker,
            audio_token_id=audio_token_id,
            audio_start_id=audio_start_id,
            audio_end_id=audio_end_id,
        )

    def load_template(self, template_path: str):
        self.chat_template = load_chat_template(template_path)
        return self

    def set_custom_text(self, key: str, text: str):
        self.custom_texts[key] = text
        return self

    def clear_custom_text(self, key: Optional[str] = None):
        if key is None:
            self.custom_texts.clear()
        else:
            self.custom_texts.pop(key, None)
        return self

    def _template_requires_audio(self) -> bool:
        if self.chat_template is None:
            return False
        for segment in self.chat_template:
            if segment.type in {"audio_contiguous", "audio_token"}:
                return True
        return False

    @staticmethod
    def _conv3_downsample_len(raw_mel_len: int) -> int:
        # kernel=3、stride=2、padding=1 的输出长度为 ceil(输入长度 / 2)。
        # 连用三次等价于 ceil(audio_frames / 8)，尾部不足 8 帧也对应一个 token。
        def conv_out_len(length: int) -> int:
            return (length - 1) // 2 + 1

        length1 = conv_out_len(int(raw_mel_len))
        length2 = conv_out_len(length1)
        length3 = conv_out_len(length2)
        return int(length3)

    def _get_whisper_feature_extractor(self):
        # 复用 Whisper 的声学特征提取工具；这里不加载 Whisper 模型权重。
        if self._whisper_feature_extractor is not None:
            return self._whisper_feature_extractor

        from transformers.models.whisper.feature_extraction_whisper import (
            WhisperFeatureExtractor,
        )

        self._whisper_feature_extractor = WhisperFeatureExtractor(
            feature_size=int(self.config.mel_dim),
            sampling_rate=int(self.config.mel_sr),
            hop_length=int(self.config.mel_hop_length),
            n_fft=int(self.config.mel_n_fft),
        )
        return self._whisper_feature_extractor

    def _extract_mel(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # 输入约定：已经重采样的单声道 [audio_samples]（或 [1, audio_samples]）。
        # 这里不做混声道或重采样；infer.py 的 load_audio 负责这些步骤。
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio)
        else:
            wav = audio
        wav = wav.to(dtype=torch.float32)
        if wav.dim() == 1:
            # [audio_samples] -> [1, audio_samples]，添加单段波形的批维。
            wav = wav.unsqueeze(0)

        # 当前仅实现此分支；配置为 False 时 mel 不会被赋值。
        if bool(getattr(self.config, "use_whisper_feature_extractor", False)):
            fe = self._get_whisper_feature_extractor()
            wav_np = wav.detach().to("cpu", torch.float32).contiguous().numpy()
            if wav_np.ndim == 2:
                # [1, audio_samples] -> [audio_samples]；传多行时实际只取第一行。
                wav_np = wav_np[0]
            # [1, audio_samples] -> [1, mel_bins, audio_frames]。
            # 直接调用特征提取内核，不走公开接口的固定时长填充/截断流程。
            feats = fe._np_extract_fbank_features(wav_np[None, ...], device="cpu")
            # 去掉单段批维，保留频率轴和时间轴。
            mel = torch.from_numpy(feats[0])

        # [mel_bins, audio_frames]，默认转为 bfloat16；转换 dtype 不改变形状。
        return mel.to(dtype=self.config.mel_dtype)

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        # 秒数使用普通数字 token：例如 12 秒是两个 token，不是一个专用时间 token。
        return [self._digit_token_ids[digit] for digit in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        # 输出仍是一维 Python ID 列表；时间标记增加列表长度，音频占位符数量不变。
        # 例如 50 个音频 token：25 个占位符 + 数字 2 + 25 个占位符 + 数字 4。
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)

        token_ids: List[int] = []
        audio_tokens_consumed = 0
        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                token_ids.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            token_ids.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            token_ids.extend([self.audio_token_id] * remaining)
        return token_ids

    def _build_audio_placeholder_ids(self, num_audio_tokens: int) -> List[int]:
        if self.enable_time_marker:
            return self._build_audio_tokens_with_time_markers(num_audio_tokens)
        return [self.audio_token_id] * num_audio_tokens

    def _build_input_from_template(
        self, num_audio_tokens: int, include_answer: bool = False
    ) -> List[int]:
        # 按模板顺序连接常量、音频和自定义文本；推理时在答案段之前停止。
        if self.chat_template is None:
            raise ValueError("Chat template not loaded.")

        input_ids: List[int] = []
        for segment in self.chat_template:
            seg_type = segment.type
            if seg_type == "constant_text_token":
                input_ids.extend(segment.text_ids.tolist())
            elif seg_type in {"audio_contiguous", "audio_token"}:
                input_ids.extend(self._build_audio_placeholder_ids(num_audio_tokens))
            elif seg_type == "text_token":
                text_token_key = segment.text_token_key
                if "answer" in text_token_key.lower() and not include_answer:
                    break
                if text_token_key not in self.custom_texts:
                    break
                text_ids = self._base_tokenizer.encode(
                    self.custom_texts[text_token_key], add_special_tokens=False
                )
                input_ids.extend(text_ids)

        return input_ids

    def _build_default_prompt(self, text: str, has_audio: bool) -> str:
        # 构造 Qwen3 对话前缀，以 assistant 起始标记结尾，让模型继续生成答案。
        # 默认有音频模板只放一个音频区域；多段输入需要调用方提供对应的多个区域。
        if has_audio:
            return (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_input_from_prompt(self, prompt: str, token_lens: List[int]) -> List[int]:
        # token_lens 长度为 audio_count，区域顺序必须与音频输入顺序一致。
        # 每个区域展开为：起始 ID + audio_tokens 个占位符（可夹时间标记）+ 结束 ID。
        spans = list(self._AUDIO_SPAN_RE.finditer(prompt))
        if len(spans) != len(token_lens):
            raise ValueError(
                f"Audio placeholder count mismatch: found {len(spans)} spans in text, "
                f"but got {len(token_lens)} audio inputs."
            )

        input_ids: List[int] = []
        cursor = 0
        for index, match in enumerate(spans):
            prefix = prompt[cursor : match.start()]
            if prefix:
                input_ids.extend(
                    self._base_tokenizer.encode(prefix, add_special_tokens=False)
                )

            input_ids.append(self.audio_start_id)
            input_ids.extend(self._build_audio_placeholder_ids(int(token_lens[index])))
            input_ids.append(self.audio_end_id)
            cursor = match.end()

        suffix = prompt[cursor:]
        if suffix:
            input_ids.extend(
                self._base_tokenizer.encode(suffix, add_special_tokens=False)
            )
        return input_ids

    def __call__(
        self,
        *,
        text: Union[str, Sequence[str], None] = None,
        audios: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        audio: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        # text 的列表只允许一个元素；audios 列表可以表示该文本样本内的多段音频。
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"Expected text batch size 1, got {len(text)}")
            prompt_text = text[0]
        else:
            prompt_text = text

        audio_list = audios if audios is not None else audio
        audio_list = [] if audio_list is None else list(audio_list)

        mels: List[torch.Tensor] = []
        raw_lengths: List[int] = []
        token_lens: List[int] = []
        for one_audio in audio_list:
            # 单段 [audio_samples] -> [mel_bins, audio_frames]。
            mel = self._extract_mel(one_audio)
            raw_len = int(mel.shape[-1])
            mels.append(mel)
            raw_lengths.append(raw_len)
            token_lens.append(self._conv3_downsample_len(raw_len))

        if mels:
            max_length = max(raw_lengths)
            # [audio_count, mel_bins, padded_audio_frames]，沿时间轴右侧补零。
            audio_batch = torch.zeros(
                (len(mels), self.config.mel_dim, max_length),
                dtype=self.config.mel_dtype,
            )
            for index, mel in enumerate(mels):
                audio_batch[index, :, : mel.shape[-1]] = mel
            seqlens_tensor = torch.tensor(raw_lengths, dtype=torch.long)
            # [audio_count]，int64，记录补零前长度，编码器据此移除无效帧。
        else:
            audio_batch = None
            seqlens_tensor = None

        if prompt_text is not None:
            if self._AUDIO_SPAN_RE.search(prompt_text) is None and audio_list:
                prompt_text = self._build_default_prompt(prompt_text, has_audio=True)
            elif self._AUDIO_SPAN_RE.search(prompt_text) is None and not audio_list:
                prompt_text = self._build_default_prompt(prompt_text, has_audio=False)
            input_ids_list = self._build_input_from_prompt(prompt_text, token_lens)
        elif self.chat_template is not None:
            # 模板分支只传第一段长度；多段音频应使用上面的显式 prompt 分支。
            input_ids_list = self._build_input_from_template(
                token_lens[0] if token_lens else 0
            )
        else:
            raise ValueError(
                "Either provide text or load a chat_template before calling the processor."
            )

        input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
        # 二者均为 [1, sequence_length]、int64；此处文本没有 padding，所以 mask 全为 1。
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        # audio_data 为浮点 Mel，audio_data_seqlens 为帧数，input_ids 为离散词表 ID。
        # 调用方另用 input_ids == audio_token_id 得到 bool 型 audio_input_mask；
        # 时间数字和音频起止标记不属于待替换的音频位置。
        data = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        if audio_batch is not None:
            data["audio_data"] = audio_batch
            data["audio_data_seqlens"] = seqlens_tensor
        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self._base_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._base_tokenizer.decode(*args, **kwargs)


__all__ = ["MelConfig", "MossAudioProcessor"]
