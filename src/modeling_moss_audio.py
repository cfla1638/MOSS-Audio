# 阅读路线：MossAudioEncoder.forward 分块 -> _encode_chunk_batch 编码 ->
# MossAudioModel.forward 将音频嵌入文本 -> DeepStack 注入 -> logits / loss。
# 形状名称约定（不同轴即使大小相同，也代表不同含义）：
#   batch_size：文本样本数；sequence_length：本次送入语言模型的 token 数。
#   audio_count：输入音频段数；mel_bins：频率通道数（当前卷积投影按 128 配置）。
#   audio_frames / padded_audio_frames：有效 / 补齐后的 Mel 时间帧数。
#   total_chunks：所有音频的分块总数；chunk_batch_size：一次编码的块数。
#   chunk_frames / chunk_tokens：块内卷积前的帧数 / 卷积后的 token 数。
#   total_audio_tokens：所有块去掉 padding 后的音频 token 总数。
#   conv_channels：卷积通道数；encoder_hidden_size：编码器内部特征维度。
#   encoder_output_size：编码器输出维度；language_hidden_size：Qwen3 隐藏维度。
#   adapter_hidden_size：模态适配器中间维度；vocab_size：词表大小。
#   padded_chunk_frames / padded_chunk_tokens：块补齐后的帧数 / 卷积输出长度。
#   max_valid_chunk_tokens：当前小批的最大有效 token 数；max_chunk_tokens：所有块的最大值。
# 配置对应：conv_channels = downsample_hidden_size，encoder_hidden_size = d_model，
# encoder_output_size = output_dim，language_hidden_size = language_config.hidden_size。
# shape 中的 1 是实际单例轴；下方说明它是广播轴还是拼接后添加的轴。
from typing import Optional, List, Union, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils.auto_docstring import auto_docstring
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from src.configuration_moss_audio import MossAudioEncoderConfig, MossAudioConfig


class SinusoidsPositionEmbedding(nn.Module):
    # 动态构造正弦位置编码，输出可沿音频块批维广播；当前 num_positions 未参与计算。
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(embedding_dim // 2).float()
        )
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)
        # [encoder_hidden_size // 2]，buffer 随模型迁移设备，但不作为训练参数。

    def forward(self, seq_len: int, device: torch.device):
        # [chunk_tokens, 1] * [1, encoder_hidden_size // 2]
        # -> [chunk_tokens, encoder_hidden_size // 2]，对位置和频率做广播乘法。
        scaled_time = torch.arange(
            seq_len, device=device, dtype=self.inv_timescales.dtype
        ).unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        # 拼接 sin/cos -> [chunk_tokens, encoder_hidden_size]（要求隐藏维度为偶数）。
        # -> [1, chunk_tokens, encoder_hidden_size]，首轴用于广播到各块。
        return pos_emb.unsqueeze(0)


class MossAudioEncoder(nn.Module):
    """先用二维卷积压缩时频轴，再用 Whisper 编码层在各音频块内部提取特征。"""

    def __init__(self, config: MossAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()

        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # 默认频率轴 128 -> 64 -> 32 -> 16；这里的 16 写死，不能任意更改 mel_bins。
        # 每个时间位置展开为 conv_channels * 16，再映射到 encoder_hidden_size。
        self.stem_proj = nn.Linear(config.downsample_hidden_size * 16, config.d_model)
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else nn.Identity()
        )

        self.deepstack_encoder_layer_indexes = list(
            config.deepstack_encoder_layer_indexes or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(self.deepstack_encoder_layer_indexes)
        }

        self.n_window = int(config.n_window)
        # 默认每块 400 Mel 帧，按 100 帧/秒约 4 秒，卷积后为 50 个音频 token。
        self.chunk_frames = int(self.n_window * 2)
        # 每次处理多少个块，不是每块的帧数，也不是文本样本的 batch_size。
        self.conv_chunksize = int(config.conv_chunksize)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        # 输入/输出同形（如 [total_chunks]），只把各长度变为 ceil(原长度 / 8)。
        def conv_out_len(L):
            return (L - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    def _encode_chunk_batch(
        self,
        input_features: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """编码一批已补齐的音频块，返回最终特征和按配置顺序收集的中间层特征。

        input_features: [chunk_batch_size, mel_bins, padded_chunk_frames]。
        seq_lengths: [chunk_batch_size]，各块的有效帧数，int64。
        返回张量及每个中间层张量均为
        [chunk_batch_size, max_valid_chunk_tokens, encoder_output_size]。
        此时仍保留块内 padding，外层 forward 会统一去除。
        """
        if input_features.dim() == 2:
            # 单块 [mel_bins, padded_chunk_frames] -> [1, mel_bins, padded_chunk_frames]。
            input_features = input_features.unsqueeze(0)

        downsampled_lengths = self._compute_downsampled_length(seq_lengths)

        # [chunk_batch_size, mel_bins, padded_chunk_frames]
        # -> [chunk_batch_size, 1, mel_bins, padded_chunk_frames]，添加卷积输入通道轴。
        x = input_features.unsqueeze(1)
        # 每层的频率和时间长度都变为 ceil(原长度 / 2)，GELU 不改变形状。
        # -> [chunk_batch_size, conv_channels, ceil(mel_bins/2), ceil(padded_chunk_frames/2)]。
        x = self.gelu(self.conv1(x))
        # -> [chunk_batch_size, conv_channels, ceil(mel_bins/4), ceil(padded_chunk_frames/4)]。
        x = self.gelu(self.conv2(x))
        # -> [chunk_batch_size, conv_channels, 16, ceil(padded_chunk_frames/8)]（mel_bins=128）。
        x = self.gelu(self.conv3(x))

        # 转置 -> [chunk_batch_size, padded_chunk_tokens, conv_channels, 16]；
        # 展平最后两轴 -> [chunk_batch_size, padded_chunk_tokens, conv_channels * 16]。
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        # 只映射最后一轴 -> [chunk_batch_size, padded_chunk_tokens, encoder_hidden_size]。
        x = self.stem_proj(x)

        # 裁去当前小批所有块都无效的尾部；各块各自的 padding 仍在。
        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

        positions = self.embed_positions(x.shape[1], x.device)
        # [1, max_valid_chunk_tokens, encoder_hidden_size] 广播相加，x 形状不变。
        x = x + positions.to(x.dtype)

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :] >= downsampled_lengths[:, None]
        )
        attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(x.dtype).min
        # padding_mask: [chunk_batch_size, max_valid_chunk_tokens]，True 表示无效。
        # 加性 attention_mask：有效位置为 0，无效位置为当前浮点类型的极小值。
        # -> [chunk_batch_size, 1, 1, max_valid_chunk_tokens]，广播到注意力头与 query 轴。
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        deepstack_hidden_states: List[Optional[torch.Tensor]] = [None] * len(
            self.deepstack_encoder_layer_indexes
        )
        for layer_idx, layer in enumerate(self.layers):
            # 每层保持 [chunk_batch_size, max_valid_chunk_tokens, encoder_hidden_size]。
            # 不同块作为独立批元素，此处注意力不会跨块；不是因果注意力。
            x = layer(
                x,
                attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )[0]
            capture_idx = self._deepstack_capture_map.get(layer_idx)
            if capture_idx is not None:
                # 索引从 0 开始，保存该层输出；列表轴表示捕获层，不是张量的批维。
                deepstack_hidden_states[capture_idx] = x

        x = self.layer_norm(x)
        # -> [chunk_batch_size, max_valid_chunk_tokens, encoder_output_size]。
        x = self.out_proj(x)

        ordered_deepstack_hidden_states = [
            h for h in deepstack_hidden_states if h is not None
        ]
        if not isinstance(self.out_proj, nn.Identity):
            # 中间层复用 out_proj，但没有经过上面的最终 layer_norm。
            ordered_deepstack_hidden_states = [
                self.out_proj(h) for h in ordered_deepstack_hidden_states
            ]
        return x, ordered_deepstack_hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        output_deepstack_hidden_states: bool = True,
    ) -> BaseModelOutputWithPast:
        """
        输入：
        - input_features: [audio_count, mel_bins, padded_audio_frames] 或 [mel_bins, total_audio_frames]。
        - feature_lens: [audio_count]，int64，单位是 Mel 帧。
        - output_deepstack_hidden_states: 是否返回中间层特征。

        默认分块长度 400 的例子：两段有效帧数 [450, 200] -> 块长度 [400, 50, 200]
        -> 有效 token 数 [50, 7, 25] -> 最终输出 [1, 82, encoder_output_size]。
        默认块长可被 8 整除，所以分块下采样后的总长度与逐段 ceil(audio_frames/8) 一致；
        若修改块长为非 8 的倍数，每块向上取整可能导致与 processor 占位符数量不一致。
        """
        
        if input_features.dim() == 3:
            if feature_lens is None:
                # feature_lens 为空表示没有 padding, 每段的有效帧数就是各自的 padded_audio_frames。
                feature_lens = torch.full(
                    (input_features.size(0),),
                    input_features.size(-1),
                    dtype=torch.long,
                    device=input_features.device,
                )
            else:
                feature_lens = feature_lens.to(
                    device=input_features.device, dtype=torch.long
                )
            # 每段先去 padding 再沿时间连接
            valid_chunks = [
                input_features[i, :, : int(feature_lens[i].item())]
                for i in range(int(input_features.shape[0]))
            ]   # [mel_bins, total_audio_frames];  total_audio_frames = sum(feature_lens)
            input_features = torch.cat(valid_chunks, dim=1)
            
        elif input_features.dim() != 2:
            raise ValueError(
                f"Expected [n_mels, T] or [B, n_mels, T], got {tuple(input_features.shape)}."
            )

        # 对于只有一段音频的情况，feature_lens 可能为 None，表示没有 padding，直接使用输入的帧数。
        if feature_lens is None:
            feature_lens = torch.tensor(
                [int(input_features.shape[1])],
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            feature_lens = feature_lens.to(
                device=input_features.device, dtype=torch.long
            )

        chunk_frames = int(self.chunk_frames)
        chunk_num = torch.ceil(
            feature_lens.to(torch.float32) / float(chunk_frames)
        ).long()    # [audio_count]，每段音频的 chunk 数

        # 每个 chunk 的长度，除了最后一块，都是 chunk_frames；最后一块可能更短。
        chunk_lengths = torch.full(
            (int(chunk_num.sum().item()),),
            chunk_frames,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % chunk_frames
        chunk_lengths[chunk_lengths == 0] = chunk_frames

        # 转置为 [total_audio_frames, mel_bins]，按各块真实长度分成列表。
        # 每个元素 [valid_chunk_frames, mel_bins]，分块不跨越原始音频的边界。
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        # pad_sequence -> [total_chunks, padded_chunk_frames, mel_bins]；
        # transpose -> [total_chunks, mel_bins, padded_chunk_frames]。
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        feature_lens_after_cnn = self._compute_downsampled_length(chunk_lengths)
        # [total_chunks]，记录每块有效 token 数；t_down_max 是这些长度的最大值。
        t_down_max = (
            int(feature_lens_after_cnn.max().item())
            if feature_lens_after_cnn.numel() > 0
            else 0
        )
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(int(L.item()), dtype=torch.bool, device=padded_feature.device)
                for L in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        # [total_chunks, max_chunk_tokens]，bool，True 为有效 token（与 padding_mask 相反）。
        if padded_mask_after_cnn.shape[1] < t_down_max:
            padded_mask_after_cnn = F.pad(
                padded_mask_after_cnn,
                (0, t_down_max - padded_mask_after_cnn.shape[1]),
                value=False,
            )

        num_deepstack = len(self.deepstack_encoder_layer_indexes)
        padded_embeds: List[torch.Tensor] = []
        deepstack_padded_embeds: List[List[torch.Tensor]] = [
            [] for _ in range(num_deepstack)
        ]
        for feat_chunk, len_chunk in zip(
            padded_feature.split(self.conv_chunksize, dim=0),
            chunk_lengths.split(self.conv_chunksize, dim=0),
        ):
            # 按块的批轴切小批：feat_chunk 为 [chunk_batch_size, mel_bins, padded_chunk_frames]，
            # len_chunk 为 [chunk_batch_size]，chunk_batch_size <= conv_chunksize。
            out, deepstack_outs = self._encode_chunk_batch(feat_chunk, len_chunk)
            # 小批输出的时间轴可能更短，补到全局 max_chunk_tokens 才能沿批轴拼接。
            if out.shape[1] < t_down_max:
                out = F.pad(out, (0, 0, 0, t_down_max - out.shape[1]))
            padded_embeds.append(out)
            if output_deepstack_hidden_states and num_deepstack > 0:
                if len(deepstack_outs) != num_deepstack:
                    raise RuntimeError(
                        "Deepstack output count does not match configured layer indexes."
                    )
                for capture_idx, ds in enumerate(deepstack_outs):
                    if ds.shape[1] < t_down_max:
                        ds = F.pad(ds, (0, 0, 0, t_down_max - ds.shape[1]))
                    deepstack_padded_embeds[capture_idx].append(ds)

        if padded_embeds:
            # -> [total_chunks, max_chunk_tokens, encoder_output_size]。
            padded_embed = torch.cat(padded_embeds, dim=0)
        else:
            padded_embed = torch.empty(
                (0, t_down_max, self.config.output_dim),
                device=padded_feature.device,
            )

        # 二维 bool 索引合并前两轴并移除 padding，按音频顺序、块顺序、块内时间排列。
        valid_tokens = padded_embed[padded_mask_after_cnn]  # [total_audio_tokens, encoder_output_size]
        # 首轴 1 是对打包结果新加的轴，不表示原始 audio_count。
        last_hidden_state = valid_tokens.unsqueeze(0)  # [1, total_audio_tokens, encoder_output_size]

        deepstack_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_deepstack_hidden_states and num_deepstack > 0:
            collected: List[torch.Tensor] = []
            for chunks_list in deepstack_padded_embeds:
                # 每个捕获层使用相同 mask，得到 [1, total_audio_tokens, encoder_output_size]。
                if chunks_list:
                    ds = torch.cat(chunks_list, dim=0)
                    collected.append(ds[padded_mask_after_cnn].unsqueeze(0))
                else:
                    collected.append(
                        torch.empty(
                            (1, 0, self.config.output_dim),
                            device=padded_feature.device,
                            dtype=padded_embed.dtype,
                        )
                    )
            deepstack_states = tuple(collected)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            hidden_states=deepstack_states,
        )


class GatedMLP(nn.Module):
    # 门控适配器：两条输入投影逐元素相乘，再投影到输出特征空间。
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # [..., input_size] -> 两路 [..., adapter_hidden_size]
        # -> SiLU(gate) * up（形状不变）-> [..., output_size]；前面的序列轴全部保留。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@auto_docstring
class MossAudioPreTrainedModel(PreTrainedModel):
    # Hugging Face 加载、设备分配和后端能力的元信息；实际前向见 MossAudioModel。
    config_class = MossAudioConfig
    config: MossAudioConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer}


class MossAudioModel(MossAudioPreTrainedModel, GenerationMixin):
    config_class = MossAudioConfig
    _tied_weights_keys: List[str] = []

    def __init__(self, config: MossAudioConfig):
        super().__init__(config)

        self.audio_encoder = MossAudioEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)

        self.audio_adapter = GatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.language_config.hidden_size,
        )

        deepstack_k = len(getattr(config.audio_config, "deepstack_encoder_layer_indexes", []) or [])
        # 最终层音频特征用 audio_adapter；每个选中的中间层另有独立的门控适配器。
        if config.deepstack_num_inject_layers is not None:
            deepstack_k = min(deepstack_k, int(config.deepstack_num_inject_layers))
        self.deepstack_audio_merger_list = nn.ModuleList(
            [
                GatedMLP(
                    input_size=config.audio_config.output_dim,
                    hidden_size=config.adapter_hidden_size,
                    output_size=config.language_config.hidden_size,
                )
                for _ in range(deepstack_k)
            ]
        )

        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(config.language_config.hidden_size, self.vocab_size, bias=False)
        # Qwen3Model 输出隐藏状态，这个 lm_head 才把最后一轴映射到词表 logits。
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(self, input_features, feature_lens):
        # 返回最终特征和中间层列表；各张量均为 [1, total_audio_tokens, encoder_output_size]。
        audio_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lens=feature_lens,
            output_deepstack_hidden_states=True,
        )
        deepstack = list(audio_outputs.hidden_states) if audio_outputs.hidden_states is not None else None
        return audio_outputs.last_hidden_state, deepstack

    def _apply_deepstack_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        audio_input_mask: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [batch_size, sequence_length, language_hidden_size]。
        # audio_input_mask: [batch_size, sequence_length]，True 仅对应音频占位 token。
        # deepstack_embeds: [1, total_audio_tokens, language_hidden_size]。
        audio_input_mask = audio_input_mask.to(hidden_states.device)
        deepstack_embeds = deepstack_embeds.to(hidden_states.device, hidden_states.dtype)
        flat = deepstack_embeds.reshape(-1, deepstack_embeds.shape[-1])
        # flat 和 hs[audio_input_mask] 都是 [total_audio_tokens, language_hidden_size]。
        # 只给音频位置加残差；输出 hs 的完整形状保持不变。
        hs = hidden_states.clone()
        hs[audio_input_mask] = hs[audio_input_mask] + flat
        return hs

    def _register_llm_deepstack_hooks(
        self,
        audio_input_mask: torch.Tensor,
        deepstack_audio_embeds: List[torch.Tensor],
    ):
        # 将第几个捕获层的适配特征，加到 Qwen3 第几个解码层的输出上。
        # 使用 forward hook，因此发生在该层完成计算之后，而不是进入该层之前。
        if deepstack_audio_embeds is None or len(deepstack_audio_embeds) == 0:
            return []

        layers = getattr(self.language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen3Model does not expose `.layers`; cannot register DeepStack hooks.")

        num_inject = len(deepstack_audio_embeds)
        handles = []

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= num_inject:
                break

            def _make_llm_hook(k: int):
                def _hook(_module, _inputs, _output):
                    if isinstance(_output, (tuple, list)):
                        hs = _output[0]
                        new_hs = self._apply_deepstack_to_hidden_states(
                            hs, audio_input_mask, deepstack_audio_embeds[k]
                        )
                        return (new_hs,) + tuple(_output[1:])
                    else:
                        return self._apply_deepstack_to_hidden_states(
                            _output, audio_input_mask, deepstack_audio_embeds[k]
                        )

                return _hook

            handles.append(layer.register_forward_hook(_make_llm_hook(layer_idx)))

        return handles

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # input_ids / labels: [batch_size, sequence_length]，int64。
        # inputs_embeds: [batch_size, sequence_length, language_hidden_size]。
        # audio_data: [audio_count, mel_bins, padded_audio_frames]，或打包的二维 Mel。
        # audio_data_seqlens: [audio_count]；audio_input_mask: [batch_size, sequence_length]。
        # attention_mask 通常是 [batch_size, total_context_length]，缓存解码时包含历史。
        # position_ids 通常为 [batch_size, sequence_length]；cache_position 为 [sequence_length]。
        # 音频打包顺序必须与 audio_input_mask 按行选出的位置顺序一致。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 查词嵌入表：[batch_size, sequence_length] -> [..., language_hidden_size]。
            inputs_embeds = self.get_input_embeddings()(input_ids)

        hook_handles = []
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError("audio_input_mask is required when audio_data is provided.")

            audio_embeds, deepstack = self.get_audio_features(audio_data, audio_data_seqlens)
            audio_embeds = self.audio_adapter(audio_embeds)
            # [1, total_audio_tokens, encoder_output_size]
            # -> [1, total_audio_tokens, language_hidden_size]，音频 token 数不变。

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            # mask: [batch_size, sequence_length] -> [..., 1] -> [..., language_hidden_size]。
            # masked_scatter_ 按展平顺序用音频特征替换占位符嵌入，完整序列形状不变。
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if deepstack is not None and len(self.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.deepstack_audio_merger_list)]):
                    # 每个中间层同样映射到 [1, total_audio_tokens, language_hidden_size]。
                    ds = self.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                try:
                    hook_handles = self._register_llm_deepstack_hooks(audio_input_mask, deepstack_audio_embeds)
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            # 本次前向结束即移除 hook，避免之后的调用重复注入音频特征。
            for h in hook_handles:
                h.remove()

        hidden_states = outputs[0]
        # [batch_size, sequence_length, language_hidden_size]
        # -> [batch_size, sequence_length, vocab_size]，每个位置预测下一个 token。
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # logits 去掉最后位置，labels 去掉首位置，实现“当前位置预测下一位置”。
            # shift_logits: [batch_size, sequence_length - 1, vocab_size]。
            # shift_labels: [batch_size, sequence_length - 1]。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            # 合并样本和时间轴 -> [batch_size * (sequence_length - 1), vocab_size]
            # 与 [batch_size * (sequence_length - 1)]；标签为 -100 的位置不贡献损失。
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # loss 是零维标量 []，默认对未屏蔽的目标 token 求平均。

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        # 首次 prefill 使用完整提示词和音频；后续利用 KV cache，只处理最后一个 token。
        # 缓存通常按层保存 key/value，其逻辑形状为
        # [batch_size, num_key_value_heads, cached_tokens, head_size]。
        position_ids = kwargs.get("position_ids", None)
        if cache_position is not None and cache_position[0] > 0:
            # [batch_size, accumulated_sequence_length] -> [batch_size, 1]。
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            audio_data = None
            # 历史音频信息已进入缓存，后续不重复编码音频或注册 DeepStack 注入。
            audio_input_mask = None
            audio_data_seqlens = None
        else:
            audio_data = kwargs.get("audio_data", None)
            audio_input_mask = kwargs.get("audio_input_mask", None)
            audio_data_seqlens = kwargs.get("audio_data_seqlens", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            # attention_mask 仍覆盖历史与当前 token，并不随 input_ids 一起裁成长度 1。
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio_data": audio_data,
                "audio_input_mask": audio_input_mask,
                "audio_data_seqlens": audio_data_seqlens,
            }
        )

        return model_inputs


__all__ = [
    "MossAudioEncoderConfig",
    "MossAudioConfig",
    "MossAudioModel",
]

if __name__ == "__main__":
    # 测试音频编码器的前向输出形状。
    config = MossAudioEncoderConfig()
    encoder = MossAudioEncoder(config)
    encoder.eval()

    # 模拟两段音频，Mel 频谱为 [audio_count, mel_bins, padded_audio_frames]。
    audio_count = 2
    mel_bins = 128
    padded_audio_frames = 450  # 假设每段音频补齐到 800 帧
    input_features = torch.randn(audio_count, mel_bins, padded_audio_frames)

    # 每段音频的有效帧数
    feature_lens = torch.tensor([450, 200], dtype=torch.long)

    with torch.no_grad():
        outputs = encoder(input_features=input_features, feature_lens=feature_lens)
        last_hidden_state = outputs.last_hidden_state
        deepstack_states = outputs.hidden_states

        print("Last hidden state shape:", last_hidden_state.shape)
        if deepstack_states is not None:
            for i, ds in enumerate(deepstack_states):
                print(f"DeepStack layer {i} shape:", ds.shape)
