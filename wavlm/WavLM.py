# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
#
# ALL THE CODE IN THIS FILE IS FROM WAVLM -- please see the original repo for more details and attribution:
# https://github.com/microsoft/unilm/tree/master/wavlm
# --------------------------------------------------------

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    init_bert_params,
    get_activation_fn,
    TransposeLast,
    GLU_Linear,
)

logger = logging.getLogger(__name__)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    根據輸入形狀隨機產生 mask span，常見於語音/文本遮蔽式預訓練。

    參數說明：
        shape: 欲產生 mask 的 shape，通常是 (batch_size, timesteps)
        padding_mask: 選用，padding 部分不能被 mask
        mask_prob: 每個 token 被選為 mask 起點的機率（大致決定要 mask 多少比例）
        mask_length: 每個 mask span 的長度
        mask_type: mask span 長度的抽樣方法
        mask_other: 部分 mask_type（如 uniform, normal）要用的參數
        min_masks: 最少要產生幾個 mask span
        no_overlap: 若 True，禁止 mask span 重疊
        min_space: 當 no_overlap 時，span 之間最少要保留多少未遮蔽元素
    """

    # 解析 batch 大小與時間軸長度
    bsz, all_sz = shape
    # 先建立一個全部為 False 的 mask 陣列
    mask = np.full((bsz, all_sz), False)

    # 預估整個 batch 應該要產生多少個 mask span
    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()  # 加一點隨機性，防止只取整數時 always bias
    )
    # 確保最少產生 min_masks 個 mask span
    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        # 如果有 padding，不能遮蔽 padding 部分，要重新算可用長度
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            # 重新計算這一筆資料要 mask 幾次
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        # 根據 mask_type 決定每個 span 的長度
        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        # 如果全部長度加起來是 0，避免空 mask，最少給一個
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            # 如果不允許重疊，要用較複雜的方式分配 mask
            mask_idc = []

            def arrange(s, e, length, keep_length):
                # 在區間 [s, e) 隨機選一個起點
                span_start = np.random.randint(s, e - length)
                # 把這個 span 的所有 index 加進去
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                # 左邊如果還夠長，留給後續 mask 用
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                # 右邊如果還夠長，也留給後續 mask 用
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]  # 可用的區間初始化為整個句子
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                # 檢查所有可用區間是否還有空間給這個長度
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break  # 沒空間就提早結束
                # 根據長度機率選一段區間
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            # 允許重疊，就可以直接抽 num_mask 個起點
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            # 從 sz-min_len 之間抽取不重複的 num_mask 個起點
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            # 根據每個 span 長度，展開所有被 mask 的 index
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        # 只保留在 sz 範圍內的 mask index，然後記錄
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    # 保證每個 batch 都有一樣長度的 mask（多的會被隨機抽掉）
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True  # 把 mask index 設成 True

    return mask  # 最終回傳 (batch_size, seq_len) 的布林 mask 矩陣


class WavLMConfig:
    def __init__(self, cfg=None):
        # --- 特徵萃取與編碼器參數 ---
        self.extractor_mode: str = (
            "default"  # 特徵擷取器模式：預設是 group norm，"layer_norm" 則每一層都 layer norm（適合 normalize=True 用）
        )
        self.encoder_layers: int = 12  # Transformer 編碼層數（深度）
        self.encoder_embed_dim: int = 768  # Transformer 的隱藏層維度
        self.encoder_ffn_embed_dim: int = 3072  # FFN（前饋網路）的隱藏層維度
        self.encoder_attention_heads: int = 12  # 注意力頭數
        self.activation_fn: str = "gelu"  # 啟動函數（常見選項：gelu、relu）

        self.layer_norm_first: bool = False  # 是否在 transformer block 前面先 layernorm
        # 卷積特徵抽取層設定（dim, kernel_size, stride）
        self.conv_feature_layers: str = (
            "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
        )
        self.conv_bias: bool = False  # 卷積層要不要 bias
        self.feature_grad_mult: float = (
            1.0  # 特徵抽取層梯度倍率（通常用來調整學習速率）
        )

        self.normalize: bool = False  # 訓練時要不要把輸入做 0 均值、1 標準差正規化

        # --- Dropout 相關 ---
        self.dropout: float = 0.1  # Transformer 裡的 dropout 機率
        self.attention_dropout: float = 0.1  # 注意力機制裡的 dropout
        self.activation_dropout: float = 0.0  # FFN 裡啟動函數後的 dropout
        self.encoder_layerdrop: float = 0.0  # Transformer 層的 drop 機率（整層丟掉）
        self.dropout_input: float = 0.0  # 特徵抽取後、丟進 Transformer 前的 dropout
        self.dropout_features: float = 0.0  # 特徵抽取完的特徵 dropout

        # --- Masking 參數（遮蔽訓練用，像 wav2vec2 的 mask）---
        self.mask_length: int = 10  # 遮蔽 span 的長度
        self.mask_prob: float = 0.65  # 每個 token 被選為遮蔽起點的機率
        self.mask_selection: str = (
            "static"  # 遮蔽長度抽樣方式（static/normal/uniform/poisson）
        )
        self.mask_other: float = 0  # 進階遮蔽參數，供 uniform/normal 等模式用
        self.no_mask_overlap: bool = False  # 遮蔽 span 之間可否重疊
        self.mask_min_space: int = (
            1  # no_mask_overlap=True 時，遮蔽 span 之間最少保留幾個未遮蔽
        )

        # --- Channel masking（feature 軸遮蔽，像 SpecAugment）---
        self.mask_channel_length: int = 10  # channel 遮蔽長度
        self.mask_channel_prob: float = 0.0  # channel 被遮蔽機率
        self.mask_channel_selection: str = "static"  # channel 遮蔽長度抽樣方式
        self.mask_channel_other: float = 0  # 進階 channel 遮蔽參數
        self.no_mask_channel_overlap: bool = False  # channel 遮蔽可否重疊
        self.mask_channel_min_space: int = 1  # channel 遮蔽間最少保留幾個未遮蔽

        # --- 位置編碼（Positional Embedding）---
        self.conv_pos: int = 128  # 卷積位置編碼的 filter 數量
        self.conv_pos_groups: int = 16  # 卷積位置編碼的群組數

        # --- 相對位置編碼（Relative Position Embedding）---
        self.relative_position_embedding: bool = False  # 是否啟用相對位置編碼
        self.num_buckets: int = 320  # 相對位置 bucket 數
        self.max_distance: int = 1280  # 最大距離
        self.gru_rel_pos: bool = False  # 是否 gated 相對位置編碼

        # --- 外部設定匯入 ---
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        # 用 dictionary 方式快速更新所有參數
        self.__dict__.update(cfg)


class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,  # 傳入剛剛定義的設定檔，管理所有架構/訓練參數
    ) -> None:
        super().__init__()
        logger.info(f"WavLM Config: {cfg.__dict__}")

        self.cfg = cfg
        # 解析特徵萃取用的卷積層參數，格式是 [(通道數, kernel大小, stride), ...]
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]  # 取最後一層的 channel 當 embedding 維度

        # 特徵抽取器，使用一連串卷積層
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        # 如果卷積特徵維度和 transformer encoder 維度不同，加一個線性投影
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # 記錄各種 masking 相關設定（遮蔽比例、長度、方式...）
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        # dropout 設定
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult  # 特徵梯度倍率

        # 遮蔽 token 的 embedding，模型會自己學習
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        # Transformer 編碼器
        self.encoder = TransformerEncoder(cfg)
        # 特徵正規化
        self.layer_norm = LayerNorm(self.embed)

    # ---- 時間軸與 channel masking 方法 ----
    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape  # B: batch, T: 時間長度, C: channel/特徵維度
        if self.mask_prob > 0:
            # 計算時間軸要遮蔽的位置
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            # 將遮蔽位置的特徵全部設成 mask_emb
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            # 計算 channel 軸要遮蔽的位置
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)  # 在時間軸複製成 (B, T, C)
                .expand(-1, T, -1)
            )
            # channel masking 直接把特徵設為 0
            x[mask_channel_indices] = 0

        return x, mask_indices

    # ---- padding mask 製作（避免把 padding 區域當成真實訊號處理）----
    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 先修正 padding mask 長度對齊特徵維度
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        # 對齊後，把 padding mask reshape 成 (batch, seq_len, -1)，再合併
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    # ---- 主要特徵抽取 function ----
    def extract_features(
        self,
        source: torch.Tensor,  # 原始語音波形
        padding_mask: Optional[torch.Tensor] = None,  # optional padding mask
        mask: bool = False,  # 是否做 masking（訓練才會打開）
        ret_conv: bool = False,  # 回傳卷積特徵還是 Transformer 特徵
        output_layer: Optional[int] = None,  # 指定回傳到哪一層
        ret_layer_results: bool = False,  # 是否要回傳每層結果
    ):

        # feature_grad_mult > 0 代表要做反向傳播
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            # 若梯度倍率不等於 1，做自訂放大/縮小
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            # 不需要梯度的情況下（只推論），避免計算梯度省空間
            with torch.no_grad():
                features = self.feature_extractor(source)

        # 特徵軸交換：(B, D, T) -> (B, T, D)
        features = features.transpose(1, 2)
        # 做 LayerNorm
        features = self.layer_norm(features)

        # 處理 padding mask，讓特徵長度和 mask 對齊
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        # 如果卷積特徵和 transformer embedding 維度不同，要先線性轉換
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # 特徵 dropout，防止 overfit
        features = self.dropout_input(features)

        # 若 mask=True（訓練自監督時），時間軸或 channel 軸做遮蔽
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features

        # 丟給 transformer encoder，拿到每一層的輸出
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        # 統一整理回傳內容
        res = {
            "x": x,
            "padding_mask": padding_mask,
            "features": features,
            "layer_results": layer_results,
        }

        # ret_conv=True 就回傳卷積特徵，否則回傳最後一層 encoder 特徵
        feature = res["features"] if ret_conv else res["x"]
        # 如果要每層結果就用 tuple 回傳
        if ret_layer_results:
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"]


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[
            Tuple[int, int, int]
        ],  # [(dim, kernel, stride), ...] 由 config 決定
        dropout: float = 0.0,  # 每層卷積後的 dropout 機率
        mode: str = "default",  # "default"=group norm, "layer_norm"=layer norm
        conv_bias: bool = False,  # 卷積層要不要 bias
        conv_type: str = "default",  # 支援 1D、2D、custom 多種捲積方式
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}  # 只支援這兩種 mode

        # 卷積層 block，依照需求選擇加 norm 或不用
        def block(
            n_in,  # 輸入 channel 數
            n_out,  # 輸出 channel 數
            k,  # 卷積 kernel 大小
            stride,  # 卷積步長
            is_layer_norm=False,  # 是否加 layer norm
            is_group_norm=False,  # 是否加 group norm（通常只有第一層才會 group norm）
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(
                    conv.weight
                )  # kaiming 初始化（比較適合 relu/gelu）
                return conv

            # 兩種 norm 不可同時開
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            # 如果要 layer norm
            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),  # 維度換到最後一軸，方便 LayerNorm
                        Fp32LayerNorm(dim, elementwise_affine=True),  # 做 LayerNorm
                        TransposeLast(),
                    ),
                    nn.GELU(),  # 啟動函數
                )
            # 如果是 group norm（只有第一層會用）
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(
                        dim, dim, affine=True
                    ),  # dim groups, 每 group 1 channel
                    nn.GELU(),
                )
            else:
                # 普通卷積，不加 norm
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type

        if self.conv_type == "default":
            # 最標準：多層 1D conv
            in_d = 1  # 語音進來只有 1 channel
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default"
                        and i == 0,  # 第一層才 group norm
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim  # 下一層的輸入變成這一層的輸出 channel
        elif self.conv_type == "conv2d":
            # 支援 2D 卷積（通常用在頻譜/梅爾特徵）
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            # 一些自定義卷積設計
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(torch.nn.LayerNorm([dim, idim]))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass  # 可以自行擴充更多 conv_type

    def forward(self, x, mask=None):
        # 將 (B, T) -> (B, 1, T) 加 channel 維度
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    # 2D 頻譜類維度處理
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))  # 拉平成 1D 特徵
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                # 把 channel 和 freq 軸合併成一個特徵軸
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x  # (B, channel, T) or (B, feat, T)


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout  # Encoder dropout 機率
        self.embedding_dim = args.encoder_embed_dim  # 隱藏層維度

        # ----- 卷積式位置編碼 (convolutional positional embedding) -----
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0  # 卷積初始化計算 std 時用
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        # 加 weight normalization 跟 GELU，並確保 zero padding
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        # ----- 相對位置編碼參數 -----
        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        # ----- 疊多層 Transformer encoder -----
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(
                        self.relative_position_embedding and i == 0
                    ),  # 只有第一層才加相對 attention bias
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first  # 是否在最前面就 layer norm
        self.layer_norm = LayerNorm(self.embedding_dim)  # 最終 layer norm
        self.layerdrop = args.encoder_layerdrop  # Encoder 隨機 drop 機率

        self.apply(init_bert_params)  # 參數初始化

    # ---- 前向傳遞入口 ----
    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        # 如果設定了 "layer_norm_first"，而且沒指定只取到某一層，最後要做 layer norm
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    # ---- 特徵萃取/多層 encoder 主體 ----
    def extract_features(
        self, x, padding_mask=None, streaming_mask=None, tgt_layer=None
    ):

        # 有 padding mask 的位置要設成 0（不讓 attention 用到 padding）
        if padding_mask is not None:
            x[padding_mask] = 0

        # ----- 卷積式位置編碼 -----
        x_conv = self.pos_conv(x.transpose(1, 2))  # (B, T, C) → (B, C, T) → conv1d
        x_conv = x_conv.transpose(1, 2)  # (B, C, T) → (B, T, C)
        x += x_conv  # 跟原本 embedding 相加

        # 前置 layernorm（通常設定在 transformer 前）
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- 為 transformer 準備 PyTorch 標準格式 (T, B, C) -----
        x = x.transpose(0, 1)  # (B, T, C) → (T, B, C)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))  # 支援指定取到第幾層就停

        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            # 進行 layerdrop（訓練才會啟用，有機率直接跳過這層）
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    self_attn_mask=streaming_mask,
                    pos_bias=pos_bias,  # for relative position bias
                )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # (T, B, C) → (B, T, C) 還原成一般格式
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    單一 Transformer Encoder 層（BERT/XLM 樣式）
    """

    def __init__(
        self,
        embedding_dim: float = 768,  # 輸入/輸出特徵維度
        ffn_embedding_dim: float = 3072,  # FFN 隱藏層維度
        num_attention_heads: float = 8,  # 注意力頭數
        dropout: float = 0.1,  # 整體 dropout 機率
        attention_dropout: float = 0.1,  # 注意力 dropout 機率
        activation_dropout: float = 0.1,  # FFN 激活 dropout 機率
        activation_fn: str = "relu",  # FFN 用的 activation function
        layer_norm_first: bool = False,  # LayerNorm 擺前還是後（Pre-LN/Post-LN）
        has_relative_attention_bias: bool = False,  # 第一層是否加相對位置編碼
        num_buckets: int = 0,  # 相對位置編碼的 bucket 數
        max_distance: int = 0,  # 相對位置最大距離
        rescale_init: bool = False,  # 初始化用參數
        gru_rel_pos: bool = False,  # 是否用 gated relative pos
    ) -> None:

        super().__init__()
        # 基本設定
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # 激活函數名稱 & 取對應函數
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)

        # ---- 多頭自注意力層，支援相對位置編碼 ----
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        # dropout & normalization
        self.dropout1 = nn.Dropout(dropout)  # attention 輸出 dropout
        self.dropout2 = nn.Dropout(self.activation_dropout)  # FFN 激活 dropout
        self.dropout3 = nn.Dropout(dropout)  # FFN 輸出 dropout

        self.layer_norm_first = layer_norm_first  # LayerNorm 放在 attention 前還是後

        # attention 輸出的 layernorm
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        # FFN：支援 GLU 或標準線性
        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # FFN 輸出的 layernorm
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,  # (seq, batch, dim)
        self_attn_mask: torch.Tensor = None,  # 遮蔽 attention 的 mask
        self_attn_padding_mask: torch.Tensor = None,  # padding 位置不要算進 attention
        need_weights: bool = False,  # 是否回傳注意力權重
        pos_bias=None,  # 相對位置 bias
    ):
        """
        可選 pre-LN 或 post-LN。
        """
        residual = x  # 殘差分支（每一步都會加回原始輸入）

        if self.layer_norm_first:
            # ----- 前置 LayerNorm -----
            x = self.self_attn_layer_norm(x)
            # ----- 多頭自注意力 -----
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x  # 殘差相加

            residual = x  # 殘差繼續往下
            x = self.final_layer_norm(x)
            # ----- FFN (含 activation) -----
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x  # 殘差相加
        else:
            # ----- 標準 Post-LN 順序 -----
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual + x  # 殘差相加

            x = self.self_attn_layer_norm(x)

            residual = x
            # ----- FFN -----
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x  # 殘差相加
            x = self.final_layer_norm(x)

        return x, attn, pos_bias  # 輸出特徵、(可選)注意力權重、位置 bias
