# 定義模型所需的所有第三方 Python 套件
dependencies = [
    "torch",
    "torchaudio",  # 處理聲音數據的 PyTorch 套件
    "numpy",
    "omegaconf",  # 組態設定檔解析
    "fastprogress",  # 進度條顯示，訓練/推論時看得到進度
    "pandas",
    "jiwer",  # 計算語音識別的誤字率（WER、CER 等指標）
]

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from omegaconf import OmegaConf
from fastprogress.fastprogress import progress_bar
from transfusion.model import TransFusion
from transfusion.diffusion import MultinomialDiffusion, index_to_log_onehot
from transfusion.score import DSH, get_schedule, to_text

from wavlm.WavLM import WavLM, WavLMConfig
from wavlm.extract import WEIGHTINGS


def extract_transfusion_features(wav: Tensor, wavlm: WavLM) -> Tensor:
    """
    將 16kHz 的歸一化波形（float tensor）轉換成可以餵給 TransFusion 模型的 WavLM 特徵。

    參數說明：
    - wav: (1, T) 16kHz 的語音波形，float32 張量
    - wavlm: 已經載入的 WavLM 模型（通常用 wavlm_large() 產生）

    回傳：
    - wavlm_features: (seq_len, dim) 轉換後的特徵向量
    """
    # 將 WEIGHTINGS（每層權重，通常是 list）轉成 tensor，並確保 device 一致
    weighting = torch.tensor(WEIGHTINGS, device=wav.device)[:, None]

    # 把 wav 轉到 WavLM 的同一台 device 上，避免 GPU/CPU 衝突
    wav_input_16khz = wav.to(next(wavlm.parameters()).device)

    # 用 WavLM 抽出所有層的中間特徵
    # ret_layer_results=True 表示同時拿到每層的結果
    rep, layer_results = wavlm.extract_features(
        wav_input_16khz,
        output_layer=wavlm.cfg.encoder_layers,  # 抽到最後一層
        ret_layer_results=True,
    )[
        0
    ]  # [0] 表示只取第一個回傳值

    # 把每一層的特徵（本來 shape 是 (dim, seq_len)）轉成 (seq_len, dim)，然後串在一起
    features = torch.cat(
        [x.transpose(0, 1) for x, _ in layer_results], dim=0
    )  # shape: (n_layers, seq_len, dim)

    # 用 weighting 針對每層做加權平均，最後結果是 (seq_len, dim)
    features = (features * weighting[:, None]).sum(dim=0)

    # 回傳最終加權後的特徵
    return features


def forward_diffusion(cfg, diff, dtype, x, t, c=None):
    """
    簡單版的 forward diffusion 流程（p 過程）：
    目的是把 x 隨著時間步 t 擴散變成更雜訊的樣子（往後加噪音），
    這是 diffusion model 的正向階段。

    參數說明：
    - cfg: 設定檔，裡面有 vocab_size（詞彙表大小）等參數
    - diff: MultinomialDiffusion 物件，管理所有擴散相關 function
    - dtype: 資料型別（例如 torch.float32）
    - x: 當前 step 的標籤（label），通常是 int64（token id）
    - t: 當前時間步（step），int
    - c: 進階用途，給 sequential progressive diffusion offset
    """

    # 把離散的 x 轉成 one-hot
    log_x_t = index_to_log_onehot(x, cfg.vocab_size, dtype=dtype)

    # 根據有沒有提供 c，選用不同版本的 forward 擴散方法
    if c is not None:
        # c 主要用於特殊進階推論（像 RePaint/repainting），加強/調整 diffusion 步驟
        x = diff.q_pred_one_timestep_scaled(log_x_t, t, c, DSH.jump_len)
    else:
        # 標準的 forward diffusion（q）步驟
        x = diff.q_pred_one_timestep(log_x_t, t)

    # 在 log space 做完運算後，實際 sample 一個新的 token（隨機取樣）
    x = diff.log_sample_categorical(x)
    return x  # 回傳當前步驟擴散後的新 label（token）


def reverse_diffusion(
    diff,
    model,
    batch,
    x_known=None,  # 已知的部分（如 inpainting 問題的已知區域）
    m=None,  # mask，1 代表已知、0 代表未知
    last_greedy=False,  # t=0 時是否用 argmax（確定性）而不是 sample
    temperature=1.0,  # 調整模型預測分布的平滑度（越大越均勻）
    alphas=None,  # ensemble 用的 alpha 列表
    ensemble_size=1,  # ensemble 次數
):
    """
    反向擴散過程 q：根據目前 x, t, x_known, m 等資訊，預測 x_{t-1}
    支援 mask（inpainting）、溫度、ensemble 以及最後一層用 argmax。
    """
    x = batch[0]  # 當前 step 的預測 x_t
    t = batch[1]  # 當前時間步 t
    if x_known is None:
        x_known = torch.zeros_like(x)  # 如果沒給已知，預設全部未知
    if m is None:
        m = torch.zeros_like(x)  # 如果沒給 mask，預設全部未知

    # 預測 x_0（乾淨資料）
    # 將 batch 所有資訊餵給模型（通常含條件、padding mask等）
    x_0_pred = model(*batch)  # Equation 8b in論文

    # Guidance (條件引導)
    # 若 guidance_w < 1，混合有條件/無條件預測，做 conditional/unconditional 混合推論
    if DSH.guidance_w != 1:
        uncond_x_0_pred = model(
            x, t, torch.zeros_like(batch[2]), torch.ones_like(batch[3]), batch[-1]
        )
        # guidance_w=1 只用條件，<1 則混合兩種
        x_0_pred = DSH.guidance_w * x_0_pred + (1 - DSH.guidance_w) * uncond_x_0_pred

    # 溫度縮放（控制預測分布的平滑度）
    x_0_pred = x_0_pred / temperature
    # softmax 得到 log 機率分布
    log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)
    # 把 x 轉成 log-onehot
    log_x_t = index_to_log_onehot(x, diff.num_classes, dtype=x_0_pred.dtype)
    # 用 diffusion 模型還原 p(x_{t-1}|x_t) 的 log 機率分布
    log_model_pred = diff.p_pred(log_x_t, t, log_x_0_pred)  # 論文 Equation 8b

    # Ensemble trick（多模型混合推論，提升穩定性
    a_t = alphas[t[0]] if alphas is not None else 0
    mat = torch.eye(ensemble_size, device=x.device) * (1 - a_t)
    mat += 1 / ensemble_size * a_t
    mat = torch.block_diag(*([mat] * (x.shape[0] // ensemble_size)))
    # log_model_pred shape: (batch, ..., vocab)
    log_model_pred = (mat[..., None, None]).log().to(x.dtype) + log_model_pred[None]
    # 對 ensemble 維度做 logsumexp，合併多個模型分布
    log_model_pred = torch.logsumexp(log_model_pred, dim=1)

    # 依據時間步做 sample 或 argmax
    if (t == 0).all() and last_greedy:  # t=0（最後一層）用 argmax
        x_tm1_unknown = log_model_pred.argmax(dim=-1)
    else:  # 其它情況都用抽樣
        x_tm1_unknown = diff.log_sample_categorical(log_model_pred)

    # 有 mask（inpainting 問題）時處理已知區域
    x_known_log = index_to_log_onehot(x_known, diff.num_classes, dtype=x_0_pred.dtype)
    if (t == 0).all():  # t=0 時直接還原
        x_tm1_known = x_known
    else:
        x_tm1_known = diff.q_sample(x_known_log, t)

    # 最終組合已知/未知區域
    x_tm1 = x_tm1_known * m.long() + x_tm1_unknown * (1 - m.long())
    return x_tm1, x_0_pred  # 回傳下個 step 的 x, 以及這一步預測的 x_0


@torch.inference_mode()
def perform_simple_inference(
    model: TransFusion, cond_emb: Tensor, diff: MultinomialDiffusion, vocab, cfg
):
    device = cond_emb.device
    dtype = torch.float32
    bs = cond_emb.shape[0]
    x = torch.randint(
        0,
        diff.num_classes,
        (cond_emb.shape[0], DSH.T_override),
        dtype=torch.long,
        device=cond_emb.device,
    )
    cond_emb = cond_emb.to(device, non_blocking=True)
    cond_padding_mask = torch.zeros_like(cond_emb, dtype=torch.bool)[..., 0]
    cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)
    cond_emb = cond_emb.to(dtype)

    # RePaint paper resample scheduling
    times = get_schedule(cfg.T, jump_n_sample=DSH.jump_n_sample, jump_len=DSH.jump_len)

    x_known = torch.zeros_like(x)
    m = torch.zeros_like(x).bool()

    c = 0  # sequentially progressive diffusion offset (Section 4.2)

    # ensemble bs (not in paper)
    alphas = torch.linspace(1, 0, cfg.T).to(device)

    # See RePaint paper algorithm
    for t_last, t_cur in progress_bar(zip(times[:-1], times[1:]), total=len(times) - 1):

        t = torch.ones((bs,), dtype=torch.long, device=x.device) * (t_last)
        if t_cur < t_last:
            if c > DSH.jump_n_sample:
                c = 0
            c += 1 / DSH.jump_len

            # Reverse diffusion: q
            xx = (x, t, cond_emb, cond_padding_mask, None)
            x, x_0_pred = reverse_diffusion(
                diff,
                model,
                xx,
                x_known,
                m,
                temperature=DSH.x_0_temp,
                alphas=alphas,
                ensemble_size=1,
            )
        else:
            # Forward diffusion: p
            if DSH.enable_kevin_scaled_inference:
                x = forward_diffusion(cfg, diff, dtype, x, t, c=c)
            else:
                x = forward_diffusion(cfg, diff, dtype, x, t, c=None)

    text_preds = [to_text(p, vocab["i2s"]) for p in x]
    return x, text_preds


# ------------------
# torch hub integration functions


def transfusion_small_462k(
    pretrained=True, progress=True, device="cuda"
) -> TransFusion:
    """Best TransFusion model described in the paper, ~250M parameters and trained for
    462 000 updates. A multinomial diffusion ASR model transcribing utterances from their WavLM embeddings.
    """
    if torch.cuda.is_available() == False:
        if str(device) != "cpu":
            logging.warning(
                f"Overriding device {device} to cpu since no GPU is available."
            )
            device = "cpu"
    # load checkpoints
    ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion_462k_slim.pt",
        map_location=device,
        progress=progress,
    )

    device = torch.device(device)
    vocab = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion-vocab.pt",
        map_location="cpu",
        progress=progress,
    )

    # load config
    cfg = OmegaConf.structured(ckpt["cfg_yaml"])
    logging.debug(f"CKPT CONFIG:\n{OmegaConf.to_yaml(cfg)}")
    logging.debug(
        f"Default diffusion sampling hyperparameters:\n{OmegaConf.to_yaml(OmegaConf.create(DSH))}"
    )

    # load model
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(device)
    if pretrained:
        model.load_state_dict(ckpt["module"])
    model.eval()
    print(
        f"TransFusion-small 462k update model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )

    # create diffusion
    diffuser = MultinomialDiffusion(
        cfg.model_cfg.vocab_size,
        cfg.model_cfg.T,
        cfg.model_cfg.diffusion_s,
        device=device,
    )

    model.vocab = vocab
    model.diffuser = diffuser
    model.perform_simple_inference = perform_simple_inference
    model.forward_diffusion = forward_diffusion
    model.reverse_diffusion = reverse_diffusion
    return model


def wavlm_large(pretrained=True, progress=True, device="cuda") -> WavLM:
    """Load the WavLM large checkpoint from the original paper."""
    if torch.cuda.is_available() == False:
        if str(device) != "cpu":
            logging.warning(
                f"Overriding device {device} to cpu since no GPU is available."
            )
            device = "cpu"
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/WavLM-Large.pt",
        map_location=device,
        progress=progress,
    )

    cfg = WavLMConfig(checkpoint["cfg"])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    model.extract_transfusion_features = extract_transfusion_features
    print(
        f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters"
    )
    return model
