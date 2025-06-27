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


@torch.inference_mode()  # 關掉 autograd，推論時不用算梯度
def perform_simple_inference(
    model: TransFusion,  # TransFusion 多分類 diffusion ASR 模型
    cond_emb: Tensor,  # 條件嵌入（通常是聲音的特徵向量）
    diff: MultinomialDiffusion,  # multinomial 擴散模型（負責 forward/reverse/sample）
    vocab,  # 詞彙表，用來還原成文字
    cfg,  # 模型及推論設定
):
    device = cond_emb.device  # 取得特徵所在裝置
    dtype = torch.float32  # 預設 float32 精度
    bs = cond_emb.shape[0]  # 取得 batch 大小

    # 隨機初始化一組 token 做為 diffusion 過程的起始狀態（每個 batch、每個時間步都是亂數）
    x = torch.randint(
        0,  # 最小值
        diff.num_classes,  # 最大值（詞彙表大小）
        (cond_emb.shape[0], DSH.T_override),  # 形狀：(batch, 序列長度)
        dtype=torch.long,  # 使用 long 型別
        device=cond_emb.device,  # 指定裝置
    )

    # 條件嵌入放到對應裝置上
    cond_emb = cond_emb.to(device, non_blocking=True)
    # 建立 padding mask，預設不遮蔽（全為 False）
    cond_padding_mask = torch.zeros_like(cond_emb, dtype=torch.bool)[..., 0]
    # padding mask 也搬到對應裝置
    cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)
    # 條件特徵轉成 float32（防止型別不一致）
    cond_emb = cond_emb.to(dtype)

    # 依照 RePaint 論文的方式，生成 diffusion 過程要經過的時間點（時間排程）
    times = get_schedule(cfg.T, jump_n_sample=DSH.jump_n_sample, jump_len=DSH.jump_len)

    # 全部設為未知區域（inpainting 用，但這裡都設 0）
    x_known = torch.zeros_like(x)
    m = torch.zeros_like(x).bool()  # mask 也全設 False

    c = 0  # progressive diffusion offset，用來調整跳躍的步伐

    # ensemble alpha 參數，支援多模型分布混合（這裡設為線性遞減）
    alphas = torch.linspace(1, 0, cfg.T).to(device)

    # 用進度條跑迴圈，遍歷每個排程中的時間點（決定每步是 forward 還是 reverse）
    for t_last, t_cur in progress_bar(zip(times[:-1], times[1:]), total=len(times) - 1):

        # 每一個 batch 全部設為相同時間步 t
        t = torch.ones((bs,), dtype=torch.long, device=x.device) * (t_last)
        # 判斷這一步是要做 reverse（去雜訊）還是 forward（加雜訊）
        if t_cur < t_last:
            # 如果 jump 次數超過上限就歸零
            if c > DSH.jump_n_sample:
                c = 0
            # 否則逐步累加 offset
            c += 1 / DSH.jump_len

            # 準備 batch 輸入格式，包含 x, t, 條件特徵、padding mask 及其他必要資訊
            xx = (x, t, cond_emb, cond_padding_mask, None)
            # 執行 reverse diffusion，將 x_t 推回 x_{t-1}
            x, x_0_pred = reverse_diffusion(
                diff,  # diffusion 物件
                model,  # TransFusion 模型
                xx,  # 當前 batch（包含 x, t, 條件等）
                x_known,  # inpainting 用的已知區域（這裡全未知）
                m,  # mask
                temperature=DSH.x_0_temp,  # 溫度參數，調整生成平滑度
                alphas=alphas,  # ensemble 混合參數
                ensemble_size=1,  # ensemble 個數（這裡只用一個）
            )
        else:
            # 做 forward diffusion，將 x_{t-1} 加雜訊變 x_t
            if DSH.enable_kevin_scaled_inference:
                # 支援特殊推論模式（會根據 c 調整加雜訊方式）
                x = forward_diffusion(cfg, diff, dtype, x, t, c=c)
            else:
                # 標準 forward diffusion
                x = forward_diffusion(cfg, diff, dtype, x, t, c=None)

    # diffusion 完所有時間步後，把 batch 裡每組 token 編碼轉成文字
    text_preds = [to_text(p, vocab["i2s"]) for p in x]
    # 回傳最終的 token（數字標籤）以及對應的文字結果
    return x, text_preds


# ------------------
# torch hub 整合用的函式


def transfusion_small_462k(
    pretrained=True,  # 是否載入預訓練權重
    progress=True,  # 顯示下載進度
    device="cuda",  # 指定裝置，預設用 GPU
) -> TransFusion:
    """
    載入論文裡最佳的 TransFusion-small 模型（約 2.5 億參數，訓練 462,000 次）
    這是一個 multinomial diffusion ASR 模型，用來從 WavLM 特徵做語音轉文字
    """

    # 如果沒有 GPU，強制使用 CPU 避免報錯
    if torch.cuda.is_available() == False:
        if str(device) != "cpu":
            logging.warning(
                f"Overriding device {device} to cpu since no GPU is available."
            )
            device = "cpu"

    # 下載並載入模型權重（PyTorch Hub 標準方式）
    ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion_462k_slim.pt",
        map_location=device,  # 自動搬到指定 device
        progress=progress,  # 是否顯示下載進度
    )

    # 重新定義 device，確保 torch.device 類型
    device = torch.device(device)

    # 下載並載入 vocab（index to string 映射表，存在 pt 檔裡）
    vocab = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/transfusion-vocab.pt",
        map_location="cpu",
        progress=progress,
    )

    # 解析模型組態設定（OmegaConf 是 Yaml 管理包）
    cfg = OmegaConf.structured(ckpt["cfg_yaml"])
    logging.debug(f"CKPT CONFIG:\n{OmegaConf.to_yaml(cfg)}")
    logging.debug(
        f"Default diffusion sampling hyperparameters:\n{OmegaConf.to_yaml(OmegaConf.create(DSH))}"
    )

    # 建立 TransFusion 模型主體，載入設定檔與最大輸出長度
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(device)
    # 如果指定要載入預訓練參數
    if pretrained:
        model.load_state_dict(ckpt["module"])  # 載入模型參數
    # 切換到 eval 模式（停用 dropout 等訓練專用元件）
    model.eval()
    print(
        f"TransFusion-small 462k update model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )

    # 建立 diffusion 運算器（負責 forward/reverse/sample）
    diffuser = MultinomialDiffusion(
        cfg.model_cfg.vocab_size,  # 詞彙表大小
        cfg.model_cfg.T,  # diffusion 步數
        cfg.model_cfg.diffusion_s,  # schedule 參數
        device=device,  # 裝置
    )

    # 綁定 vocab、diffuser、推論 function 到 model 上，方便直接呼叫
    model.vocab = vocab
    model.diffuser = diffuser
    model.perform_simple_inference = perform_simple_inference
    model.forward_diffusion = forward_diffusion
    model.reverse_diffusion = reverse_diffusion

    # 回傳完整模型
    return model


def wavlm_large(pretrained=True, progress=True, device="cuda") -> WavLM:
    """
    載入原始論文釋出的 WavLM-Large 預訓練權重
    """
    # 如果沒有 GPU，強制切換到 CPU
    if torch.cuda.is_available() == False:
        if str(device) != "cpu":
            logging.warning(
                f"Overriding device {device} to cpu since no GPU is available."
            )
            device = "cpu"
    # 從網路下載 WavLM-Large 的權重檔案
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/RF5/transfusion-asr/releases/download/v1.0/WavLM-Large.pt",
        map_location=device,  # 直接載入到指定裝置（GPU 或 CPU）
        progress=progress,  # 是否顯示下載進度
    )

    # 解析設定檔，建立 WavLMConfig 物件（管理模型所有超參數）
    cfg = WavLMConfig(checkpoint["cfg"])
    # 轉換 device 物件型別（PyTorch 要求這樣寫才標準）
    device = torch.device(device)
    # 建立 WavLM 主體
    model = WavLM(cfg)
    # 如果指定載入預訓練權重
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    # 把模型搬到對應 device（GPU 或 CPU）
    model = model.to(device)
    # 切換為 eval 模式，停用 dropout/batchnorm 的訓練行為
    model.eval()
    # 給 model 增加 extract_transfusion_features 方法，方便直接萃取 transfusion 特徵
    model.extract_transfusion_features = extract_transfusion_features
    # 印出模型參數量
    print(
        f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters"
    )
    # 回傳已經準備好的模型
    return model
