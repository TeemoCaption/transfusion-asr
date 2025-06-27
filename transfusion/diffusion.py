"""
Discrete multinomial diffusion code adapted from https://github.com/ehoogeboom/multinomial_diffusion.

Please see the original repo (https://github.com/ehoogeboom/multinomial_diffusion) and paper for full
details on how multinomial diffusion works -- thanks to the original authors!
"""

import torch
from torch import Tensor
from torch.functional import F
import numpy as np


# -------------- Multinomial utility functions -----------

MIN_LOG_ARG = 1e-7  # originally was 1e-40


def log_1_min_a(a):
    return torch.log((1 - a.exp()).clamp_(min=1e-30))


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: Tensor, t, x_shape):
    """
    從一維參數（像 alpha、alpha_cumprod、beta）
    按照 batch t 指定的步數，抽出對應的數值，再 broadcast 成 batch 對齊 shape

    參數說明：
    - a: 一維參數表（例如 (timesteps,)）
    - t: 每個 batch 樣本要取哪一個 timestep，形狀通常是 (bs,)
    - x_shape: 目標 shape，像 (bs, ..., ...) 用來對齊 broadcast
    """
    b, *_ = t.shape  # batch size
    out = a.gather(-1, t)  # 針對每個 batch，從 a 裡面取出第 t[i] 個值
    return out.reshape(
        b, *((1,) * (len(x_shape) - 1))
    )  # reshape 成 (bs, 1, 1, ...)，方便 broadcast


def index_to_log_onehot(x, num_classes, dim=-1, dtype=torch.float32):
    """
    把 indices x（像是 label 或 token id）轉成 one-hot log 機率
    - 輸入 x 形狀 (bs, ...)
    - 輸出 log(onehot)，形狀 (bs, ..., num_classes)
    """
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x, num_classes)  # 先變成 one-hot (bs, ..., num_classes)
    if dim == 1:
        # 如果 one-hot 軸不是最後一個，要交換順序，符合 (bs, num_classes, ...)
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
    else:
        pass  # 通常 dim=-1 就不需要調整

    # 把 one-hot 轉成 log(onehot)，0 會變 -inf，所以要 clamp 到 MIN_LOG_ARG（防止 log(0) 爆炸）
    log_x = torch.log(
        x_onehot.to(dtype).clamp(min=MIN_LOG_ARG)
    )  # log(0) -> -30，避免 NaN

    return log_x


def sum_except_batch(x: Tensor, num_dims=1) -> Tensor:
    """
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


# -------------- Multinomial diffusion class -------------


class MultinomialDiffusion:
    def __init__(
        self,
        num_classes,
        timesteps=200,
        diffusion_s=0.008,
        loss_type="vb_stochastic",
        parametrization="x0",
        dtype=torch.float32,
        device="cpu",
    ):
        super(MultinomialDiffusion, self).__init__()
        # 只支援一種 loss 跟參數化方式
        assert loss_type in ("vb_stochastic",)
        assert parametrization in ("x0", "direct")

        self.num_classes = num_classes  # 類別數（例如詞彙表大小）
        self.loss_type = loss_type
        self.num_timesteps = timesteps  # 擴散步數
        self.parametrization = parametrization

        # ----- 計算 cosine schedule，用於決定每步加多少雜訊 -----
        alphas = self.cosine_beta_schedule(timesteps, diffusion_s)

        # 下面一堆數學計算都是為了 log-prob 快速查表
        alphas = alphas.to(torch.float64)
        log_alpha = alphas.log()  # log(alpha_t)
        log_cumprod_alpha = torch.cumsum(
            log_alpha, dim=-1
        )  # 累積相乘（其實是 log 累積相加）

        log_1_min_alpha = log_1_min_a(log_alpha)  # log(1-alpha_t)

        log_1_min_cumprod_alpha = log_1_min_a(
            log_cumprod_alpha
        )  # log(1 - \bar{alpha}_t)

        a = log_add_exp(log_alpha, log_1_min_alpha)  # log(1) = 0，檢查沒算錯

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item()
            < 1e-5
        )
        assert (
            torch.cumsum(log_alpha, dim=-1) - log_cumprod_alpha
        ).abs().sum().item() < 1.0e-5

        # 註冊 buffer，全部用 float32 轉 device
        self.log_alpha = log_alpha.to(dtype).to(device)
        self.log_1_min_alpha = log_1_min_alpha.to(dtype).to(device)
        self.log_cumprod_alpha = log_cumprod_alpha.to(dtype).to(device)
        self.log_1_min_cumprod_alpha = log_1_min_cumprod_alpha.to(dtype).to(device)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008) -> Tensor:
        """
        產生 cosine schedule，出處：https://arxiv.org/abs/2102.09672
        回傳的是 alpha（留存訊號比例），而不是 beta（加噪比例）
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = torch.clamp(alphas, 0.001, 1.0)
        return torch.sqrt(alphas)

    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor, dim=-1) -> Tensor:
        """計算兩個 categorical 分布的 KL divergence"""
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=dim)
        return kl

    def q_pred_one_timestep(self, log_x_t: Tensor, t: Tensor) -> Tensor:
        """
        計算 q(x_t | x_{t-1})：每步擴散「加雜訊」後的機率分布
        注意這裡直接用 x_t 來算（利用公式對稱性，見 appendix）
        """
        dt = log_x_t.dtype
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape).to(dt)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape).to(dt)

        # log_prob = log_sum_exp( log_x_t+log_alpha_t, log_1_min_alpha_t-log(num_classes) )
        log_probs = log_add_exp(
            log_x_t + log_alpha_t, log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred_one_timestep_scaled(
        self, log_x_t: Tensor, t: Tensor, c: int, jump_len: int
    ) -> Tensor:
        """
        跟上面一樣，只是針對推論時要做 repaint (repainting) 做進階「加雜訊」強度調整
        """
        dt = log_x_t.dtype
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape).to(dt)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape).to(dt)

        # Magic！這裡用 sigmoid 調控噪音強度分布（細節見 TransFusion 論文或原始碼註解）
        xax = torch.arange(0, log_x_t.shape[1], 1).to(log_x_t.device)
        aa = log_x_t.shape[1] * (c / jump_len)
        sig = 1 / (1 + torch.exp(-(xax - aa + 20) / 8))
        log_alpha_t = (torch.log(1 / sig)[None, :, None] + log_alpha_t).clamp(
            -torch.inf, 0
        )
        log_1_min_alpha_t = torch.log(sig)[None, :, None] + log_1_min_alpha_t

        log_probs = log_add_exp(
            log_x_t + log_alpha_t, log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred(self, log_x_start: Tensor, t) -> Tensor:
        """
        計算 q(x_t | x_0)：從 x_0 經過 t 步加雜訊後的機率分布
        """
        dt = log_x_start.dtype
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape).to(
            dt
        )
        log_1_min_cumprod_alpha = extract(
            self.log_1_min_cumprod_alpha, t, log_x_start.shape
        ).to(dt)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes),
        )
        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        """
        計算 q(x_{t-1} | x_t, x_0)：
        根據 Bayesian 公式把 x_t、x_0 綁在一起還原 x_{t-1} 機率分布
        """
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)  # q(x_{t-1} | x_0)
        # 如果 t==0 就直接用 x_0，不要亂算
        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # Numerator: log q(x_{t-1} | x_0) + q(x_t | x_{t-1})（因為有 log domain，直接加）
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        # Denominator: sum exp, 保證是合法分布
        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - torch.logsumexp(
            unnormed_logprobs, dim=-1, keepdim=True
        )
        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x_t, t, log_x0_pred):
        """
        反向還原：p(x_{t-1}|x_t)，用模型預測的 x_0 來估計最可能的 x_{t-1}
        """
        log_model_pred = self.q_posterior(log_x_start=log_x0_pred, log_x_t=log_x_t, t=t)
        return log_model_pred

    def log_sample_categorical(self, logprobs: Tensor, dim=-1) -> Tensor:
        """
        根據 log 機率從多分類分布抽樣（gumbel-max trick！）
        回傳 (batch, ...) 的抽樣 index
        """
        uniform = torch.rand_like(logprobs)
        gumbel_noise = -torch.log(
            (-torch.log(uniform.clamp_(min=MIN_LOG_ARG))).clamp_(min=MIN_LOG_ARG)
        )
        sample = (gumbel_noise + logprobs).argmax(dim=dim)
        return sample

    def q_sample(self, log_x_start, t):
        """
        直接根據 q(x_t|x_0) 抽一個 x_t，等同於 forward 多步擴散的結果
        """
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        sample = self.log_sample_categorical(log_EV_qxt_x0)
        # log_sample = index_to_log_onehot(sample, self.num_classes)
        return sample  # log_sample

    def compute_Lt(
        self,
        log_x_start: Tensor,
        log_x_t: Tensor,
        log_x0_pred: Tensor,
        t,
        detach_mean=False,
        include_kl_prior=True,
    ):
        """
        計算損失（ELBO 下界）：
        - t=0 時是 negative log-likelihood
        - t>0 時是 KL divergence
        - 可以選擇要不要加 prior loss
        """
        dtype = log_x_start.dtype
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x_t=log_x_t, t=t, log_x0_pred=log_x0_pred)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        # t=0 時負對數似然
        decoder_nll = -(log_x_start.exp() * log_model_prob).sum(dim=-1)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).to(dtype)
        loss = mask * decoder_nll + (1.0 - mask) * kl

        if include_kl_prior:
            pt = torch.ones_like(t, dtype=dtype)
            kl_prior = self.kl_prior(log_x_start)
            loss = (kl) + kl_prior

        return loss

    def kl_prior(self, log_x_start: Tensor) -> Tensor:
        """
        計算 KL(q(x_T|x_0) || uniform)，這是 ELBO 的正則化項
        """
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device, dtype=torch.long)

        log_qxT_prob = self.q_pred(
            log_x_start, t=(self.num_timesteps - 1) * ones
        )  # q(x_T | x_0)
        log_half_prob = -torch.log(
            self.num_classes * torch.ones_like(log_qxT_prob)
        )  # log(1/K)
        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)


def index2logit(x: Tensor, vocab_size: int, dtype=torch.float32):
    """
    把 index label 轉成 centered one-hot（用於 multinomial 擴散）
    """
    x = F.one_hot(x, num_classes=vocab_size).to(dtype)
    x = x * (vocab_size / (vocab_size - 1)) - 1 / (vocab_size - 1)
    return x
