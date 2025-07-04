import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastprogress.fastprogress import master_bar, progress_bar
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader

from transfusion.config import ModelConfig
from transfusion.dataset import PogDataset, collate_batch
from transfusion.diffusion import MultinomialDiffusion, index_to_log_onehot
from transfusion.eval import eval_multinomial_cer
from transfusion.model import TransFusion


@dataclass
class DistributedConfig:
    dist_backend: str = (
        "nccl"  # 分散式訓練時用的後端通訊協定（通常 GPU 用 'nccl'，CPU 可以用 'gloo'）
    )
    dist_url: str = (
        "tcp://localhost:54321"  # 分散式初始化的 URL（定義各節點怎麼找到彼此）
    )
    # n_nodes: int = 1 # 這個參數交給 deepspeed 管理，不在這裡設定
    n_gpus_per_node: int = 1  # 每個節點使用幾張 GPU（單機單卡或多卡）


@dataclass
class TrainConfig:
    # 分散式訓練設定
    distributed: DistributedConfig = DistributedConfig()
    # 模型設定（裡面包含架構、參數等）
    model_cfg: ModelConfig = ModelConfig

    device: str = "cuda"  # 預設使用 GPU 訓練
    seed: int = 1775  # 隨機種子（確保結果可重現）

    batch_size: int = 8  # 一個 batch 幾筆資料
    num_workers: int = 16  # DataLoader 載入時的子執行緒數量
    # fp16: bool = False # 是否啟用半精度訓練（這裡交給 deepspeed 管理）

    summary_interval: int = 25  # 幾個 step 計算 summary（例如 tensorboard 可視化）
    checkpoint_interval: int = 5000  # 幾個 step 存一次模型 checkpoint
    stdout_interval: int = 100  # 幾個 step 印一次訓練資訊到終端機
    validation_interval: int = 5000  # 幾個 step 跑一次驗證

    # 學習率/訓練上限等（多數交給 deepspeed yaml 控制）
    max_steps: int = 100_000_000  # 最多訓練多少 step

    # 資料相關設定
    checkpoint_path: str = MISSING  # checkpoint 存放路徑
    train_csv: str = MISSING  # 訓練集 csv 路徑
    valid_csv: str = MISSING  # 驗證集 csv 路徑
    valid_n_cer_eval: int = 250  # 驗證時每次抽幾筆計算 CER（字元錯誤率）
    vocab_path: str = MISSING  # 字彙表（vocab）檔案路徑
    resume_checkpoint: str = ""  # 是否從舊的 checkpoint 繼續訓練
    sample_rate: int = 16000  # 音檔取樣率
    seq_len: int = 16000  # 輸入音訊最大長度（單位：取樣點）
    max_transcript_length: int = (
        300  # 超過這個長度的 transcript 只佔 0.4%（預設最大轉錄長度）
    )


def train(rank, cfg: TrainConfig, deepspeed_cfg: argparse.Namespace):
    print(f"[RANK {rank}] Deepspeed cfg: {deepspeed_cfg}")

    # -------------------
    # 初始化分散式訓練
    if cfg.distributed.n_gpus_per_node > 1:
        # 若一台機器用多張 GPU，初始化分散式後端（通常用 NCCL）
        deepspeed.init_distributed(
            backend=cfg.distributed.dist_backend,  # 使用 NCCL/GLOO 等協定
            init_method=cfg.distributed.dist_url,  # 分散式啟動連線方式
        )

    device = torch.device(f"cuda:{rank:d}")  # 設定當前 process 對應的 GPU

    # --------------------
    # 定義模型與損失函式
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(
        device
    )  # 建立 TransFusion 模型並搬到對應 GPU

    # 檢查 diffusion 類型，這裡只支援 multinomial
    if cfg.model_cfg.diffusion_type != "multinomial":
        raise NotImplementedError()

    if rank == 0:
        # 只有主進程才顯示 log 與建立目錄（避免重複）
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Model initialized as:\n {model}")
        logging.info(f"checkpoints directory : {cfg.checkpoint_path}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
    print(
        f"[RANK {rank}] Model has {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )

    # ------------------------
    # 載入訓練與驗證集
    vocab = torch.load(cfg.vocab_path)  # 載入 vocab（字元對照表）

    train_df = pd.read_csv(cfg.train_csv)  # 載入訓練資料的 csv
    valid_df = pd.read_csv(cfg.valid_csv)  # 載入驗證資料的 csv
    # 建立 Dataset 物件，負責資料處理與編碼
    train_ds = PogDataset(
        train_df, vocab["s2i"], vocab["i2s"], cfg.model_cfg.T, cfg.max_transcript_length
    )
    valid_ds = PogDataset(
        valid_df, vocab["s2i"], vocab["i2s"], cfg.model_cfg.T, cfg.max_transcript_length
    )

    # ------------------------
    # 初始化 deepspeed（訓練包裝器，同時建立 optimizer、dataloader、scheduler）
    model_engine, optim, train_dl, scheduler = deepspeed.initialize(
        args=deepspeed_cfg,  # deepspeed 相關設定
        model=model,  # 模型主體
        model_parameters=model.parameters(),
        training_data=train_ds,  # 訓練資料集
        collate_fn=collate_batch,  # 資料 batch 整合方式
    )
    # 修正 deepspeed 產生的 train_dl 的長度（某些情況下 deepspeed 的 dataloader 長度會有 bug）
    real_train_len = len(train_dl.data_sampler) // train_dl.batch_size
    train_dl.len = real_train_len

    try:
        # 取得 tensorboard 的 log 目錄，主流寫法
        ds_log_path = (
            Path(model_engine.tensorboard_output_path())
            / model_engine.tensorboard_job_name()
        )
    except Exception as e:
        # 若舊版 deepspeed 則用替代寫法
        ds_log_path = (
            Path(model_engine.monitor.tb_monitor.output_path)
            / model_engine.monitor.tb_monitor.job_name
        )

    # 如果有指定從某個 checkpoint 繼續訓練
    if cfg.resume_checkpoint != "":
        _, client_sd = model_engine.load_checkpoint(
            cfg.checkpoint_path, cfg.resume_checkpoint
        )
        steps = client_sd["steps"] + 1  # 取出已經訓練到的步數
        last_epoch = client_sd["last_epoch"]  # 取出上次 epoch
    else:
        steps = 0  # 從頭開始訓練
        last_epoch = 0
        client_sd = {}

    fp16 = model_engine.fp16_enabled()  # 是否啟用半精度運算（加速、省顯存）

    # --------------------------
    # Set up diffusion manager

    if cfg.model_cfg.diffusion_type == "multinomial":
        diffuser = MultinomialDiffusion(
            cfg.model_cfg.vocab_size,
            cfg.model_cfg.T,
            cfg.model_cfg.diffusion_s,
            dtype=torch.float16 if fp16 else torch.float32,
            device=device,
        )
        loss_fn = torch.nn.SmoothL1Loss().to(device)
    else:
        raise NotImplementedError()

    # --------------------------
    # Logging init
    max_epochs = math.ceil(cfg.max_steps / len(train_dl))
    print(f"[RANK {rank}] deepspeed fp16={fp16} | max epochs: {max_epochs}")

    if rank == 0:
        print(f"[RANK {rank}] deepspeed logging to {ds_log_path}")
        try:
            sw = model_engine.get_summary_writer()
        except Exception as e:
            sw = model_engine.monitor.tb_monitor.summary_writer

        mb = master_bar(range(max(0, last_epoch), max_epochs))
        sw.add_text(
            "config", "```\n" + OmegaConf.to_yaml(cfg) + "\n```", global_step=steps
        )
        smooth_loss = None
        valid_dl = DataLoader(
            valid_ds,
            cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=cfg.num_workers,
        )
    else:
        mb = range(max(0, last_epoch), max_epochs)

    # --------------------------
    # Training loop
    model_engine.train()

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write("Epoch: {}".format(epoch + 1))
            pb = progress_bar(enumerate(train_dl), total=len(train_dl), parent=mb)
        else:
            pb = enumerate(train_dl)

        if steps > cfg.max_steps:
            break

        for i, batch in pb:
            # -----------------------
            #  Read batch
            if rank == 0:
                start_b = time.time()
            x, t, cond_emb, x_padding_mask, cond_padding_mask = batch
            x = x.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)  # (bs, seq_len)
            cond_emb = cond_emb.to(device, non_blocking=True)
            x_padding_mask = x_padding_mask.to(device, non_blocking=True)
            cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)

            if fp16:
                dtype = torch.float16
                cond_emb = cond_emb.half()
            else:
                dtype = torch.float32

            # -----------------------
            # Perform diffusion perturbation

            if cfg.model_cfg.diffusion_type == "multinomial":
                log_x_0 = index_to_log_onehot(x, cfg.model_cfg.vocab_size, dtype=dtype)
                x_t = diffuser.q_sample(log_x_0, t)
                log_x_t = index_to_log_onehot(
                    x_t, cfg.model_cfg.vocab_size, dtype=dtype
                )

                x_0_pred = model_engine(
                    x_t, t, cond_emb, cond_padding_mask, x_padding_mask
                )  # (bs, seq_len, vocab size)
                log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)

                loss = diffuser.compute_Lt(log_x_0, log_x_t, log_x_0_pred, t)
                loss = loss.mean(dim=0)
            else:
                raise NotImplementedError()

            # Backwards
            model_engine.backward(loss)
            if steps % cfg.summary_interval == 0:
                gnorm = model_engine.get_global_grad_norm()
            model_engine.step()

            # checkpointing
            if steps % cfg.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{cfg.checkpoint_path}/ckpt_{steps:08d}.pt"

                client_sd["steps"] = steps
                client_sd["last_epoch"] = epoch
                client_sd["cfg_yaml"] = OmegaConf.to_yaml(cfg)

                model_engine.save_checkpoint(
                    cfg.checkpoint_path,
                    Path(checkpoint_path).stem,
                    client_state=client_sd,
                )

                print(f"[RANK {rank}] Saved checkpoint to {checkpoint_path}")

            # ----------------------
            # Validation & logging
            if rank == 0:
                if smooth_loss is None:
                    smooth_loss = float(loss.item())
                else:
                    smooth_loss = smooth_loss + 0.1 * (float(loss.item()) - smooth_loss)
                # STDOUT logging
                if steps % cfg.stdout_interval == 0:
                    mb.write(
                        "steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB".format(
                            steps,
                            loss.item(),
                            time.time() - start_b,
                            torch.cuda.max_memory_allocated() / 1e9,
                        )
                    )
                if steps % (cfg.stdout_interval // 5) == 0:
                    mb.child.comment = (
                        "steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}".format(
                            steps, loss.item(), time.time() - start_b
                        )
                    )

                # Tensorboard summary logging
                if steps % cfg.summary_interval == 0:
                    sw.add_scalar("training/loss_smooth", smooth_loss, steps)
                    sw.add_scalar("training/loss_raw", loss.item(), steps)
                    sw.add_scalar("opt/lr", float(optim.param_groups[0]["lr"]), steps)
                    if gnorm is not None:
                        sw.add_scalar("opt/grad_norm", float(gnorm), steps)

                # Validation
                if steps % cfg.validation_interval == 0 and steps != 0:
                    model_engine.eval()
                    loss_fn.eval()

                    val_err_tot = 0
                    cers = []
                    cer_noepses = []
                    with torch.no_grad():
                        for j, batch in progress_bar(
                            enumerate(valid_dl), total=len(valid_dl), parent=mb
                        ):
                            x, t, cond_emb, x_padding_mask, cond_padding_mask = batch
                            x = x.to(device, non_blocking=True)
                            t = t.to(device, non_blocking=True)  # (bs, seq_len)
                            cond_emb = cond_emb.to(device, non_blocking=True)
                            x_padding_mask = x_padding_mask.to(
                                device, non_blocking=True
                            )
                            cond_padding_mask = cond_padding_mask.to(
                                device, non_blocking=True
                            )

                            if fp16:
                                cond_emb = cond_emb.half()

                            # Perform diffusion perturbation
                            if cfg.model_cfg.diffusion_type == "multinomial":
                                log_x_0 = index_to_log_onehot(
                                    x, cfg.model_cfg.vocab_size, dtype=dtype
                                )
                                x_t = diffuser.q_sample(log_x_0, t)
                                log_x_t = index_to_log_onehot(
                                    x_t, cfg.model_cfg.vocab_size, dtype=dtype
                                )

                                x_0_pred = model_engine(
                                    x_t, t, cond_emb, cond_padding_mask, x_padding_mask
                                )  # (bs, seq_len, vocab size)
                                log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)

                                loss = diffuser.compute_Lt(
                                    log_x_0, log_x_t, log_x_0_pred, t
                                )
                                loss = loss.mean(dim=0)
                            else:
                                raise NotImplementedError()

                            val_err_tot += loss

                            # skip CERs for most validation batches -- takes too long
                            if j * cfg.batch_size > cfg.valid_n_cer_eval:
                                continue

                            if cfg.model_cfg.diffusion_type == "multinomial":
                                cer, cer_noeps = eval_multinomial_cer(
                                    steps,
                                    j,
                                    (x, t, cond_emb, cond_padding_mask, x_padding_mask),
                                    model_engine,
                                    diffuser,
                                    cfg.model_cfg.T,
                                    True if j < 4 else False,
                                    sw,
                                    vocab["i2s"],
                                )
                                cer_noepses.append(cer_noeps)
                            else:
                                raise NotImplementedError()
                            cers.append(cer)

                        cers = torch.concat(cers, dim=0)
                        mean_cer = cers.mean()
                        std_cer = cers.std()
                        sw.add_scalar("validation/cer_mean", mean_cer, steps)
                        sw.add_scalar("validation/cer_std", std_cer, steps)
                        sw.add_histogram("validation/cers", cers, steps)
                        if cfg.model_cfg.diffusion_type == "multinomial":
                            cer_noepses = torch.concat(cer_noepses, dim=0)
                            mean_cer_noeps = cer_noepses.mean()
                            std_cer_noeps = cer_noepses.std()
                            sw.add_scalar(
                                "validation/cer_noeps_mean", mean_cer_noeps, steps
                            )
                            sw.add_scalar(
                                "validation/cer_noeps_std", std_cer_noeps, steps
                            )
                            sw.add_histogram(
                                "validation/cers_noeps", cer_noepses, steps
                            )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/loss", val_err, steps)
                        mb.write(
                            f"validation run complete at {steps:,d} steps. validation loss: {val_err:5.4f}"
                        )

                    model_engine.train()
                    loss_fn.train()
                    sw.add_scalar(
                        "memory/max_allocated_gb",
                        torch.cuda.max_memory_allocated() / 1e9,
                        steps,
                    )
                    sw.add_scalar(
                        "memory/max_reserved_gb",
                        torch.cuda.max_memory_reserved() / 1e9,
                        steps,
                    )
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

            steps += 1
            if steps > cfg.max_steps:
                print(f"[RANK {rank}] FINISHED TRAINING")
                break

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )
    print("Training completed!")


def main():
    print("Initializing Training Process..")
    logging.getLogger().setLevel(logging.INFO)

    # Setup CLI args
    parser = argparse.ArgumentParser(
        usage="\n"
        + "-" * 10
        + " Default config "
        + "-" * 10
        + "\n"
        + str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig)))
    )

    deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    # Parse args
    a, _ = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()

    # We must remove any config arguments deepspeed injected,
    # otherwise we will have duplicate deepspeed keys in `override_cfg`
    # and cli args `a`.
    keys_to_drop = []
    for key in override_cfg:
        if key.startswith("--"):
            keys_to_drop.append(key)
    for key in keys_to_drop:
        delattr(override_cfg, key)

    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)
    logging.info(f"Running with config:\n {OmegaConf.to_yaml(cfg)}")

    # Set seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Launch training
    train(a.local_rank, cfg, a)


if __name__ == "__main__":
    main()
