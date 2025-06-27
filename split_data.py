import pandas as pd
from pathlib import Path
from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import argparse
import torch

THIS_DIR = Path(__file__).parent  # 取得目前檔案所在的資料夾


def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    # 建立 LibriSpeech 所有資料檔案的 DataFrame
    all_files = []
    folders = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]  # LibriSpeech 常見的七大子資料夾
    for f in folders:
        # 遍歷所有 flac 檔案路徑
        all_files.extend(list((root_path / f).rglob("**/*.flac")))
    # 取得 speaker ID（ls-加上說話者編號）
    speakers = ["ls-" + f.stem.split("-")[0] for f in all_files]
    # 取得每個檔案屬於哪個 subset
    subset = [f.parents[2].stem for f in all_files]
    # 建成一個 DataFrame，包含檔案路徑、speaker、subset
    df = pd.DataFrame({"path": all_files, "speaker": speakers, "subset": subset})
    return df


def get_transcriptions(df: pd.DataFrame) -> pd.DataFrame:
    # 根據 DataFrame 路徑，自動抓取每個音檔對應的文字標註（transcription）
    transcripts = {}  # 暫存已經讀過的 transcription
    out_transcripts = []  # 最後要寫進 DataFrame 的 transcription
    # 用 progress_bar 包起來，顯示進度條
    for i, row in progress_bar(df.iterrows(), total=len(df)):
        p = Path(row.path)  # 取得當前 row 的路徑物件
        if p.stem in transcripts:
            # 如果之前已經讀過這個音檔的標註，直接拿出來
            out_transcripts.append(transcripts[p.stem])
        else:
            # 找出當前資料夾下對應的 .trans.txt（存所有句子的 transcription）
            with open(
                p.parent / f"{p.parents[1].stem}-{p.parents[0].stem}.trans.txt", "r"
            ) as file:
                lines = file.readlines()
                for l in lines:
                    # 逐行解析，uttr_id = 音檔 stem，transcrip = 句子
                    uttr_id, transcrip = l.split(" ", maxsplit=1)
                    transcripts[uttr_id] = transcrip.strip()
            # 再把當前這個音檔的 transcription 加進 list
            out_transcripts.append(transcripts[p.stem])
    # 新增 transcription 欄位到 DataFrame
    df["transcription"] = out_transcripts
    return df


def get_wavlm_feat_paths(df: pd.DataFrame, ls_path, wavlm_path) -> pd.DataFrame:
    # 幫每個音檔路徑，找出對應預存的 WavLM 特徵檔（.pt）
    pb = progress_bar(df.iterrows(), total=len(df))
    targ_paths = []
    for i, row in pb:
        rel_path = Path(row.path).relative_to(
            ls_path
        )  # 計算音檔路徑相對於原始資料夾的路徑
        targ_path = (wavlm_path / rel_path).with_suffix(
            ".pt"
        )  # 換成目標資料夾、附檔名改 .pt
        assert targ_path.is_file()  # 檢查檔案真的存在
        targ_paths.append(targ_path)  # 加進 list
    df["wavlm_path"] = targ_paths  # 新增到 DataFrame
    return df


def get_vocab(df: pd.DataFrame, eps_idx=0):
    # 建立一個初始集合，包含 'eps' 這個特殊符號（通常用來代表空白或 padding）
    vocab = set(("eps",))
    # 逐行遍歷 DataFrame，顯示進度條
    for i, row in progress_bar(df.iterrows(), total=len(df)):
        # 把 transcription 欄位的內容轉成大寫，然後拆成單一字元 list
        chars = list(str(row.transcription).strip().upper())
        # 用 set union 把所有新出現的字元加入 vocab
        vocab |= set(chars)
    # 依照 Unicode 編碼順序排序（'eps' 固定排在最前面）
    vocab = sorted(list(vocab), key=lambda x: ord(x) if x != "eps" else -1)
    return vocab  # 回傳完整的字元表


def main():
    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(
        description="Generate train & valid csvs from dataset directories"
    )

    # 指定 LibriSpeech 資料集的根目錄
    parser.add_argument(
        "--librispeech_path",
        required=True,
        type=str,
        help="path to root of librispeech dataset",
    )
    # 指定對應的 WavLM 特徵檔案根目錄
    parser.add_argument(
        "--ls_wavlm_path",
        required=True,
        type=str,
        help="path to root of WavLM features extracted using extract.py",
    )
    # 可選參數，是否要包含 test 資料
    parser.add_argument(
        "--include_test",
        action="store_true",
        default=False,
        help="include processing and saving test.csv for test subsets",
    )

    # 解析參數
    args = parser.parse_args()

    # 產生包含所有語音檔案的 DataFrame
    if args.librispeech_path is not None:
        ls_df = make_librispeech_df(Path(args.librispeech_path))

    # 自動補上 transcription 欄位（對應文字標註）
    ls_df = get_transcriptions(ls_df)

    # 幫每個檔案找出對應的 WavLM 特徵檔案路徑
    ls_df = get_wavlm_feat_paths(
        ls_df, Path(args.librispeech_path), Path(args.ls_wavlm_path)
    )
    # 將原本的 path 欄位改名為 audio_path
    ls_df.rename(columns={"path": "audio_path"}, inplace=True)
    # 根據 subset 欄位篩選出訓練集（包含 "train" 字樣的行）
    train_csv = ls_df[ls_df.subset.str.contains("train")]
    # 篩選出驗證集（包含 "dev" 字樣的行）
    valid_csv = ls_df[ls_df.subset.str.contains("dev")]
    # 將訓練集與驗證集依照音檔路徑排序，讓 csv 順序一致
    train_csv = train_csv.sort_values("audio_path")
    valid_csv = valid_csv.sort_values("audio_path")

    # 建立 splits/ 資料夾存放輸出
    os.makedirs("splits/", exist_ok=True)
    # 存成 train.csv，欄位包含 audio_path、speaker、subset、transcription、wavlm_path
    train_csv.to_csv("splits/train.csv", index=False)
    valid_csv.to_csv("splits/valid.csv", index=False)
    print(
        f"Saved train csv (N={len(train_csv)}) and valid csv (N={len(valid_csv)} to splits/"
    )

    # 如果指定要包含 test 資料，也一併存下 test.csv
    if args.include_test:
        test_csv = ls_df[ls_df.subset.str.contains("test")]
        test_csv = test_csv.sort_values("audio_path")
        test_csv.to_csv("splits/test.csv", index=False)
        print(f"Saved test csv (N={len(test_csv)}) to splits/test.csv")

    # 產生 vocab 字元表，同時存成 i2s（index to string）、s2i（string to index）對照表
    vocab = get_vocab(ls_df)
    i2s = vocab
    s2i = {s: i for i, s in enumerate(vocab)}
    # 存成 splits/vocab.pt 方便後續模型載入
    torch.save({"i2s": i2s, "s2i": s2i}, "splits/vocab.pt")
    print("Vocab: ", s2i)


if __name__ == "__main__":
    main()
