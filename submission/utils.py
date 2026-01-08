import os
import sys
import glob
import gc
import warnings
from contextlib import contextmanager
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        def __call__(self, *args, **kwargs):
            try:
                tqdm_object.update(n=self.batch_size)
            except Exception:
                pass
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        try:
            tqdm_object.close()
        except Exception:
            pass

def read_train_metadata(train_dir: str) -> pd.DataFrame:
    """
    Read train metadata.csv. Must include repertoire_id, filename, label_positive.
    """
    meta_path = os.path.join(train_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.csv not found in {train_dir}")
    meta = pd.read_csv(meta_path)
    needed = {"repertoire_id", "filename", "label_positive"}
    if not needed.issubset(meta.columns):
        raise ValueError(f"metadata.csv missing required columns {needed}")

    meta["repertoire_id"] = meta["repertoire_id"].astype(str)
    meta["filename"] = meta["filename"].astype(str)
    meta["label_positive"] = meta["label_positive"].astype(int)
    return meta


def iter_repertoires_from_metadata(train_dir: str, meta: pd.DataFrame, ids: Optional[np.ndarray] = None):
    """
    Yields (rep_id, file_path, label_bool) for reps in ids (or all if ids None).
    """
    if ids is None:
        sub = meta
    else:
        ids_set = set(ids.tolist())
        sub = meta[meta["repertoire_id"].isin(ids_set)]

    for row in sub.itertuples(index=False):
        rep_id = row.repertoire_id
        file_path = os.path.join(train_dir, row.filename)
        yield rep_id, file_path, bool(row.label_positive)


def iter_repertoires_no_metadata(data_dir: str):
    """
    Yields (rep_id, file_path) for test dirs without metadata.
    """
    for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
        rep_id = os.path.basename(file_path).replace(".tsv", "")
        yield rep_id, file_path


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def normalize_sequence_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure sequence rows (label_positive_probability sentinel -999) have deterministic IDs.
    """
    required = {"ID", "dataset", "label_positive_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"submissions.csv missing required columns: {sorted(missing)}")

    lp = pd.to_numeric(df["label_positive_probability"], errors="coerce")
    seq_mask = lp <= -998

    if seq_mask.any():
        ranks = df.loc[seq_mask].groupby("dataset", sort=False).cumcount() + 1
        df.loc[seq_mask, "ID"] = (
            df.loc[seq_mask, "dataset"].astype(str)
            + "_seq_top_"
            + ranks.astype(str)
        )

    return df


def concatenate_output_files(out_dir: str) -> None:
    predictions_pattern = os.path.join(out_dir, "*_test_predictions.tsv")
    sequences_pattern = os.path.join(out_dir, "*_important_sequences.tsv")

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []
    for pred_file in predictions_files:
        try:
            df_list.append(pd.read_csv(pred_file, sep="\t"))
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
    for seq_file in sequences_files:
        try:
            df_list.append(pd.read_csv(seq_file, sep="\t"))
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")

    if not df_list:
        concatenated_df = pd.DataFrame(
            columns=["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
        )
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)

    concatenated_df = normalize_sequence_ids(concatenated_df)
    submissions_file = os.path.join(out_dir, "submissions.csv")
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))
    return pairs

def load_and_encode_kmers(data_dir: str, n_jobs: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gapped masks only (NO contiguous 4-mer). Returns (features_df, metadata_df)
    """
    MASKS_CONFIG = [
        (5, (0, 2, 3, 4), "{}.{}{}{}"),
        (5, (0, 1, 3, 4), "{}{}.{}{}"),
        (5, (0, 1, 2, 4), "{}{}{}.{}"),
        (6, (0, 1, 4, 5), "{}{}..{}{}"),
        (4, (0, 2, 3), "{}.{}{}"),
        (4, (0, 1, 3), "{}{}.{}"),
        (5, (0, 2, 4), "{}.{}.{}"),
        (5, (0, 3, 4), "{}..{}{}"),
        (5, (0, 1, 4), "{}{}..{}"),
        (6, (0, 4, 5), "{}...{}{}"),
        (6, (0, 1, 5), "{}{}...{}"),
    ]

    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_metadata = os.path.exists(metadata_path)
    tasks = []

    if has_metadata:
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            tasks.append((os.path.join(data_dir, row.filename), str(row.repertoire_id), row.label_positive))
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            tasks.append((file_path, rep_id, None))

    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def _encode_one(file_path, rep_id, label):
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa"])
        except Exception:
            return None

        kmer_counts = Counter()
        total = 0
        for seq in df["junction_aa"].dropna():
            if not isinstance(seq, str):
                continue
            L = len(seq)
            for (k_len, indices, fmt_str) in MASKS_CONFIG:
                if L < k_len:
                    continue
                for i in range(L - k_len + 1):
                    residues = [seq[i + idx] for idx in indices]
                    feat = fmt_str.format(*residues)
                    kmer_counts[feat] += 1
                    total += 1

        freqs = {k: v / total for k, v in kmer_counts.items()} if total > 0 else {}
        return ({"ID": rep_id, **freqs}, {"ID": rep_id, "label_positive": label})

    results = Parallel(n_jobs=n_jobs)(delayed(_encode_one)(fp, rid, lbl) for (fp, rid, lbl) in tasks)

    feat_rows, meta_rows = [], []
    for r in results:
        if r is None:
            continue
        feat_rows.append(r[0])
        meta_rows.append(r[1])

    if not feat_rows:
        return pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(feat_rows).fillna(0).set_index("ID")

    # fudge rare
    nnz = (X != 0).sum(axis=0)
    X = X.loc[:, nnz[nnz >= 2].index]

    if X.shape[1] == 0:
        meta = pd.DataFrame(meta_rows)
        return pd.DataFrame(index=X.index), meta

    X[:] = normalize(X.values, norm="l2")
    X = X.astype(np.float32)

    meta = pd.DataFrame(meta_rows)
    return X, meta


def load_and_encode_kmers_ligo(data_dir: str, k: int = 4, n_jobs: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    LIgO: contiguous k-mers + simple gaps (X.XX, XX.X)
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_metadata = os.path.exists(metadata_path)
    tasks = []

    if has_metadata:
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            tasks.append((os.path.join(data_dir, row.filename), str(row.repertoire_id), row.label_positive))
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            tasks.append((file_path, rep_id, None))

    if not tasks:
        return pd.DataFrame(), pd.DataFrame()

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def _encode_one(file_path, rep_id, label):
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa"])
        except Exception:
            return None

        counts = Counter()
        total = 0
        for seq in df["junction_aa"].dropna():
            if not isinstance(seq, str):
                continue
            L = len(seq)
            if L < k:
                continue
            for i in range(L - k + 1):
                km = seq[i:i + k]
                counts[km] += 1
                total += 1

                if i + 4 <= L:
                    counts[f"{seq[i]}.{seq[i+2]}{seq[i+3]}"] += 1
                    total += 1
                    counts[f"{seq[i]}{seq[i+1]}.{seq[i+3]}"] += 1
                    total += 1

        freqs = {f: c / total for f, c in counts.items()} if total > 0 else {}
        feat = {"ID": rep_id, **freqs}
        meta = {"ID": rep_id}
        if label is not None:
            meta["label_positive"] = label
        return feat, meta

    results = Parallel(n_jobs=n_jobs)(delayed(_encode_one)(fp, rid, lbl) for (fp, rid, lbl) in tasks)

    feat_rows, meta_rows = [], []
    for r in results:
        if r is None:
            continue
        feat_rows.append(r[0])
        meta_rows.append(r[1])

    if not feat_rows:
        return pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(feat_rows).fillna(0).set_index("ID")
    nnz = (X != 0).sum(axis=0)
    X = X.loc[:, nnz[nnz >= 2].index]
    if X.shape[1] == 0:
        meta = pd.DataFrame(meta_rows)
        return pd.DataFrame(index=X.index), meta
    X[:] = normalize(X.values, norm="l2")
    X = X.astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, meta


def load_and_encode_kmers_trunc(
    data_dir: str,
    k: int = 4,
    n_jobs: int = 8,
    motif_crop: int = 4,
    min_repertoires: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CROPPED k-mers + left/right halves ONLY.
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_metadata = os.path.exists(metadata_path)

    tasks: list[tuple[str, str, Optional[bool]]] = []
    if has_metadata:
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            tasks.append((os.path.join(data_dir, row.filename), str(row.repertoire_id), row.label_positive))
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            tasks.append((file_path, rep_id, None))

    if not tasks:
        return (
            pd.DataFrame().set_index(pd.Index([], name="ID")),
            pd.DataFrame(columns=["ID", "label_positive"]),
        )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    def crop_seq(s: str, trim: int) -> str:
        if not isinstance(s, str):
            return ""
        if len(s) <= 2 * trim:
            return ""
        return s[trim:-trim]

    def _encode_one(file_path: str, rep_id: str, label, k: int):
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa"])
        except Exception:
            return None

        df = df.dropna(subset=["junction_aa"])
        df = df.drop_duplicates(subset=["junction_aa"])

        if df.empty:
            feat = {"ID": rep_id}
            meta = {"ID": rep_id}
            if label is not None:
                meta["label_positive"] = label
            return feat, meta

        kmer_counts = Counter()
        left_counts = Counter()
        right_counts = Counter()
        total_k = 0
        left_total = 0
        right_total = 0

        for raw in df["junction_aa"]:
            seq = crop_seq(raw, motif_crop)
            L = len(seq)
            if L < k:
                continue

            n_windows = L - k + 1
            mid = n_windows / 2.0

            for i in range(n_windows):
                km = seq[i:i + k]
                kmer_counts[km] += 1
                total_k += 1
                if i < mid:
                    left_counts[km] += 1
                    left_total += 1
                else:
                    right_counts[km] += 1
                    right_total += 1

        if total_k > 0:
            kmer_freqs = {km: c / total_k for km, c in kmer_counts.items()}
        else:
            kmer_freqs = {}

        if left_total > 0:
            left_freqs = {f"L_{km}": c / left_total for km, c in left_counts.items()}
        else:
            left_freqs = {}

        if right_total > 0:
            right_freqs = {f"R_{km}": c / right_total for km, c in right_counts.items()}
        else:
            right_freqs = {}

        feat = {"ID": rep_id, **kmer_freqs, **left_freqs, **right_freqs}

        meta = {"ID": rep_id}
        if label is not None:
            meta["label_positive"] = label
        return feat, meta

    with tqdm_joblib(tqdm(total=len(tasks), desc="Encoding repertoires (trunc)")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_encode_one)(file_path, rep_id, label, k)
            for (file_path, rep_id, label) in tasks
        )

    feat_rows, meta_rows = [], []
    for r in results:
        if r is None:
            continue
        feat_rows.append(r[0])
        meta_rows.append(r[1])

    if not feat_rows:
        return (
            pd.DataFrame().set_index(pd.Index([], name="ID")),
            pd.DataFrame(columns=["ID", "label_positive"]),
        )

    X = pd.DataFrame(feat_rows).fillna(0.0).set_index("ID")

    nnz = (X != 0).sum(axis=0)
    keep = nnz[nnz >= min_repertoires].index
    X = X.loc[:, keep]

    if X.shape[1] == 0:
        meta = pd.DataFrame(meta_rows)
        return pd.DataFrame(index=X.index), meta

    X[:] = normalize(X.values, norm="l2")
    X = X.astype(np.float32)

    meta = pd.DataFrame(meta_rows)
    return X, meta


def load_and_encode_kmers_posi(data_dir: str, k: int = 4, n_jobs: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Positional S/M/E k-mers + length + V/J + VJ
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    has_metadata = os.path.exists(metadata_path)
    tasks = []

    if has_metadata:
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            tasks.append((os.path.join(data_dir, row.filename), str(row.repertoire_id), row.label_positive))
    else:
        for file_path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            rep_id = os.path.basename(file_path).replace(".tsv", "")
            tasks.append((file_path, rep_id, None))

    if not tasks:
        return (
            pd.DataFrame().set_index(pd.Index([], name="ID")),
            pd.DataFrame(columns=["ID", "label_positive"]),
        )

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    LENGTH_MIN, LENGTH_MAX = 6, 32
    LENGTH_BINS = list(range(LENGTH_MIN, LENGTH_MAX + 1))

    def _encode_one(file_path: str, rep_id: str, label, k: int):
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
        except Exception:
            return None

        df = df.dropna(subset=["junction_aa"])
        if df.empty:
            feat = {"ID": rep_id}
            meta = {"ID": rep_id}
            if label is not None:
                meta["label_positive"] = label
            return feat, meta

        df["L"] = df["junction_aa"].str.len()
        n_seqs = len(df)

        pos_counts = Counter()
        total_k = 0
        for seq in df["junction_aa"]:
            L = len(seq)
            if L < k:
                continue
            denom = (L - 1) if L > 1 else 1
            for i in range(L - k + 1):
                km = seq[i:i + k]
                total_k += 1
                center = i + (k - 1) / 2.0
                rel = center / denom
                if rel < 1 / 3:
                    bucket = "S"
                elif rel < 2 / 3:
                    bucket = "M"
                else:
                    bucket = "E"
                pos_counts[f"{bucket}_{km}"] += 1

        pos_freqs = {f: c / total_k for f, c in pos_counts.items()} if total_k > 0 else {}

        length_counts = df["L"].value_counts()
        length_feats = {f"len_{Lval}_frac": float(length_counts.get(Lval, 0)) / n_seqs for Lval in LENGTH_BINS}
        length_feats["len_mean"] = float(df["L"].mean())
        length_feats["len_std"] = float(df["L"].std(ddof=0)) if n_seqs > 1 else 0.0

        v_counts = df["v_call"].value_counts(dropna=False)
        j_counts = df["j_call"].value_counts(dropna=False)
        vj_counts = df.groupby(["v_call", "j_call"]).size()

        vj_feats = {}
        for (v, j), cnt in vj_counts.items():
            vj_feats[f"VJ_{str(v)}__{str(j)}_frac"] = float(cnt) / n_seqs
        for v, cnt in v_counts.items():
            vj_feats[f"V_{str(v)}_frac"] = float(cnt) / n_seqs
        for j, cnt in j_counts.items():
            vj_feats[f"J_{str(j)}_frac"] = float(cnt) / n_seqs

        feat = {"ID": rep_id, **pos_freqs, **length_feats, **vj_feats}
        meta = {"ID": rep_id}
        if label is not None:
            meta["label_positive"] = label
        return feat, meta

    with tqdm_joblib(tqdm(total=len(tasks), desc="Encoding repertoires (posi)")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_encode_one)(file_path, rep_id, label, k)
            for (file_path, rep_id, label) in tasks
        )

    feat_rows, meta_rows = [], []
    for r in results:
        if r is None:
            continue
        feat_rows.append(r[0])
        meta_rows.append(r[1])

    if not feat_rows:
        return pd.DataFrame(), pd.DataFrame()

    X = pd.DataFrame(feat_rows).fillna(0).set_index("ID")
    nnz = (X != 0).sum(axis=0)
    X = X.loc[:, nnz[nnz >= 2].index]
    if X.shape[1] == 0:
        meta = pd.DataFrame(meta_rows)
        return pd.DataFrame(index=X.index), meta
    X[:] = normalize(X.values, norm="l2")
    X = X.astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, meta

def build_label_aware_seq_library_streaming(
    train_dir: str,
    meta: pd.DataFrame,
    ids: np.ndarray,
    y_series: pd.Series,
    min_total_reps: int = 2,
    max_library_size: int = 10000,
    alpha: float = 0.5,
    log_odds_min: float = 0.0,
) -> pd.DataFrame:
    """
    Streaming builder:
      - iterates repertoire files in `ids`
      - counts in how many pos/neg repertoires each (junction_aa,v_call,j_call) appears
      - returns seq_lib with columns: junction_aa, v_call, j_call, pos_reps, neg_reps, total_reps, log_odds, seq_key
    """
    ids_set = set(ids.tolist())
    sub = meta[meta["repertoire_id"].isin(ids_set)]
    total_pos = int(y_series.loc[sub["repertoire_id"]].sum())
    total_neg = int(len(sub) - total_pos)
    if total_pos == 0 or total_neg == 0:
        return pd.DataFrame()

    stats: Dict[tuple, list] = {}  # key -> [pos_reps, total_reps]

    for row in sub.itertuples(index=False):
        rep_id = str(row.repertoire_id)
        label_pos = bool(y_series.loc[rep_id])
        file_path = os.path.join(train_dir, row.filename)

        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
        except Exception:
            continue

        df = df.dropna(subset=["junction_aa", "v_call", "j_call"]).drop_duplicates()
        if df.empty:
            continue

        for rec in df[["junction_aa", "v_call", "j_call"]].itertuples(index=False, name=None):
            if rec not in stats:
                stats[rec] = [0, 0]
            if label_pos:
                stats[rec][0] += 1
            stats[rec][1] += 1

        del df
        gc.collect()

    if not stats:
        return pd.DataFrame()

    rows = []
    for (junction_aa, v_call, j_call), (pos_reps, total_reps) in stats.items():
        if total_reps < min_total_reps:
            continue
        neg_reps = total_reps - pos_reps
        rows.append((junction_aa, v_call, j_call, pos_reps, neg_reps, total_reps))

    del stats
    gc.collect()

    if not rows:
        return pd.DataFrame()

    seq = pd.DataFrame(
        rows,
        columns=["junction_aa", "v_call", "j_call", "pos_reps", "neg_reps", "total_reps"],
    )

    a = alpha
    seq["p_pos"] = (seq["pos_reps"] + a) / (total_pos + 2 * a)
    seq["p_neg"] = (seq["neg_reps"] + a) / (total_neg + 2 * a)
    eps = 1e-12
    seq["log_odds"] = np.log(seq["p_pos"] + eps) - np.log(seq["p_neg"] + eps)

    seq = seq[seq["log_odds"] >= log_odds_min].sort_values("log_odds", ascending=False)
    if seq.empty:
        return pd.DataFrame()
    if max_library_size is not None and len(seq) > max_library_size:
        seq = seq.head(max_library_size)

    seq["seq_key"] = seq["junction_aa"] + "|" + seq["v_call"] + "|" + seq["j_call"]
    return seq.reset_index(drop=True)


def build_seq_presence_features_streaming(
    data_dir: str,
    meta_or_none: Optional[pd.DataFrame],
    ids: List[str],
    seq_lib: pd.DataFrame,
) -> np.ndarray:
    """
    Build 0/1 presence matrix for `ids` from `seq_lib` by streaming files.
    Returns numpy array shape (len(ids), n_features).
    """
    if seq_lib is None or seq_lib.empty:
        return np.zeros((len(ids), 0), dtype=np.uint8)

    tuples = list(zip(seq_lib["junction_aa"], seq_lib["v_call"], seq_lib["j_call"]))
    tup2col = {t: i for i, t in enumerate(tuples)}
    n_feats = len(tuples)

    X = np.zeros((len(ids), n_feats), dtype=np.uint8)
    id2row = {rid: i for i, rid in enumerate(ids)}

    if meta_or_none is not None:
        meta = meta_or_none.set_index("repertoire_id")
        for rid in ids:
            if rid not in meta.index:
                continue
            file_path = os.path.join(data_dir, meta.loc[rid, "filename"])
            try:
                df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
            except Exception:
                continue
            df = df.dropna(subset=["junction_aa", "v_call", "j_call"]).drop_duplicates()
            r = id2row[rid]
            for rec in df[["junction_aa", "v_call", "j_call"]].itertuples(index=False, name=None):
                j = tup2col.get(rec)
                if j is not None:
                    X[r, j] = 1
            del df
    else:
        for rid, file_path in iter_repertoires_no_metadata(data_dir):
            if rid not in id2row:
                continue
            try:
                df = pd.read_csv(file_path, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
            except Exception:
                continue
            df = df.dropna(subset=["junction_aa", "v_call", "j_call"]).drop_duplicates()
            r = id2row[rid]
            for rec in df[["junction_aa", "v_call", "j_call"]].itertuples(index=False, name=None):
                j = tup2col.get(rec)
                if j is not None:
                    X[r, j] = 1
            del df

    return X
