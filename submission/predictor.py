import os
import gc
import heapq
from typing import List, Optional, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from submission.utils import (
    read_train_metadata,
    iter_repertoires_no_metadata,
    load_and_encode_kmers,
    load_and_encode_kmers_ligo,
    load_and_encode_kmers_trunc,
    load_and_encode_kmers_posi,
    build_label_aware_seq_library_streaming,
    build_seq_presence_features_streaming,
)

class KmerClassifier:
    """
    L1 logistic regression with standardization.
    Key: CV runs with n_jobs=1 by default to avoid process copying X.
    """

    def __init__(self, c_values=None, cv_folds=5, opt_metric="roc_auc", random_state=123, cv_n_jobs=1):
        self.c_values = c_values or [10.5, 10, 5, 1, 0.5, 0.1, 0.05, 0.03, 0.0275]
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.cv_n_jobs = cv_n_jobs

        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None

    def _make_pipeline(self, C: float) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=500
            ))
        ])

    def tune(self, X, y) -> "KmerClassifier":
        """
        CV-tune C. Does NOT fit a final model.
        """
        if isinstance(X, pd.DataFrame):
            Xv = X.values
        else:
            Xv = X
        yv = y.values if isinstance(y, pd.Series) else y

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scorer = "roc_auc" if self.opt_metric == "roc_auc" else "balanced_accuracy"

        results = []
        for C in tqdm(self.c_values, desc="Tuning C values"):
            pipe = self._make_pipeline(C)
            scores = cross_val_score(pipe, Xv, yv, cv=cv, scoring=scorer, n_jobs=self.cv_n_jobs)
            results.append({"C": C, "mean_score": scores.mean(), "std_score": scores.std()})

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_["mean_score"].idxmax()
        self.best_C_ = float(self.cv_results_.loc[best_idx, "C"])
        self.best_score_ = float(self.cv_results_.loc[best_idx, "mean_score"])
        return self

    def fit_best(self, X, y) -> "KmerClassifier":
        """
        Fit final model on full data using best_C_ (must have tuned already).
        """
        if self.best_C_ is None:
            raise ValueError("Call tune() before fit_best().")

        if isinstance(X, pd.DataFrame):
            Xv = X.values
        else:
            Xv = X
        yv = y.values if isinstance(y, pd.Series) else y

        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(Xv, yv)
        return self

GAPPED_MASKS_CONFIG = [
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


def _crop_seq_for_trunc(raw: str, motif_crop: int) -> str:
    if not isinstance(raw, str):
        return ""
    if len(raw) <= 2 * motif_crop:
        return ""
    return raw[motif_crop:-motif_crop]


def _effective_weights_from_pipeline(pipe: Pipeline, feature_names: List[str]) -> Dict[str, float]:
    """
    Convert Pipeline(StandardScaler -> LogisticRegression) into effective weights in ORIGINAL feature space.
    """
    if pipe is None:
        raise ValueError("Pipeline is None.")
    if not hasattr(pipe, "named_steps"):
        raise ValueError("Model must be a sklearn Pipeline with named_steps.")

    if "scaler" not in pipe.named_steps or "classifier" not in pipe.named_steps:
        raise ValueError("Pipeline must contain 'scaler' and 'classifier' steps.")

    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["classifier"]

    coef = clf.coef_.reshape(-1)  # (n_features,)
    scale = getattr(scaler, "scale_", None)
    if scale is None:
        raise ValueError("StandardScaler missing scale_ (was it fit?).")

    scale = np.asarray(scale, dtype=np.float64)
    scale_safe = np.where(scale == 0, 1.0, scale)

    if len(coef) != len(feature_names) or len(scale_safe) != len(feature_names):
        raise ValueError(
            f"Shape mismatch: coef={len(coef)}, scale={len(scale_safe)}, feature_names={len(feature_names)}"
        )

    w_eff = coef / scale_safe
    return {fn: float(w) for fn, w in zip(feature_names, w_eff)}

def _accum_gapped(seq: str, prefix: str, wmap: Dict[str, float],
                 sum_w: np.ndarray, cnt: np.ndarray, offset: int) -> None:
    if not isinstance(seq, str):
        return
    L = len(seq)
    getw = wmap.get

    for mi, (k_len, indices, fmt) in enumerate(GAPPED_MASKS_CONFIG):
        if L < k_len:
            continue
        end = L - k_len + 1
        base_idx = offset + mi
        for i in range(end):
            residues = [seq[i + idx] for idx in indices]
            tok = fmt.format(*residues)
            feat = prefix + tok
            sum_w[base_idx] += float(getw(feat, 0.0))
            cnt[base_idx] += 1


def _score_gapped(seq: str, prefix: str, wmap: Dict[str, float],
                 C: np.ndarray) -> float:
    if not isinstance(seq, str):
        return 0.0
    L = len(seq)
    getw = wmap.get
    score = 0.0

    for mi, (k_len, indices, fmt) in enumerate(GAPPED_MASKS_CONFIG):
        if L < k_len:
            continue
        end = L - k_len + 1
        raw = 0.0
        for i in range(end):
            residues = [seq[i + idx] for idx in indices]
            tok = fmt.format(*residues)
            raw += float(getw(prefix + tok, 0.0))
        score += raw - (end * float(C[mi]))
    return float(score)


def _accum_ligo(seq: str, prefix: str, wmap: Dict[str, float],
                sum_w: np.ndarray, cnt: np.ndarray, offset: int) -> None:
    if not isinstance(seq, str):
        return
    L = len(seq)
    if L < 4:
        return
    getw = wmap.get
    end = L - 4 + 1

    for i in range(end):
        contig = seq[i:i + 4]
        gap1 = f"{seq[i]}.{seq[i+2]}{seq[i+3]}"
        gap2 = f"{seq[i]}{seq[i+1]}.{seq[i+3]}"

        sum_w[offset + 0] += float(getw(prefix + contig, 0.0)); cnt[offset + 0] += 1
        sum_w[offset + 1] += float(getw(prefix + gap1,   0.0)); cnt[offset + 1] += 1
        sum_w[offset + 2] += float(getw(prefix + gap2,   0.0)); cnt[offset + 2] += 1


def _score_ligo(seq: str, prefix: str, wmap: Dict[str, float],
                C: np.ndarray) -> float:
    if not isinstance(seq, str):
        return 0.0
    L = len(seq)
    if L < 4:
        return 0.0
    getw = wmap.get
    end = L - 4 + 1

    raw0 = raw1 = raw2 = 0.0
    for i in range(end):
        contig = seq[i:i + 4]
        gap1 = f"{seq[i]}.{seq[i+2]}{seq[i+3]}"
        gap2 = f"{seq[i]}{seq[i+1]}.{seq[i+3]}"
        raw0 += float(getw(prefix + contig, 0.0))
        raw1 += float(getw(prefix + gap1,   0.0))
        raw2 += float(getw(prefix + gap2,   0.0))

    score = (raw0 - end * float(C[0])) + (raw1 - end * float(C[1])) + (raw2 - end * float(C[2]))
    return float(score)


def _accum_trunc(seq: str, prefix: str, wmap: Dict[str, float],
                 sum_w: np.ndarray, cnt: np.ndarray, offset: int,
                 motif_crop: int) -> None:
    if not isinstance(seq, str):
        return
    cropped = _crop_seq_for_trunc(seq, motif_crop)
    L = len(cropped)
    if L < 4:
        return

    getw = wmap.get
    end = L - 4 + 1
    mid = end / 2.0

    for i in range(end):
        km = cropped[i:i + 4]
        sum_w[offset + 0] += float(getw(prefix + km, 0.0)); cnt[offset + 0] += 1
        if i < mid:
            sum_w[offset + 1] += float(getw(prefix + f"L_{km}", 0.0)); cnt[offset + 1] += 1
        else:
            sum_w[offset + 2] += float(getw(prefix + f"R_{km}", 0.0)); cnt[offset + 2] += 1


def _score_trunc(seq: str, prefix: str, wmap: Dict[str, float],
                 C: np.ndarray, motif_crop: int) -> float:
    if not isinstance(seq, str):
        return 0.0
    cropped = _crop_seq_for_trunc(seq, motif_crop)
    L = len(cropped)
    if L < 4:
        return 0.0

    getw = wmap.get
    end = L - 4 + 1
    mid = end / 2.0

    raw_g = raw_l = raw_r = 0.0
    n_l = n_r = 0

    for i in range(end):
        km = cropped[i:i + 4]
        raw_g += float(getw(prefix + km, 0.0))
        if i < mid:
            raw_l += float(getw(prefix + f"L_{km}", 0.0)); n_l += 1
        else:
            raw_r += float(getw(prefix + f"R_{km}", 0.0)); n_r += 1

    score = (raw_g - end * float(C[0])) + (raw_l - n_l * float(C[1])) + (raw_r - n_r * float(C[2]))
    return float(score)


def _accum_posi(seq: str, prefix: str, wmap: Dict[str, float],
                sum_w: np.ndarray, cnt: np.ndarray, offset: int) -> None:
    if not isinstance(seq, str):
        return
    L = len(seq)
    if L < 4:
        return

    getw = wmap.get
    end = L - 4 + 1
    denom = (L - 1) if L > 1 else 1
    for i in range(end):
        km = seq[i:i + 4]
        center = i + 1.5
        rel = center / denom
        if rel < 1/3:
            b = 0
            tok = f"S_{km}"
        elif rel < 2/3:
            b = 1
            tok = f"M_{km}"
        else:
            b = 2
            tok = f"E_{km}"

        sum_w[offset + b] += float(getw(prefix + tok, 0.0))
        cnt[offset + b] += 1


def _score_posi(seq: str, prefix: str, wmap: Dict[str, float],
                C: np.ndarray) -> float:
    if not isinstance(seq, str):
        return 0.0
    L = len(seq)
    if L < 4:
        return 0.0

    getw = wmap.get
    end = L - 4 + 1
    denom = (L - 1) if L > 1 else 1

    rawS = rawM = rawE = 0.0
    nS = nM = nE = 0

    for i in range(end):
        km = seq[i:i + 4]
        center = i + 1.5
        rel = center / denom
        if rel < 1/3:
            rawS += float(getw(prefix + f"S_{km}", 0.0)); nS += 1
        elif rel < 2/3:
            rawM += float(getw(prefix + f"M_{km}", 0.0)); nM += 1
        else:
            rawE += float(getw(prefix + f"E_{km}", 0.0)); nE += 1

    score = (rawS - nS * float(C[0])) + (rawM - nM * float(C[1])) + (rawE - nE * float(C[2]))
    return float(score)

class ImmuneStatePredictor:
    def __init__(
        self,
        n_jobs: int = 8,
        device: str = "cpu",
        enable_seq_family: bool = True,
        enable_auto2: bool = False,
        c_values: Optional[List[float]] = None,
        cv_folds: int = 5,
    ):
        total_cores = os.cpu_count() or 1
        self.n_jobs = total_cores if n_jobs == -1 else min(n_jobs, total_cores)
        self.device = device
        self.enable_seq_family = enable_seq_family
        self.enable_auto2 = enable_auto2
        self.c_values = c_values or [10.5, 10, 9.5, 5, 1, 0.5, 0.1, 0.05, 0.03, 0.0275]
        self.cv_folds = cv_folds
        self.model_family_ = None
        self.best_C_ = None
        self.best_score_ = None
        self.auto2_pair_ = None

        self.model = None
        self.feature_names_ = None

        self.seq_lib_ = None
        self.seq_feature_names_ = None

        self.important_sequences_ = None

    def _collapse_duplicate_columns(self, X_df: pd.DataFrame) -> pd.DataFrame:
        if X_df is None or X_df.empty:
            return X_df
        if X_df.columns.has_duplicates:
            X_df = X_df.T.groupby(level=0).sum().T
        return X_df

    def _encode_family_train(self, family: str, train_dir: str, train_ids_index: pd.Index) -> pd.DataFrame:
        if family == "kmer":
            X_df, _ = load_and_encode_kmers(train_dir, n_jobs=self.n_jobs)
            X_df = X_df.reindex(train_ids_index).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"kmer__{c}", inplace=True)
            return X_df

        if family == "kmer_ligo":
            X_df, _ = load_and_encode_kmers_ligo(train_dir, k=4, n_jobs=self.n_jobs)
            X_df = X_df.reindex(train_ids_index).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"ligo__{c}", inplace=True)
            return X_df

        if family == "kmer_trunc":
            X_df, _ = load_and_encode_kmers_trunc(train_dir, k=4, n_jobs=self.n_jobs)
            X_df = X_df.reindex(train_ids_index).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"trunc__{c}", inplace=True)
            return X_df

        if family == "kmer_posi":
            X_df, _ = load_and_encode_kmers_posi(train_dir, k=4, n_jobs=self.n_jobs)
            X_df = X_df.reindex(train_ids_index).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"posi__{c}", inplace=True)
            return X_df

        if family == "kmer_combined":
            X_a, _ = load_and_encode_kmers(train_dir, n_jobs=self.n_jobs)
            X_b, _ = load_and_encode_kmers_ligo(train_dir, k=4, n_jobs=self.n_jobs)

            X_a = X_a.reindex(train_ids_index).fillna(0.0)
            X_b = X_b.reindex(train_ids_index).fillna(0.0)

            X_a.rename(columns=lambda c: f"base__{c}", inplace=True)
            X_b.rename(columns=lambda c: f"ligo__{c}", inplace=True)

            X_df = pd.concat([X_a, X_b], axis=1).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)

            X_df.rename(columns=lambda c: f"combined__{c}", inplace=True)

            del X_a, X_b
            gc.collect()
            return X_df

        raise ValueError(f"Unknown family: {family}")

    def _encode_family_test(self, family: str, test_dir: str) -> Tuple[pd.DataFrame, List[str]]:
        if family == "kmer":
            X_df, _ = load_and_encode_kmers(test_dir, n_jobs=self.n_jobs)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"kmer__{c}", inplace=True)
            return X_df, X_df.index.tolist()

        if family == "kmer_ligo":
            X_df, _ = load_and_encode_kmers_ligo(test_dir, k=4, n_jobs=self.n_jobs)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"ligo__{c}", inplace=True)
            return X_df, X_df.index.tolist()

        if family == "kmer_trunc":
            X_df, _ = load_and_encode_kmers_trunc(test_dir, k=4, n_jobs=self.n_jobs)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"trunc__{c}", inplace=True)
            return X_df, X_df.index.tolist()

        if family == "kmer_posi":
            X_df, _ = load_and_encode_kmers_posi(test_dir, k=4, n_jobs=self.n_jobs)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"posi__{c}", inplace=True)
            return X_df, X_df.index.tolist()

        if family == "kmer_combined":
            X_a, _ = load_and_encode_kmers(test_dir, n_jobs=self.n_jobs)
            X_b, _ = load_and_encode_kmers_ligo(test_dir, k=4, n_jobs=self.n_jobs)

            all_ids = X_a.index.union(X_b.index)
            X_a = X_a.reindex(all_ids).fillna(0.0)
            X_b = X_b.reindex(all_ids).fillna(0.0)

            X_a.rename(columns=lambda c: f"base__{c}", inplace=True)
            X_b.rename(columns=lambda c: f"ligo__{c}", inplace=True)

            X_df = pd.concat([X_a, X_b], axis=1).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.rename(columns=lambda c: f"combined__{c}", inplace=True)

            del X_a, X_b
            gc.collect()
            return X_df, all_ids.tolist()

        raise ValueError(f"Unknown family: {family}")

    def _score_kmer_family(
        self,
        family: str,
        train_dir: str,
        train_ids_index: pd.Index,
        y: pd.Series,
        c_values: list,
        cv_folds: int,
        cv_n_jobs: int = 1,
    ) -> Tuple[Optional[float], Optional[float]]:
        print(f"\nScoring family: {family}")

        X_df = self._encode_family_train(family, train_dir, train_ids_index)

        if X_df is None or X_df.empty or X_df.shape[1] == 0:
            print(f"[{family}] Empty features; skipping.")
            return None, None

        clf = KmerClassifier(
            c_values=c_values,
            cv_folds=cv_folds,
            opt_metric="roc_auc",
            random_state=123,
            cv_n_jobs=1,
        )
        clf.tune(X_df, y)

        score, best_C = clf.best_score_, clf.best_C_
        print(f"[{family}] CV roc_auc: {score:.4f} | best_C={best_C}")

        del X_df, clf
        gc.collect()
        return score, best_C

    def _score_auto2(
        self,
        fam1: str,
        fam2: str,
        train_dir: str,
        train_ids_index: pd.Index,
        y: pd.Series,
        c_values: list,
        cv_folds: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        print(f"\nScoring family: auto2 ({fam1} + {fam2})")

        X1 = self._encode_family_train(fam1, train_dir, train_ids_index)
        X2 = self._encode_family_train(fam2, train_dir, train_ids_index)

        if X1 is None or X2 is None or X1.empty or X2.empty:
            print("[auto2] Missing/empty component features; skipping.")
            return None, None

        X1.rename(columns=lambda c: f"m1__{c}", inplace=True)
        X2.rename(columns=lambda c: f"m2__{c}", inplace=True)

        X = pd.concat([X1, X2], axis=1).fillna(0.0)
        X = self._collapse_duplicate_columns(X)

        if X.columns.has_duplicates:
            dups = X.columns[X.columns.duplicated()].unique().tolist()[:10]
            raise ValueError(f"[auto2] Duplicate feature columns remain (sample): {dups}")

        X.values[:] = normalize(X.values, norm="l2")

        clf = KmerClassifier(
            c_values=c_values,
            cv_folds=cv_folds,
            opt_metric="roc_auc",
            random_state=123,
            cv_n_jobs=1,
        )
        clf.tune(X, y)

        score, best_C = clf.best_score_, clf.best_C_
        print(f"[auto2] CV roc_auc: {score:.4f} | best_C={best_C}")

        del X1, X2, X, clf
        gc.collect()
        return score, best_C

    def _score_seq_family_streaming(
        self,
        train_dir: str,
        meta: pd.DataFrame,
        train_ids: np.ndarray,
        y_series: pd.Series,
        c_values: list,
        cv_folds: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        print(f"\nScoring family: seq (streaming folds)")

        y_arr = y_series.loc[train_ids].astype(int).values
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=123)

        scores_by_C = {C: [] for C in c_values}

        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(train_ids, y_arr), start=1):
            print(f"  Seq fold {fold_i}/{cv_folds}")

            ids_tr = train_ids[tr_idx]
            ids_va = train_ids[va_idx]
            y_tr = y_arr[tr_idx]
            y_va = y_arr[va_idx]

            seq_lib = build_label_aware_seq_library_streaming(
                train_dir=train_dir,
                meta=meta,
                ids=ids_tr,
                y_series=y_series,
                min_total_reps=3,
                max_library_size=20000,
                alpha=0.5,
                log_odds_min=0.5,
            )
            if seq_lib is None or seq_lib.empty:
                print("  Seq lib empty in fold; aborting seq family.")
                return None, None

            X_tr = build_seq_presence_features_streaming(train_dir, meta, ids_tr.tolist(), seq_lib)
            X_va = build_seq_presence_features_streaming(train_dir, meta, ids_va.tolist(), seq_lib)

            if X_tr.shape[1] == 0:
                print("  Seq features empty in fold; aborting seq family.")
                return None, None

            for C in c_values:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(
                        penalty="l1",
                        C=C,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=123,
                        max_iter=500
                    ))
                ])
                pipe.fit(X_tr, y_tr)
                probs = pipe.predict_proba(X_va)[:, 1]
                scores_by_C[C].append(roc_auc_score(y_va, probs))

            del seq_lib, X_tr, X_va, pipe, probs
            gc.collect()

        mean_scores = {C: float(np.mean(v)) for C, v in scores_by_C.items()}
        best_C = max(mean_scores.items(), key=lambda kv: kv[1])[0]
        best_score = mean_scores[best_C]
        print(f"[seq] CV roc_auc: {best_score:.4f} | best_C={best_C}")

        return best_score, best_C

    def fit(self, train_dir: str):
        print(f"\n--- Training on {train_dir} (sequential family scoring) ---")

        meta = read_train_metadata(train_dir)
        train_ids = meta["repertoire_id"].astype(str).values
        y_series = meta.set_index("repertoire_id")["label_positive"].astype(int)
        train_ids_index = pd.Index(train_ids)

        c_values = self.c_values
        cv_folds = self.cv_folds

        family_results: dict[str, tuple[Optional[float], Optional[float]]] = {}
        best_name, best_score, best_C = None, -1.0, None

        y_train = y_series.loc[train_ids]

        s, C = self._score_kmer_family("kmer", train_dir, train_ids_index, y_train, c_values, cv_folds)
        family_results["kmer"] = (s, C)
        if s is not None and s > best_score:
            best_name, best_score, best_C = "kmer", s, C

        s, C = self._score_kmer_family("kmer_ligo", train_dir, train_ids_index, y_train, c_values, cv_folds)
        family_results["kmer_ligo"] = (s, C)
        if s is not None and s > best_score:
            best_name, best_score, best_C = "kmer_ligo", s, C

        s, C = self._score_kmer_family("kmer_combined", train_dir, train_ids_index, y_train, c_values, cv_folds)
        family_results["kmer_combined"] = (s, C)
        if s is not None and s > best_score:
            best_name, best_score, best_C = "kmer_combined", s, C

        s, C = self._score_kmer_family("kmer_trunc", train_dir, train_ids_index, y_train, c_values, cv_folds)
        family_results["kmer_trunc"] = (s, C)
        if s is not None and s > best_score:
            best_name, best_score, best_C = "kmer_trunc", s, C

        s, C = self._score_kmer_family("kmer_posi", train_dir, train_ids_index, y_train, c_values, cv_folds)
        family_results["kmer_posi"] = (s, C)
        if s is not None and s > best_score:
            best_name, best_score, best_C = "kmer_posi", s, C

        if self.enable_seq_family:
            s, C = self._score_seq_family_streaming(train_dir, meta, train_ids, y_series, c_values, cv_folds)
            family_results["seq"] = (s, C)
            if s is not None and s > best_score:
                best_name, best_score, best_C = "seq", s, C
        else:
            print("\nSkipping seq family (enable_seq_family=False)")

        if self.enable_auto2:
            candidates = []
            for fam, (sc, c) in family_results.items():
                if fam == "seq":
                    continue
                if sc is None:
                    continue
                candidates.append((fam, float(sc)))
            candidates.sort(key=lambda x: x[1], reverse=True)

            if len(candidates) >= 2:
                fam1 = candidates[0][0]
                fam2 = candidates[1][0]
                s, C = self._score_auto2(fam1, fam2, train_dir, train_ids_index, y_train, c_values, cv_folds)
                family_results["auto2"] = (s, C)
                if s is not None and s > best_score:
                    best_name, best_score, best_C = "auto2", s, C
                    self.auto2_pair_ = (fam1, fam2)
            else:
                print("\nSkipping auto2 (need at least two scored non-seq families).")
        else:
            print("\nSkipping auto2 (enable_auto2=False).")

        self.model_family_ = best_name
        self.best_score_ = best_score
        self.best_C_ = best_C

        if self.model_family_ == "auto2":
            print(
                f"\n=> Selected family: auto2({self.auto2_pair_[0]} + {self.auto2_pair_[1]}) "
                f"| CV roc_auc={self.best_score_:.4f} | best_C={self.best_C_}"
            )
        else:
            print(f"\n=> Selected family: {self.model_family_} | CV roc_auc={self.best_score_:.4f} | best_C={self.best_C_}")

        if self.model_family_ in {"kmer", "kmer_ligo", "kmer_trunc", "kmer_posi", "kmer_combined"}:
            X_df = self._encode_family_train(self.model_family_, train_dir, train_ids_index)
            X_df = X_df.reindex(train_ids_index).fillna(0.0)

            self.feature_names_ = X_df.columns.tolist()

            clf = KmerClassifier(
                c_values=[float(self.best_C_)],
                cv_folds=5,
                opt_metric="roc_auc",
                random_state=123,
                cv_n_jobs=1,
            )
            clf.best_C_ = float(self.best_C_)
            clf.fit_best(X_df, y_train)
            self.model = clf.model_

            del X_df, clf
            gc.collect()

        elif self.model_family_ == "auto2":
            if not self.auto2_pair_:
                raise RuntimeError("Selected auto2 but auto2_pair_ is missing.")
            fam1, fam2 = self.auto2_pair_

            X1 = self._encode_family_train(fam1, train_dir, train_ids_index)
            X2 = self._encode_family_train(fam2, train_dir, train_ids_index)

            X1.rename(columns=lambda c: f"m1__{c}", inplace=True)
            X2.rename(columns=lambda c: f"m2__{c}", inplace=True)

            X_df = pd.concat([X1, X2], axis=1).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.values[:] = normalize(X_df.values, norm="l2")

            if X_df.columns.has_duplicates:
                dups = X_df.columns[X_df.columns.duplicated()].unique().tolist()[:10]
                raise ValueError(f"[auto2-fit] Duplicate columns remain (sample): {dups}")

            self.feature_names_ = X_df.columns.tolist()

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty="l1",
                    C=float(self.best_C_),
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=123,
                    max_iter=500
                ))
            ])
            pipe.fit(X_df.values, y_train.values.astype(int))
            self.model = pipe

            del X1, X2, X_df, pipe
            gc.collect()

        elif self.model_family_ == "seq":
            seq_lib = build_label_aware_seq_library_streaming(
                train_dir=train_dir,
                meta=meta,
                ids=train_ids,
                y_series=y_series,
                min_total_reps=3,
                max_library_size=20000,
                alpha=0.5,
                log_odds_min=0.5,
            )
            if seq_lib is None or seq_lib.empty:
                raise RuntimeError("Selected seq family but seq_lib is empty.")

            self.seq_lib_ = seq_lib
            self.seq_feature_names_ = seq_lib["seq_key"].tolist()

            X = build_seq_presence_features_streaming(train_dir, meta, train_ids.tolist(), seq_lib)

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty="l1",
                    C=float(self.best_C_),
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=123,
                    max_iter=500
                ))
            ])
            pipe.fit(X, y_train.values.astype(int))

            self.model = pipe
            self.feature_names_ = self.seq_feature_names_

            del X, pipe
            gc.collect()

        else:
            raise RuntimeError(f"Unknown model_family_ {self.model_family_}")

        if self.model_family_ == "seq":
            self.important_sequences_ = self.identify_associated_sequences_seq_20k_plus_logodds(train_dir, top_k=50000)
        else:
            self.important_sequences_ = self.identify_associated_sequences_baseline_corrected(train_dir, top_k=50000)

        print("Training complete.")
        return self

    def predict_proba(self, test_dir: str) -> pd.DataFrame:
        if self.model is None or self.model_family_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        print(f"Predicting on {test_dir} using family {self.model_family_}...")
        dataset_name = os.path.basename(test_dir)

        if self.model_family_ == "seq":
            if self.seq_lib_ is None or self.seq_lib_.empty:
                raise RuntimeError("seq_lib_ missing for seq model.")

            ids = [rid for rid, _fp in iter_repertoires_no_metadata(test_dir)]
            X = build_seq_presence_features_streaming(test_dir, None, ids, self.seq_lib_)
            probs = self.model.predict_proba(X)[:, 1]

            preds = pd.DataFrame({
                "ID": ids,
                "dataset": [dataset_name] * len(ids),
                "label_positive_probability": probs,
                "junction_aa": -999.0,
                "v_call": -999.0,
                "j_call": -999.0
            })
            return preds[["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]]

        if self.model_family_ == "auto2":
            if not self.auto2_pair_:
                raise RuntimeError("auto2_pair_ missing for auto2 model.")
            fam1, fam2 = self.auto2_pair_

            X1, ids1 = self._encode_family_test(fam1, test_dir)
            X2, ids2 = self._encode_family_test(fam2, test_dir)

            all_ids = pd.Index(ids1).union(pd.Index(ids2))
            X1 = X1.reindex(all_ids).fillna(0.0)
            X2 = X2.reindex(all_ids).fillna(0.0)

            X1.rename(columns=lambda c: f"m1__{c}", inplace=True)
            X2.rename(columns=lambda c: f"m2__{c}", inplace=True)

            X_df = pd.concat([X1, X2], axis=1).fillna(0.0)
            X_df = self._collapse_duplicate_columns(X_df)
            X_df.values[:] = normalize(X_df.values, norm="l2")

            if X_df.columns.has_duplicates:
                dups = X_df.columns[X_df.columns.duplicated()].unique().tolist()[:10]
                raise ValueError(f"[auto2-predict] Duplicate columns remain (sample): {dups}")

            ids = all_ids.tolist()

            del X1, X2
            gc.collect()

        else:
            X_df, ids = self._encode_family_test(self.model_family_, test_dir)

        X_df = X_df.reindex(columns=self.feature_names_, fill_value=0.0)
        probs = self.model.predict_proba(X_df.values)[:, 1]

        preds = pd.DataFrame({
            "ID": ids,
            "dataset": [dataset_name] * len(ids),
            "label_positive_probability": probs,
        })
        preds["junction_aa"] = -999.0
        preds["v_call"] = -999.0
        preds["j_call"] = -999.0

        del X_df
        gc.collect()

        return preds[["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]]

    def _make_bc_components(self) -> List[dict]:
        comps = []
        offset = 0

        def add_component(tp: str, prefix: str, n_masks: int,
                          dedupe_per_file: bool = False, motif_crop: int = 4):
            nonlocal offset
            comps.append({
                "type": tp,
                "prefix": prefix,
                "n_masks": n_masks,
                "offset": offset,
                "dedupe_per_file": bool(dedupe_per_file),
                "motif_crop": int(motif_crop),
            })
            offset += n_masks

        def add_family(fam: str, outer: str = ""):
            if fam == "kmer":
                add_component("gapped", outer + "kmer__", 11, dedupe_per_file=False)
            elif fam == "kmer_ligo":
                add_component("ligo", outer + "ligo__", 3, dedupe_per_file=False)
            elif fam == "kmer_trunc":
                add_component("trunc", outer + "trunc__", 3, dedupe_per_file=True, motif_crop=4)
            elif fam == "kmer_posi":
                add_component("posi", outer + "posi__", 3, dedupe_per_file=False)
            elif fam == "kmer_combined":
                add_component("gapped", outer + "combined__base__", 11, dedupe_per_file=False)
                add_component("ligo",  outer + "combined__ligo__", 3,  dedupe_per_file=False)
            else:
                raise ValueError(f"Unknown family for bc components: {fam}")

        if self.model_family_ in {"kmer", "kmer_ligo", "kmer_trunc", "kmer_posi", "kmer_combined"}:
            add_family(self.model_family_)
        elif self.model_family_ == "auto2":
            if not self.auto2_pair_:
                raise RuntimeError("auto2_pair_ missing but model_family_ == auto2")
            fam1, fam2 = self.auto2_pair_
            add_family(fam1, outer="m1__")
            add_family(fam2, outer="m2__")
        else:
            raise RuntimeError(f"BC scoring not supported for model_family_={self.model_family_}")

        return comps

    def _estimate_bc_mask_baselines(
        self,
        train_dir: str,
        meta: pd.DataFrame,
        comps: List[dict],
        wmap: Dict[str, float],
    ) -> np.ndarray:
        n_masks_total = sum(c["n_masks"] for c in comps)
        sum_w = np.zeros(n_masks_total, dtype=np.float64)
        cnt = np.zeros(n_masks_total, dtype=np.int64)

        neg_meta = meta[meta["label_positive"].astype(bool) == False]
        if len(neg_meta) == 0:
            return np.zeros(n_masks_total, dtype=np.float64)

        any_dedupe = any(c["dedupe_per_file"] for c in comps)

        for row in tqdm(neg_meta.itertuples(index=False), total=len(neg_meta), desc="Estimating baselines (negatives)"):
            fp = os.path.join(train_dir, row.filename)
            try:
                df = pd.read_csv(fp, sep="\t", usecols=["junction_aa"])
            except Exception:
                continue

            seqs = df["junction_aa"].dropna().tolist()
            if not seqs:
                del df
                continue

            unique_seqs = None
            if any_dedupe:
                unique_seqs = list(set(s for s in seqs if isinstance(s, str)))

            for comp in comps:
                tp = comp["type"]
                prefix = comp["prefix"]
                off = comp["offset"]
                if comp["dedupe_per_file"]:
                    it = unique_seqs if unique_seqs is not None else []
                else:
                    it = seqs

                if tp == "gapped":
                    for s in it:
                        _accum_gapped(s, prefix, wmap, sum_w, cnt, off)
                elif tp == "ligo":
                    for s in it:
                        _accum_ligo(s, prefix, wmap, sum_w, cnt, off)
                elif tp == "trunc":
                    mc = comp["motif_crop"]
                    for s in it:
                        _accum_trunc(s, prefix, wmap, sum_w, cnt, off, motif_crop=mc)
                elif tp == "posi":
                    for s in it:
                        _accum_posi(s, prefix, wmap, sum_w, cnt, off)
                else:
                    raise ValueError(f"Unknown component type {tp}")

            del df
            gc.collect()

        C = np.zeros_like(sum_w, dtype=np.float64)
        nz = cnt > 0
        C[nz] = sum_w[nz] / cnt[nz]
        return C

    def _score_junction_bc(self, junction_aa: str, comps: List[dict], wmap: Dict[str, float], C: np.ndarray) -> float:
        if not isinstance(junction_aa, str):
            return 0.0

        total = 0.0
        for comp in comps:
            tp = comp["type"]
            prefix = comp["prefix"]
            off = comp["offset"]
            n = comp["n_masks"]
            Cslice = C[off:off+n]

            if tp == "gapped":
                total += _score_gapped(junction_aa, prefix, wmap, Cslice)
            elif tp == "ligo":
                total += _score_ligo(junction_aa, prefix, wmap, Cslice)
            elif tp == "trunc":
                total += _score_trunc(junction_aa, prefix, wmap, Cslice, motif_crop=comp["motif_crop"])
            elif tp == "posi":
                total += _score_posi(junction_aa, prefix, wmap, Cslice)
            else:
                raise ValueError(f"Unknown component type {tp}")

        return float(total)

    def identify_associated_sequences_baseline_corrected(self, train_dir: str, top_k: int = 50000) -> pd.DataFrame:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit; cannot rank sequences.")

        dataset_name = os.path.basename(train_dir)
        meta = read_train_metadata(train_dir)

        wmap = _effective_weights_from_pipeline(self.model, self.feature_names_)
        comps = self._make_bc_components()
        C = self._estimate_bc_mask_baselines(train_dir, meta, comps, wmap)

        heap_list: List[Tuple[float, Tuple[str, str, str]]] = []
        in_heap: set = set()

        score_cache: Dict[str, float] = {}

        for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Ranking sequences (BC)"):
            fp = os.path.join(train_dir, row.filename)
            try:
                df = pd.read_csv(fp, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
            except Exception:
                continue

            df = df.dropna(subset=["junction_aa", "v_call", "j_call"])
            if df.empty:
                del df
                continue

            for rec in df.itertuples(index=False, name=None):
                junction_aa, v_call, j_call = rec
                if not isinstance(junction_aa, str):
                    continue

                key = (junction_aa, str(v_call), str(j_call))
                if key in in_heap:
                    continue

                sc = score_cache.get(junction_aa)
                if sc is None:
                    sc = self._score_junction_bc(junction_aa, comps, wmap, C)
                    score_cache[junction_aa] = sc
                    if len(score_cache) > 1_000_000:
                        score_cache.clear()

                if len(heap_list) < top_k:
                    heapq.heappush(heap_list, (sc, key))
                    in_heap.add(key)
                else:
                    if sc <= heap_list[0][0]:
                        continue
                    old_sc, old_key = heapq.heappop(heap_list)
                    in_heap.remove(old_key)
                    heapq.heappush(heap_list, (sc, key))
                    in_heap.add(key)

            del df
            gc.collect()

        heap_list.sort(key=lambda x: x[0], reverse=True)

        rows = []
        for i, (sc, (junction_aa, v_call, j_call)) in enumerate(heap_list, start=1):
            rows.append((f"{dataset_name}_seq_top_{i}", dataset_name, -999.0, junction_aa, v_call, j_call))

        if len(rows) < top_k:
            need = top_k - len(rows)
            fallback = self.identify_associated_sequences_streaming(train_dir, top_k=top_k + 5000)
            used = set((r[3], r[4], r[5]) for r in rows)
            extra = []
            for rr in fallback.itertuples(index=False, name=None):
                _id, _ds, _p, aa, v, j = rr
                k = (aa, v, j)
                if k in used:
                    continue
                used.add(k)
                extra.append((f"{dataset_name}_seq_top_{len(rows)+len(extra)+1}", dataset_name, -999.0, aa, v, j))
                if len(extra) >= need:
                    break
            rows.extend(extra)

        return pd.DataFrame(
            rows[:top_k],
            columns=["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
        )

    def identify_associated_sequences_seq_20k_plus_logodds(self, train_dir: str, top_k: int = 50000) -> pd.DataFrame:
        if self.model_family_ != "seq":
            raise RuntimeError("This method is only for seq model_family_.")
        if self.model is None or self.seq_lib_ is None or self.seq_lib_.empty:
            raise RuntimeError("seq_lib_ or model missing for seq ranking.")

        dataset_name = os.path.basename(train_dir)

        wmap = _effective_weights_from_pipeline(self.model, self.feature_names_)
        seq_df = self.seq_lib_.copy()
        seq_df["w_eff"] = seq_df["seq_key"].map(lambda k: wmap.get(k, 0.0)).astype(float)

        n_total = len(seq_df)

        zero_eps = 1e-10
        abs_w = seq_df["w_eff"].abs()

        n_zero = int((abs_w <= zero_eps).sum())
        n_pos  = int((seq_df["w_eff"] >  zero_eps).sum())
        n_neg  = int((seq_df["w_eff"] < -zero_eps).sum())

        print(
            f"[seq-rank] weights: total={n_total:,} | "
            f"near_zero(|w|<= {zero_eps:g})={n_zero:,} | pos={n_pos:,} | neg={n_neg:,}"
        )

        eps = 1e-12
        seq_df_pos = seq_df[seq_df["w_eff"] > eps].copy()

        sort_cols = ["w_eff"]
        asc = [False]
        if "log_odds" in seq_df_pos.columns:
            sort_cols.append("log_odds"); asc.append(False)
        if "total_reps" in seq_df_pos.columns:
            sort_cols.append("total_reps"); asc.append(False)

        seq_df_pos = seq_df_pos.sort_values(sort_cols, ascending=asc)

        model_part = seq_df_pos.head(min(len(seq_df_pos), 20000)).copy()

        selected_keys = set(model_part["seq_key"].tolist())

        meta = read_train_metadata(train_dir)
        y_series = meta.set_index("repertoire_id")["label_positive"].astype(int)

        extended = build_label_aware_seq_library_streaming(
            train_dir=train_dir,
            meta=meta,
            ids=meta["repertoire_id"].astype(str).values,
            y_series=y_series,
            min_total_reps=3,
            max_library_size=max(top_k + 20000 + 5000, 80000),
            alpha=0.5,
            log_odds_min=0.5,
        )

        extras = []
        if extended is not None and not extended.empty:
            for r in extended.itertuples(index=False):
                sk = r.seq_key
                if sk in selected_keys:
                    continue
                extras.append((r.junction_aa, r.v_call, r.j_call, sk))
                selected_keys.add(sk)
                if len(model_part) + len(extras) >= top_k:
                    break

        rows = []
        i = 1
        for r in model_part.itertuples(index=False):
            rows.append((f"{dataset_name}_seq_top_{i}", dataset_name, -999.0, r.junction_aa, r.v_call, r.j_call))
            i += 1
            if len(rows) >= top_k:
                break

        if len(rows) < top_k:
            for (aa, v, j, _sk) in extras:
                rows.append((f"{dataset_name}_seq_top_{i}", dataset_name, -999.0, aa, v, j))
                i += 1
                if len(rows) >= top_k:
                    break

        if len(rows) < top_k:
            need = top_k - len(rows)
            fallback = self.identify_associated_sequences_streaming(train_dir, top_k=top_k + 5000)
            used = set((r[3], r[4], r[5]) for r in rows)
            for rr in fallback.itertuples(index=False, name=None):
                _id, _ds, _p, aa, v, j = rr
                k = (aa, v, j)
                if k in used:
                    continue
                used.add(k)
                rows.append((f"{dataset_name}_seq_top_{len(rows)+1}", dataset_name, -999.0, aa, v, j))
                if len(rows) >= top_k:
                    break

        return pd.DataFrame(
            rows[:top_k],
            columns=["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
        )

    def identify_associated_sequences_seq_model_only(self, train_dir: str, top_k: int = 50000) -> pd.DataFrame:
        if self.model_family_ != "seq":
            raise RuntimeError("This method is only for seq model_family_.")
        if self.model is None or self.seq_lib_ is None or self.seq_lib_.empty:
            raise RuntimeError("seq_lib_ or model missing for seq ranking.")

        dataset_name = os.path.basename(train_dir)

        wmap = _effective_weights_from_pipeline(self.model, self.feature_names_)
        seq_df = self.seq_lib_.copy()
        seq_df["w_eff"] = seq_df["seq_key"].map(lambda k: float(wmap.get(k, 0.0)))

        eps = 1e-12
        w = seq_df["w_eff"].values
        grp = np.where(w > eps, 0, np.where(w < -eps, 2, 1))
        seq_df["grp"] = grp

        sort_cols = ["grp", "w_eff"]
        asc = [True, False]
        if "log_odds" in seq_df.columns:
            sort_cols.append("log_odds"); asc.append(False)
        if "total_reps" in seq_df.columns:
            sort_cols.append("total_reps"); asc.append(False)
        sort_cols.append("seq_key"); asc.append(True)

        seq_df = seq_df.sort_values(sort_cols, ascending=asc)

        chosen = seq_df.head(min(len(seq_df), top_k))

        rows = []
        for i, r in enumerate(chosen.itertuples(index=False), start=1):
            rows.append((f"{dataset_name}_seq_top_{i}", dataset_name, -999.0, r.junction_aa, r.v_call, r.j_call))

        if len(rows) < top_k:
            need = top_k - len(rows)
            fallback = self.identify_associated_sequences_streaming(train_dir, top_k=top_k + 5000)
            used = set((rr[3], rr[4], rr[5]) for rr in rows)
            for rr in fallback.itertuples(index=False, name=None):
                _id, _ds, _p, aa, v, j = rr
                k = (aa, v, j)
                if k in used:
                    continue
                used.add(k)
                rows.append((f"{dataset_name}_seq_top_{len(rows)+1}", dataset_name, -999.0, aa, v, j))
                if len(rows) >= top_k:
                    break

        return pd.DataFrame(
            rows[:top_k],
            columns=["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
        )

    def identify_associated_sequences_streaming(self, train_dir: str, top_k: int = 50000) -> pd.DataFrame:
        print(f"Building top-{top_k} frequent sequences (streaming) for {train_dir}...")
        dataset_name = os.path.basename(train_dir)
        meta = read_train_metadata(train_dir)

        counter = Counter()

        for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Counting sequences"):
            fp = os.path.join(train_dir, row.filename)
            try:
                df = pd.read_csv(fp, sep="\t", usecols=["junction_aa", "v_call", "j_call"])
            except Exception:
                continue
            df = df.dropna(subset=["junction_aa", "v_call", "j_call"])
            if df.empty:
                continue

            vc = df.value_counts(["junction_aa", "v_call", "j_call"])
            for k, v in vc.items():
                counter[k] += int(v)

            del df, vc
            if len(counter) > 5_000_000:
                counter = Counter(dict(counter.most_common(top_k * 50)))
            gc.collect()

        most = counter.most_common(top_k)
        rows = []
        for i, ((junction_aa, v_call, j_call), _cnt) in enumerate(most, start=1):
            rows.append((f"{dataset_name}_seq_top_{i}", dataset_name, -999.0, junction_aa, v_call, j_call))

        return pd.DataFrame(
            rows,
            columns=["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]
        )
