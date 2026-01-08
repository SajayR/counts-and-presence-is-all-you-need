import os
import argparse
import gc
import pandas as pd
from typing import List
from submission.predictor import ImmuneStatePredictor
from submission.utils import save_tsv, validate_dirs_and_files, concatenate_output_files

def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    all_preds = []
    for test_dir in test_dirs:
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions written to `{preds_path}`.")

def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(predictor.important_sequences_, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")

def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int = 8, device: str = "cpu",
         enable_seq: bool = True) -> None:
    """
    Runs training + prediction for a single train_dir against one or more test_dirs.
    """
    print(f"\nProcessing {train_dir} ({len(test_dirs)} test dirs)")
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device, enable_seq_family=enable_seq)
    predictor.fit(train_dir)
    predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)
    del predictor, predictions
    gc.collect()

def run():
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dir", required=True, nargs="+", help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for computation.")
    parser.add_argument("--disable_seq", action="store_true", help="Disable seq family to save time/IO.")
    args = parser.parse_args()

    main(
        train_dir=args.train_dir,
        test_dirs=args.test_dir,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        device=args.device,
        enable_seq=not args.disable_seq,
    )

    concatenate_output_files(args.out_dir)


if __name__ == "__main__":
    run()
