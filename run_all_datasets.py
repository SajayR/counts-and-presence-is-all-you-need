import argparse
import os
from submission.main import main
from submission.utils import get_dataset_pairs, concatenate_output_files


def run_all(train_root: str, test_root: str, out_dir: str, n_jobs: int, device: str, disable_seq: bool) -> None:
    pairs = get_dataset_pairs(train_root, test_root)
    if not pairs:
        print("No train/test dataset pairs found.")
        return

    for train_dir, test_dirs in pairs:
        if not test_dirs:
            print(f"Skipping {train_dir}: no matching test dirs found.")
            continue
        main(
            train_dir=train_dir,
            test_dirs=test_dirs,
            out_dir=out_dir,
            n_jobs=n_jobs,
            device=device,
            enable_seq=not disable_seq,
        )

    concatenate_output_files(out_dir)


def run():
    parser = argparse.ArgumentParser(description="Batch runner over train_dataset_* and test_dataset_* folders.")
    parser.add_argument("--train_datasets_dir", required=True, help="Directory containing train_dataset_* folders")
    parser.add_argument("--test_datasets_dir", required=True, help="Directory containing test_dataset_* folders")
    parser.add_argument("--out_dir", required=True, help="Directory to write predictions/sequence files")
    parser.add_argument("--n_jobs", type=int, default=24, help="Number of CPU cores to use. Use -1 for all available cores.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for computation.")
    parser.add_argument("--disable_seq", action="store_true", help="Disable seq family to save time/IO.")
    args = parser.parse_args()

    run_all(
        train_root=args.train_datasets_dir,
        test_root=args.test_datasets_dir,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        device=args.device,
        disable_seq=args.disable_seq,
    )


if __name__ == "__main__":
    run()
