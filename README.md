## AIRR-ML-25 submission code

This repo now contains our full competition submission.

- `submission/predictor.py`: ImmuneStatePredictor with sequential family scoring (kmer, ligo, trunc, posi, seq, optional auto2), training, prediction, and important-sequence ranking.
- `submission/utils.py`: data loading/validation, k-mer encoders, streaming sequence library builders, output concatenation (with sequence ID normalization).
- `submission/main.py`: single-train-dir entrypoint matching the template contract.
- `run_all_datasets.py`: optional helper to loop over `train_dataset_*`/`test_dataset_*` pairs.

### Quick start (single train/test set)
```bash
python -m submission.main \
  --train_dir /path/to/train_dataset_X \
  --test_dir /path/to/test_dataset_X /path/to/other_test_dirs... \
  --out_dir /path/to/output \
  --n_jobs 8 \
  --device cpu
```
This writes:
- `/path/to/output/train_dataset_X_test_predictions.tsv`
- `/path/to/output/train_dataset_X_important_sequences.tsv`
- `/path/to/output/submissions.csv` (all preds + sequences, normalized IDs)

### Batch over all datasets
```bash
python run_all_datasets.py \
  --train_datasets_dir /path/to/train_datasets \
  --test_datasets_dir /path/to/test_datasets \
  --out_dir /path/to/output \
  --n_jobs 8 \
  --device cpu
```
This loops pairs from `get_dataset_pairs`, runs the single-dataset main per pair, then concatenates to `submissions.csv`.

### Docker usage
- Build locally: `docker build -t predict-airr .`
OR
- Pull published image: `docker pull sajayr/predict-airr:latest`
- Run on a single train/test pair (mount your data and output):
  ```bash
  docker run --rm \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    sajayr/predict-airr:latest \
    --train_dir /data/train_datasets/train_dataset_1 \
    --test_dir /data/test_datasets/test_dataset_1 \
    --out_dir /output \
    --n_jobs 4 \
    --device cpu
  ```
- Batch script inside the container:
  ```bash
  docker run --rm \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    sajayr/predict-airr:latest \
    python run_all_datasets.py \
      --train_datasets_dir /data/train_datasets \
      --test_datasets_dir /data/test_datasets \
      --out_dir /output \
      --n_jobs 4 \
      --device cpu
  ```

### Programmatic use
```python
from submission.main import main
from submission.utils import get_dataset_pairs

pairs = get_dataset_pairs("/path/to/train_datasets", "/path/to/test_datasets")
for train_dir, test_dirs in pairs:
    if not test_dirs:
        continue
    main(train_dir=train_dir, test_dirs=test_dirs, out_dir="/path/to/output", n_jobs=8, device="cpu")
```
