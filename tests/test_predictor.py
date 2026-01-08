import pytest
import os
import shutil
import pandas as pd
from submission.predictor import ImmuneStatePredictor


@pytest.fixture
def test_environment():
    """
    A pytest fixture to set up a temporary environment with mock data.
    This runs before each test that uses it and cleans up afterward.
    """
    temp_dir = "temp_pytest_data"
    train_data_dir = os.path.join(temp_dir, "train_data")
    test_data_dir = os.path.join(temp_dir, "test_data")
    out_dir = os.path.join(temp_dir, "output")

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    meta_rows = []
    for i in range(6):
        rep_id = f"rep{i}"
        fname = f"{rep_id}.tsv"
        label = 1 if i % 2 == 0 else 0
        df = pd.DataFrame({
            "junction_aa": ["AAAAA", "AAAAB", "AAAAC", "AAAAD"],
            "v_call": ["V1"] * 4,
            "j_call": ["J1"] * 4,
        })
        df.to_csv(os.path.join(train_data_dir, fname), sep="\t", index=False)
        meta_rows.append({"repertoire_id": rep_id, "filename": fname, "label_positive": label})
    pd.DataFrame(meta_rows).to_csv(os.path.join(train_data_dir, "metadata.csv"), index=False)

    for i in range(2):
        rep_id = f"test{i}"
        df = pd.DataFrame({
            "junction_aa": ["AAAAA", "AAAAB"],
            "v_call": ["V2"] * 2,
            "j_call": ["J2"] * 2,
        })
        df.to_csv(os.path.join(test_data_dir, f"{rep_id}.tsv"), sep="\t", index=False)

    yield {"train_dir": train_data_dir, "test_dir": test_data_dir, "out_dir": out_dir}

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_initialization():
    """Test that the predictor initializes with default values."""
    predictor = ImmuneStatePredictor()
    assert predictor.model is None
    assert predictor.important_sequences_ is None
    assert predictor.n_jobs >= 1


def test_fit_changes_internal_state(test_environment):
    """Test that fit() trains a model and modifies the instance state."""
    predictor = ImmuneStatePredictor(n_jobs=1, enable_seq_family=False, c_values=[0.5, 0.1], cv_folds=2)
    assert predictor.model is None, "Model should be None before fitting."

    fit_return = predictor.fit(train_dir=test_environment["train_dir"])
    assert isinstance(fit_return, ImmuneStatePredictor), "fit() should return self."
    assert predictor.model is not None, "Model should not be None after fitting."


def test_predict_proba_raises_error_before_fit(test_environment):
    """Test that predict_proba fails if the model hasn't been fitted."""
    predictor = ImmuneStatePredictor()
    with pytest.raises(RuntimeError):
        predictor.predict_proba(test_dir=test_environment["test_dir"])


def test_predict_proba_returns_correct_format_after_fit(test_environment):
    """Test the output format of predict_proba after fitting."""
    predictor = ImmuneStatePredictor(n_jobs=1, enable_seq_family=False, c_values=[0.5], cv_folds=2)
    predictor.fit(train_dir=test_environment["train_dir"])

    predictions_df = predictor.predict_proba(test_dir=test_environment["test_dir"])
    assert isinstance(predictions_df, pd.DataFrame), "Output should be a pandas DataFrame."

    expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    assert list(predictions_df.columns) == expected_cols

    assert len(predictions_df) == 2, "Output should have one row per test sample."

    assert pd.api.types.is_numeric_dtype(predictions_df['label_positive_probability'])
    assert (predictions_df['label_positive_probability'] >= 0).all()
    assert (predictions_df['label_positive_probability'] <= 1).all()

    assert (predictions_df['junction_aa'] == -999.0).all(), "All junction_aa values should be -999.0"
    assert (predictions_df['v_call'] == -999.0).all(), "All v_call values should be -999.0"
    assert (predictions_df['j_call'] == -999.0).all(), "All j_call values should be -999.0"


def test_identify_associated_sequences_returns_correct_format(test_environment):
    """Test the output format of identify_associated_sequences."""
    predictor = ImmuneStatePredictor(n_jobs=1, enable_seq_family=False, c_values=[0.5], cv_folds=2)
    predictor.fit(train_dir=test_environment["train_dir"])

    seq_df = predictor.important_sequences_
    assert isinstance(seq_df, pd.DataFrame)

    expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    assert list(seq_df.columns) == expected_cols
    assert not seq_df.empty
    assert (seq_df['label_positive_probability'] == -999.0).all(), "All label_positive_probability values should be -999.0"
