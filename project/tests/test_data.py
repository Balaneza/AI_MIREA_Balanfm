from pathlib import Path

from src.aps_failure.data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_aps_csv_loader_reads_original_uci_files():
    x_train, y_train = load_dataset(PROJECT_ROOT / "data" / "raw" / "aps_failure_training_set.csv")

    assert x_train.shape == (60000, 170)
    assert int(y_train.sum()) == 1000
    assert set(y_train.unique()) == {0, 1}
    assert "aa_000" in x_train.columns
