import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_notebook_match_recorded_metrics():
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notebook = (ROOT / "notebooks" / "03_modelling.ipynb").read_text(encoding="utf-8")

    expected_fragments = {
        "baseline_auc": f"{metrics['baseline_auc']:.3f}",
        "cv_auc": f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}",
        "test_auc": f"{metrics['test_auc']:.3f}",
        "train_encounters": f"{metrics['train_encounters']:,}",
        "test_encounters": f"{metrics['test_encounters']:,}",
        "train_patients": f"{metrics['train_patients']:,}",
        "test_patients": f"{metrics['test_patients']:,}",
    }
    for label, value in expected_fragments.items():
        assert value in readme, f"README is missing recorded {label}: {value}"

    for label in ("baseline_auc", "cv_auc_mean", "test_auc", "sensitivity", "specificity"):
        value = f"{metrics[label]:.3f}"
        assert value in notebook, f"Notebook is missing recorded {label}: {value}"


def test_recorded_target_is_explicit_in_readme():
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert metrics["target"] == "any readmission (<30 or >30 days)"
    assert "**Target definition.**" in readme
    assert "not the standard 30-day outcome" in readme
