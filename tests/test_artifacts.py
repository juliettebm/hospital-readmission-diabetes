from pathlib import Path

import numpy as np
import pytest

from src.readmission.artifacts import ArtifactCompatibilityError, load_artifacts
from src.readmission.features import build_dims


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_artifacts_load_and_predict():
    model, params = load_artifacts(ROOT / "models")
    raw = {
        "number_diagnoses": 5,
        "number_inpatient": 0,
        "number_emergency": 0,
        "number_outpatient": 0,
        "time_in_hospital": 3,
        "num_lab_procedures": 35,
        "num_medications": 15,
    }

    dimensions = build_dims(raw, params)
    probabilities = model.predict_proba(dimensions)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_missing_artifacts_have_an_actionable_error():
    with pytest.raises(ArtifactCompatibilityError, match="Missing artifact"):
        load_artifacts(ROOT / "models-that-do-not-exist")
