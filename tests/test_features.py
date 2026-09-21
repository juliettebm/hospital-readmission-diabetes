import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.readmission.features import FeatureSchemaError, build_dims


RAW_COLUMNS = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
]


@pytest.fixture
def params():
    scaler = MinMaxScaler().fit(
        pd.DataFrame([[0] * len(RAW_COLUMNS), [10] * len(RAW_COLUMNS)], columns=RAW_COLUMNS)
    )
    return {
        "scaler": scaler,
        "cols_instab": RAW_COLUMNS[:3],
        "w_instab": np.array([0.5, 0.3, 0.2]),
        "cols_sev": RAW_COLUMNS[3:6],
        "w_sev": np.array([0.4, 0.35, 0.25]),
    }


def test_build_dims_applies_scaling_and_weights(params):
    raw = dict(zip(RAW_COLUMNS, [2, 4, 6, 8, 10, 0, 5]))

    result = build_dims(raw, params)

    assert list(result.columns) == ["dim_terrain", "dim_instability", "dim_severity"]
    assert result.loc[0, "dim_terrain"] == pytest.approx(0.5)
    assert result.loc[0, "dim_instability"] == pytest.approx(0.34)
    assert result.loc[0, "dim_severity"] == pytest.approx(0.67)


def test_build_dims_clips_values_outside_training_range(params):
    raw = dict.fromkeys(RAW_COLUMNS, 20)

    result = build_dims(raw, params, clip=True)

    assert result.iloc[0].to_dict() == {
        "dim_terrain": 1.0,
        "dim_instability": 1.0,
        "dim_severity": 1.0,
    }


def test_build_dims_supports_notebook_batches_and_preserves_index(params):
    raw = pd.DataFrame(
        [[0] * len(RAW_COLUMNS), [10] * len(RAW_COLUMNS)],
        columns=RAW_COLUMNS,
        index=["train-row", "test-row"],
    )

    result = build_dims(raw, params)

    assert result.index.tolist() == ["train-row", "test-row"]
    assert result.loc["train-row"].tolist() == [0.0, 0.0, 0.0]
    assert result.loc["test-row"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_build_dims_rejects_missing_raw_feature(params):
    raw = dict.fromkeys(RAW_COLUMNS, 1)
    del raw["number_diagnoses"]

    with pytest.raises(FeatureSchemaError, match="number_diagnoses"):
        build_dims(raw, params)
