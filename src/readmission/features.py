"""Feature engineering shared by training notebooks and the Streamlit app."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


DIMENSION_COLUMNS = ("dim_terrain", "dim_instability", "dim_severity")
PARAMETER_KEYS = ("scaler", "cols_instab", "w_instab", "cols_sev", "w_sev")


class FeatureSchemaError(ValueError):
    """Raised when raw inputs or feature parameters do not match the schema."""


def _validate_params(params: Mapping[str, object]) -> None:
    missing = [key for key in PARAMETER_KEYS if key not in params]
    if missing:
        raise FeatureSchemaError(f"Missing feature parameter(s): {', '.join(missing)}")

    scaler = params["scaler"]
    if not hasattr(scaler, "transform") or not hasattr(scaler, "feature_names_in_"):
        raise FeatureSchemaError("'scaler' must be a fitted transformer with feature_names_in_")

    for columns_key, weights_key in (("cols_instab", "w_instab"), ("cols_sev", "w_sev")):
        columns = list(params[columns_key])
        weights = np.asarray(params[weights_key], dtype=float)
        if not columns or len(columns) != len(weights):
            raise FeatureSchemaError(
                f"'{columns_key}' and '{weights_key}' must be non-empty and have equal lengths"
            )
        if not np.isfinite(weights).all():
            raise FeatureSchemaError(f"'{weights_key}' must contain only finite values")


def build_dims(
    raw: Mapping[str, float] | pd.DataFrame,
    params: Mapping[str, object],
    *,
    clip: bool = False,
) -> pd.DataFrame:
    """Build the three model dimensions using the fitted training-set scaler.

    Set ``clip=True`` at interactive inference time to bound values outside the
    scaler's observed training range. Training and evaluation retain the fitted
    scaler's standard extrapolation behaviour by default.
    """

    _validate_params(params)
    scaler = params["scaler"]
    input_columns = list(scaler.feature_names_in_)
    missing = [column for column in input_columns if column not in raw]
    if missing:
        raise FeatureSchemaError(f"Missing raw feature(s): {', '.join(missing)}")

    if isinstance(raw, pd.DataFrame):
        frame = raw.loc[:, input_columns]
    else:
        frame = pd.DataFrame([{column: raw[column] for column in input_columns}])
    try:
        scaled_values = scaler.transform(frame)
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaError(f"Invalid raw feature values: {exc}") from exc

    scaled = pd.DataFrame(scaled_values, columns=input_columns, index=frame.index)
    if clip:
        scaled = scaled.clip(0, 1)

    def weighted_dimension(columns_key: str, weights_key: str) -> pd.Series:
        columns = list(params[columns_key])
        unknown = [column for column in columns if column not in scaled]
        if unknown:
            raise FeatureSchemaError(
                f"Unknown feature(s) in '{columns_key}': {', '.join(unknown)}"
            )
        return scaled.loc[:, columns].dot(np.asarray(params[weights_key], dtype=float))

    return pd.DataFrame(
        {
            "dim_terrain": scaled["number_diagnoses"],
            "dim_instability": weighted_dimension("cols_instab", "w_instab"),
            "dim_severity": weighted_dimension("cols_sev", "w_sev"),
        },
        index=frame.index,
    )
