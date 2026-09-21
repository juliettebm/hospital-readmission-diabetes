"""Load and validate the serialized inference artifacts."""

from pathlib import Path

import joblib
import numpy as np

from .features import DIMENSION_COLUMNS, FeatureSchemaError, _validate_params


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a serialized artifact is missing or incompatible."""


def _validate_model(model: object) -> None:
    if not hasattr(model, "predict_proba"):
        raise ArtifactCompatibilityError("The model does not expose predict_proba().")
    features = tuple(getattr(model, "feature_names_in_", ()))
    if features != DIMENSION_COLUMNS:
        raise ArtifactCompatibilityError(
            f"Unexpected model features: {features!r}; expected {DIMENSION_COLUMNS!r}."
        )
    classes = tuple(getattr(model, "classes_", ()))
    if classes != (0, 1):
        raise ArtifactCompatibilityError(
            f"Unexpected model classes: {classes!r}; expected binary classes (0, 1)."
        )


def load_artifacts(models_dir: str | Path) -> tuple[object, dict[str, object]]:
    """Load artifacts and fail with an actionable compatibility error."""

    models_path = Path(models_dir)
    model_path = models_path / "readmission_model.pkl"
    params_path = models_path / "feature_params.pkl"
    missing = [str(path) for path in (model_path, params_path) if not path.is_file()]
    if missing:
        raise ArtifactCompatibilityError(f"Missing artifact(s): {', '.join(missing)}")

    try:
        model = joblib.load(model_path)
        params = joblib.load(params_path)
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "Unable to deserialize the artifacts. Install the exact versions from "
            f"requirements.txt. Original error: {exc}"
        ) from exc

    if not isinstance(params, dict):
        raise ArtifactCompatibilityError("feature_params.pkl must contain a dictionary.")
    try:
        _validate_params(params)
    except FeatureSchemaError as exc:
        raise ArtifactCompatibilityError(f"Invalid feature parameters: {exc}") from exc
    _validate_model(model)

    if not np.isclose(np.asarray(params["w_instab"], dtype=float).sum(), 1.0):
        raise ArtifactCompatibilityError("Instability weights must sum to 1.")
    if not np.isclose(np.asarray(params["w_sev"], dtype=float).sum(), 1.0):
        raise ArtifactCompatibilityError("Severity weights must sum to 1.")

    return model, params
