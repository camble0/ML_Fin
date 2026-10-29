import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.household_interpretation import (
    interpret_prediction,
    load_priority_code_labels,
    load_geographic_code_pairs,
    load_value_label_maps,
    load_variable_descriptions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
PUBLIC_REPORT_PATH = PROJECT_ROOT / "data" / "public" / "analytical_report.json"
FEATURE_VALUE_LABEL_VARIABLES = {
    "province_code": "province",
    "district_code": "district",
    "urban_rural_code": "ur",
    "consumption_quintile": "quintile",
    "dwelling_type_code": "s5aq2",
    "dwelling_owner_code": "s5aq7",
    "owns_residential_house": "s5bq6",
    "drinking_water_source_code": "s5cq3",
    "electricity_grid_connected": "s5cq14",
    "internet_at_home": "s5cq20a",
    "toilet_type_code": "s5cq25",
    "any_livestock": "s7aq4",
}


def category_value_key(value) -> str:
    numeric_value = float(value)
    return str(int(numeric_value)) if numeric_value.is_integer() else str(value)


def format_category_label(feature: str, value, resources: dict) -> str:
    source_variable = FEATURE_VALUE_LABEL_VARIABLES.get(feature)
    code = category_value_key(value)
    label = resources["value_labels"].get(source_variable or "", {}).get(code)
    if not label:
        raise ValueError(
            f"No verified survey label for {feature} (source {source_variable}, code {code})."
        )
    return label


def load_ui_resources(model_name: str = "random_forest") -> dict:
    if not PUBLIC_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Published analytical report is missing: {PUBLIC_REPORT_PATH}")
    report = json.loads(PUBLIC_REPORT_PATH.read_text(encoding="utf-8"))
    evaluation = report.get("baseline_model_evaluation", {})
    artifacts = evaluation.get("model_artifacts", {})
    if model_name not in artifacts:
        raise FileNotFoundError(f"Saved model artifact is not registered: {model_name}")

    model_path = PROJECT_ROOT / artifacts[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model artifact is missing: {model_path}")
    model = joblib.load(model_path)

    reference_path = PROJECT_ROOT / evaluation.get("training_reference_profile", "")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Training reference profile is missing: {reference_path}")
    reference_profile = json.loads(reference_path.read_text(encoding="utf-8"))

    features = report.get("model_predictor_columns", [])
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Saved model pipeline has no fitted preprocessor.")
    expected = []
    for _, _, columns in preprocessor.transformers:
        expected.extend(columns)
    if set(expected) != set(features) or len(expected) != len(features):
        raise ValueError("Saved model feature schema does not match the analytical report.")

    fitted_categories = preprocessor.named_transformers_["categorical"].named_steps[
        "onehot"
    ].categories_
    categorical_features = report["categorical_predictors"]
    category_options = {
        feature: [value.item() if hasattr(value, "item") else value for value in values]
        for feature, values in zip(categorical_features, fitted_categories)
    }
    value_labels = load_value_label_maps()
    missing_labels = [
        f"{feature}={category_value_key(value)}"
        for feature in categorical_features
        for value in category_options[feature]
        if not value_labels.get(FEATURE_VALUE_LABEL_VARIABLES.get(feature, ""), {}).get(
            category_value_key(value)
        )
    ]
    if missing_labels:
        raise ValueError(
            "Missing verified survey labels for model categories: "
            + ", ".join(missing_labels)
        )
    return {
        "model": model,
        "model_name": model_name,
        "report": report,
        "evaluation": evaluation,
        "reference_values": reference_profile["values"],
        "features": features,
        "categorical_features": categorical_features,
        "numeric_features": report["numeric_predictors"],
        "category_options": category_options,
        "value_labels": value_labels,
        "priority_labels": load_priority_code_labels(),
        "geographic_code_pairs": load_geographic_code_pairs(),
        "variable_descriptions": load_variable_descriptions(),
    }


def validate_inputs(inputs: dict, resources: dict) -> pd.DataFrame:
    features = resources["features"]
    missing = sorted(set(features) - set(inputs))
    if missing:
        raise ValueError(f"Missing required household characteristics: {missing}")

    row = {}
    categorical = set(resources["categorical_features"])
    for feature in features:
        value = inputs[feature]
        if value is None:
            raise ValueError(f"Enter a value for {feature.replace('_', ' ')}.")
        if feature in categorical:
            options = resources["category_options"][feature]
            if value not in options:
                raise ValueError(f"Select a supported value for {feature.replace('_', ' ')}.")
            row[feature] = value
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Enter a numeric value for {feature.replace('_', ' ')}.") from error
        if not np.isfinite(numeric_value):
            raise ValueError(f"Enter a finite value for {feature.replace('_', ' ')}.")
        if feature.startswith("share_") or feature.endswith("_share"):
            if not 0 <= numeric_value <= 1:
                raise ValueError(f"{feature.replace('_', ' ').capitalize()} must be between 0 and 1.")
        if feature == "resident_member_count" and (
            numeric_value < 1 or not numeric_value.is_integer()
        ):
            raise ValueError("Household size must be a positive whole number.")
        if feature == "mean_resident_age_years" and not 0 <= numeric_value <= 120:
            raise ValueError("Mean resident age must be between 0 and 120 years.")
        if feature == "access_median_travel_time_minutes" and not 0 <= numeric_value <= 1440:
            raise ValueError("Service travel time must be between 0 and 1,440 minutes.")
        if feature in {"household_any_loan_attempt", "household_any_member_owes_money"}:
            if numeric_value not in {0, 1}:
                raise ValueError(f"{feature.replace('_', ' ').capitalize()} must be No or Yes.")
        elif feature not in {"resident_member_count", "mean_resident_age_years", "access_median_travel_time_minutes"} and numeric_value < 0:
            raise ValueError(f"{feature.replace('_', ' ').capitalize()} cannot be negative.")
        row[feature] = numeric_value

    if {"province_code", "district_code"}.issubset(row):
        province_key = category_value_key(row["province_code"])
        district_code = int(float(row["district_code"]))
        supported_districts = resources["geographic_code_pairs"].get(province_key, [])
        if district_code not in supported_districts:
            raise ValueError(
                "The selected district does not belong to the selected province or city."
            )

    return pd.DataFrame([row], columns=features)


def predict_household(
    inputs: dict,
    household_id: str,
    resources: dict,
) -> dict:
    features = validate_inputs(inputs, resources)
    try:
        interpreter_id = float(household_id)
    except (TypeError, ValueError):
        interpreter_id = 0
    household_row = pd.Series({"hhid": interpreter_id, **features.iloc[0].to_dict()})
    try:
        result = interpret_prediction(
            model=resources["model"],
            household_row=household_row,
            reference_values=resources["reference_values"],
            dataset_report=resources["report"],
            variable_descriptions=resources["variable_descriptions"],
            stata_value_labels=resources["value_labels"],
            priority_value_labels=resources["priority_labels"],
            model_name=resources["model_name"],
        )
        result["household_id"] = household_id
        return result
    except Exception:
        logger.exception("Household prediction or interpretation failed")
        raise
