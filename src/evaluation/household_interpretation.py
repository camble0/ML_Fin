import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "financial_inclusion_household.csv"
DATASET_REPORT_PATH = PROJECT_ROOT / "reports" / "analytical_dataset_report.json"
INVENTORY_PATH = PROJECT_ROOT / "data" / "processed" / "eicv7_metadata_inventory.json"
STRUCTURE_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "eicv7_priority_structure.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "financial_inclusion_interpretation_demo.json"

PROFILE_FIELDS = [
    ("household_size", "resident_member_count", None, "resident members"),
    ("mean_resident_age_years", "mean_resident_age_years", None, "years"),
    ("share_female_residents", "share_female_residents", None, "fraction"),
    ("working_age_share", "share_residents_age_15_64", None, "fraction"),
    ("province", "province_code", "province", "category"),
    ("residence_area", "urban_rural_code", "ur", "category"),
    ("consumption_quintile", "consumption_quintile", "quintile", "category"),
    ("electricity_grid_connected", "electricity_grid_connected", "s5cq14", "category"),
    ("internet_at_home", "internet_at_home", "s5cq20a", "category"),
    ("dwelling_type", "dwelling_type_code", "s5aq2", "category"),
    ("drinking_water_source", "drinking_water_source_code", "s5cq3", "category"),
]

FEATURE_SOURCE_VARIABLES = {
    "province_code": ["province"],
    "district_code": ["district"],
    "urban_rural_code": ["ur"],
    "consumption_quintile": ["quintile"],
    "dwelling_type_code": ["s5aq2"],
    "dwelling_owner_code": ["s5aq7"],
    "owns_residential_house": ["s5bq6"],
    "drinking_water_source_code": ["s5cq3"],
    "electricity_grid_connected": ["s5cq14"],
    "internet_at_home": ["s5cq20a"],
    "toilet_type_code": ["s5cq25"],
    "any_livestock": ["s7aq4"],
    "resident_member_count": ["pid"],
    "mean_resident_age_years": ["s1q3y"],
    "share_female_residents": ["s1q1"],
    "share_residents_age_15_64": ["s1q3y"],
    "share_residents_ever_attended_school": ["s4aq1"],
    "share_residents_with_mobile_phone": ["s4bq8"],
    "share_residents_using_internet": ["s4bq15"],
    "share_residents_worked_own_farm": ["s6aq2"],
    "share_residents_worked_nonfarm_wage": ["s6aq5"],
    "share_residents_running_nonfarm_business": ["s6aq6"],
    "access_service_use_often_share": ["s5fq1"],
    "access_service_satisfied_share": ["s5fq6"],
    "access_service_improving_share": ["s5fq7"],
    "access_median_travel_time_minutes": ["s5fq3a", "s5fq3b"],
    "household_any_loan_attempt": ["s10aq1"],
    "household_any_member_owes_money": ["s10aq3"],
}

FEATURE_SOURCE_FILES = {
    "province_code": "CS_S01_S5_S7_Household.dta",
    "district_code": "CS_S01_S5_S7_Household.dta",
    "urban_rural_code": "CS_S01_S5_S7_Household.dta",
    "consumption_quintile": "CS_S01_S5_S7_Household.dta",
    "dwelling_type_code": "CS_S01_S5_S7_Household.dta",
    "dwelling_owner_code": "CS_S01_S5_S7_Household.dta",
    "owns_residential_house": "CS_S01_S5_S7_Household.dta",
    "drinking_water_source_code": "CS_S01_S5_S7_Household.dta",
    "electricity_grid_connected": "CS_S01_S5_S7_Household.dta",
    "internet_at_home": "CS_S01_S5_S7_Household.dta",
    "toilet_type_code": "CS_S01_S5_S7_Household.dta",
    "any_livestock": "CS_S01_S5_S7_Household.dta",
    "resident_member_count": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "mean_resident_age_years": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_female_residents": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_age_15_64": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_ever_attended_school": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_with_mobile_phone": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_using_internet": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_worked_own_farm": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_worked_nonfarm_wage": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "share_residents_running_nonfarm_business": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
    "access_service_use_often_share": "CS_S5F_access_to_services.dta",
    "access_service_satisfied_share": "CS_S5F_access_to_services.dta",
    "access_service_improving_share": "CS_S5F_access_to_services.dta",
    "access_median_travel_time_minutes": "CS_S5F_access_to_services.dta",
    "household_any_loan_attempt": "CS_S10A1_A2_credits.dta",
    "household_any_member_owes_money": "CS_S10A1_A2_credits.dta",
}

FEATURE_TRANSFORMATIONS = {
    "province_code": "Retained as a categorical survey code; one-hot encoded by the model.",
    "district_code": "Retained as a categorical survey code; one-hot encoded by the model.",
    "urban_rural_code": "Retained as a categorical survey code; one-hot encoded by the model.",
    "consumption_quintile": "Retained as the survey's consumption-quintile category; one-hot encoded by the model.",
    "dwelling_type_code": "Retained as the dwelling-type category; one-hot encoded by the model.",
    "dwelling_owner_code": "Retained as the dwelling-owner category; one-hot encoded by the model.",
    "owns_residential_house": "Retained as the Yes/No survey response category; one-hot encoded by the model.",
    "drinking_water_source_code": "Retained as the main drinking-water-source category; one-hot encoded by the model.",
    "electricity_grid_connected": "Retained as the Yes/No grid-connection response; one-hot encoded by the model.",
    "internet_at_home": "Retained as the Yes/No household-internet response; one-hot encoded by the model.",
    "toilet_type_code": "Retained as the household toilet-type category; one-hot encoded by the model.",
    "any_livestock": "Retained as the household livestock response category; one-hot encoded by the model.",
    "resident_member_count": "Counts nonmissing person IDs among roster records marked as household members.",
    "mean_resident_age_years": "Mean of nonmissing age-in-years values among roster records marked as household members.",
    "share_female_residents": "Female-coded (2) responses divided by observed Male/Female responses among resident members.",
    "share_residents_age_15_64": "Resident members aged 15-64 divided by resident members with age observed.",
    "share_residents_ever_attended_school": "Yes responses divided by observed Yes/No responses among resident members.",
    "share_residents_with_mobile_phone": "Yes responses divided by observed Yes/No responses among resident members.",
    "share_residents_using_internet": "Yes responses divided by observed Yes/No responses among resident members.",
    "share_residents_worked_own_farm": "Yes responses divided by observed Yes/No responses among resident members.",
    "share_residents_worked_nonfarm_wage": "Yes responses divided by observed Yes/No responses among resident members.",
    "share_residents_running_nonfarm_business": "Yes responses divided by observed Yes/No responses among resident members.",
    "access_service_use_often_share": "Often or Regularly service responses divided by observed Not at all/Sometimes/Often/Regularly responses.",
    "access_service_satisfied_share": "Satisfied responses divided by observed Satisfied/Not satisfied responses across service items.",
    "access_service_improving_share": "Improving responses divided by observed trend codes across service items.",
    "access_median_travel_time_minutes": "Converts hours to minutes, adds the reported minutes where both parts exist, then takes the household median.",
    "household_any_loan_attempt": "1 if any valid household loan-attempt response is Yes; 0 if valid responses exist and none is Yes; otherwise missing.",
    "household_any_member_owes_money": "1 if any valid household debt response is Yes; 0 if valid responses exist and none is Yes; otherwise missing.",
}


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def format_household_id(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def value_key(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def load_value_label_maps() -> dict[str, dict[str, str]]:
    return {}


def load_variable_descriptions() -> dict[tuple[str, str], str]:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    return {
        (file_entry["filename"], variable["name"]): variable.get("label") or variable["name"]
        for file_entry in inventory["files"]
        for variable in file_entry.get("variables", [])
    }


def build_feature_manifest(report: dict, variable_descriptions: dict) -> dict:
    manifest = {}
    for group in report["feature_transformations"]:
        for transformation in group["transformations"]:
            for feature in transformation["resulting_features"]:
                source_file = FEATURE_SOURCE_FILES.get(feature, group["source_file"])
                source_variables = FEATURE_SOURCE_VARIABLES.get(
                    feature, transformation["original_variables"]
                )
                source_labels = [
                    variable_descriptions.get((source_file, variable), variable)
                    for variable in source_variables
                ]
                manifest[feature] = {
                    "source_file": source_file,
                    "source_variables": source_variables,
                    "source_variable_labels": source_labels,
                    "transformation": FEATURE_TRANSFORMATIONS.get(
                        feature, transformation["transformation"]
                    ),
                    "description": (
                        f"Derived from {', '.join(source_labels)}. "
                        f"{FEATURE_TRANSFORMATIONS.get(feature, transformation['transformation'])}"
                    ),
                }
    missing = set(report["model_predictor_columns"]) - set(manifest)
    if missing:
        raise ValueError(f"Missing feature documentation: {sorted(missing)}")
    return manifest


def load_priority_code_labels() -> dict[str, dict[str, str]]:
    report = json.loads(STRUCTURE_REPORT_PATH.read_text(encoding="utf-8"))
    return report.get("code_labels", {})


def describe_profile_value(
    row: pd.Series,
    feature: str,
    source_variable: str | None,
    unit: str,
    stata_value_labels: dict[str, dict[str, str]],
    priority_value_labels: dict[str, dict[str, str]],
) -> dict:
    value = json_value(row[feature])
    if value is None:
        return {"value": None, "display": "Not reported", "unit": unit}
    if unit == "category":
        label = (
            priority_value_labels.get(source_variable or "", {}).get(value_key(value))
            or stata_value_labels.get(source_variable or "", {}).get(value_key(value))
            or f"Code {value_key(value)}"
        )
        return {"value": value, "display": label, "unit": unit}
    if unit == "fraction":
        return {"value": float(value), "display": f"{float(value):.0%}", "unit": unit}
    if unit == "years":
        return {"value": round(float(value), 1), "display": f"{float(value):.1f} years", "unit": unit}
    return {"value": int(value), "display": f"{int(value)} {unit}", "unit": unit}


def make_household_profile(
    row: pd.Series,
    stata_value_labels: dict[str, dict[str, str]],
    priority_value_labels: dict[str, dict[str, str]],
) -> dict:
    return {
        profile_name: describe_profile_value(
            row,
            feature,
            source_variable,
            unit,
            stata_value_labels,
            priority_value_labels,
        )
        for profile_name, feature, source_variable, unit in PROFILE_FIELDS
    }


def local_ablation_factors(
    model,
    feature_row: pd.DataFrame,
    reference_values: dict,
    feature_manifest: dict,
    stata_value_labels: dict[str, dict[str, str]],
    top_k: int = 5,
) -> list[dict]:
    classes = list(model.classes_)
    positive_index = classes.index(1)
    original_probability = float(model.predict_proba(feature_row)[0, positive_index])
    factors = []

    for feature in feature_row.columns:
        reference_value = reference_values.get(feature)
        if reference_value is None:
            continue
        perturbed = feature_row.copy()
        perturbed.loc[perturbed.index[0], feature] = reference_value
        changed_probability = float(model.predict_proba(perturbed)[0, positive_index])
        delta = original_probability - changed_probability
        if np.isclose(delta, 0.0, atol=1e-9):
            continue

        feature_info = feature_manifest[feature]
        source_variables = feature_info["source_variables"]
        source_variable = source_variables[0] if len(source_variables) == 1 else None
        observed_value = json_value(feature_row.iloc[0][feature])
        observed_label = None
        reference_label = None
        if source_variable and feature in {
            "province_code",
            "district_code",
            "urban_rural_code",
            "consumption_quintile",
            "dwelling_type_code",
            "dwelling_owner_code",
            "owns_residential_house",
            "drinking_water_source_code",
            "electricity_grid_connected",
            "internet_at_home",
            "toilet_type_code",
            "any_livestock",
        }:
            observed_label = stata_value_labels.get(source_variable, {}).get(
                value_key(observed_value)
            ) if observed_value is not None else None
            reference_label = stata_value_labels.get(source_variable, {}).get(
                value_key(reference_value)
            )

        factors.append(
            {
                "feature_name": feature,
                "feature_description": feature_info["description"],
                "source_file": feature_info["source_file"],
                "source_variables": source_variables,
                "source_variable_labels": feature_info["source_variable_labels"],
                "observed_value": observed_value,
                "observed_value_label": observed_label,
                "reference_value": json_value(reference_value),
                "reference_value_label": reference_label,
                "local_ablation_probability_delta": float(delta),
                "direction": (
                    "raises_positive_class_probability"
                    if delta > 0
                    else "lowers_positive_class_probability"
                ),
                "method": "One-feature-at-a-time replacement with the training-fold reference value.",
            }
        )

    factors.sort(
        key=lambda factor: abs(factor["local_ablation_probability_delta"]),
        reverse=True,
    )
    return factors[:top_k]


def interpret_prediction(
    model,
    household_row: pd.Series,
    reference_values: dict,
    dataset_report: dict,
    variable_descriptions: dict,
    stata_value_labels: dict[str, dict[str, str]],
    priority_value_labels: dict[str, dict[str, str]],
    model_name: str,
    top_k: int = 5,
) -> dict:
    features = dataset_report["model_predictor_columns"]
    feature_row = pd.DataFrame(
        [{feature: household_row[feature] for feature in features}],
        columns=features,
    )
    probabilities = model.predict_proba(feature_row)[0]
    classes = list(model.classes_)
    class_probabilities = {
        str(int(class_value)): float(probability)
        for class_value, probability in zip(classes, probabilities)
    }
    positive_probability = class_probabilities["1"]
    negative_probability = class_probabilities["0"]
    predicted_class = int(model.predict(feature_row)[0])
    predicted_probability = class_probabilities[str(predicted_class)]
    confidence_score = abs(positive_probability - negative_probability)

    feature_manifest = build_feature_manifest(
        dataset_report, variable_descriptions
    )
    key_factors = local_ablation_factors(
        model,
        feature_row,
        reference_values,
        feature_manifest,
        stata_value_labels,
        top_k=top_k,
    )
    profile = make_household_profile(
        household_row,
        stata_value_labels,
        priority_value_labels,
    )
    class_label = (
        "At least one household member reported a bank or Momo account"
        if predicted_class == 1
        else "No household member reported a bank or Momo account"
    )
    observed_profile = ", ".join(
        f"{name.replace('_', ' ')}: {value['display']}"
        for name, value in list(profile.items())[:5]
    )
    interpretation = (
        f"The {model_name.replace('_', ' ')} model predicts: {class_label.lower()} "
        f"(predicted-class probability {predicted_probability:.1%}). "
        f"Selected survey profile: {observed_profile}. These profile entries are "
        "survey-derived observations; the class is a model prediction, not an "
        "observed fact or a causal conclusion."
    )

    return {
        "schema_version": "1.0",
        "household_id": format_household_id(household_row["hhid"]),
        "model_name": model_name,
        "prediction": {
            "target": dataset_report["target"]["name"],
            "predicted_class": predicted_class,
            "predicted_label": class_label,
            "probability": predicted_probability,
            "probabilities": {
                "no_bank_or_momo_account": negative_probability,
                "has_bank_or_momo_account": positive_probability,
            },
        },
        "confidence": {
            "score": float(confidence_score),
            "definition": "Absolute probability margin |P(class 1) - P(class 0)|, ranging from 0 (tie) to 1 (maximum margin).",
            "calibration_status": "Not assessed; this score is not a calibrated probability of correctness or certainty.",
        },
        "key_factors": key_factors,
        "key_factor_method": {
            "name": "Model-agnostic one-feature-at-a-time reference ablation",
            "definition": "For each feature, replace only this household's value with its training-fold median (numeric) or mode (categorical), then report the change in the fitted model's class-1 probability relative to the original row.",
            "interpretation": "Positive delta means the observed feature value raises the model's class-1 score relative to that reference replacement; negative delta means it lowers it. Effects are not additive and are not causal contributions.",
        },
        "household_profile": profile,
        "interpretation": interpretation,
        "limitations": [
            "This is a prediction from cross-sectional Rwanda survey data; it does not establish that any characteristic causes financial inclusion.",
            "Model probabilities have not been calibrated or externally validated. The confidence score is only a class-probability margin, not certainty.",
            "Local ablation is model-specific sensitivity, not an additive attribution. Correlated features and interactions can change the result; replacing one value with a training-fold reference may create an uncommon feature combination.",
            "The target includes a documented structural-No rule based on s10cq1 for households with no s10cq4 response; confirm questionnaire skip logic.",
            "Survey weights were not used in model fitting, and clust is treated as a cluster candidate for splitting pending design-documentation confirmation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interpretable prediction payload for a held-out household."
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="random_forest",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    dataset_report = json.loads(DATASET_REPORT_PATH.read_text(encoding="utf-8"))
    baseline_report = dataset_report["baseline_model_evaluation"]
    model_path = PROJECT_ROOT / baseline_report["model_artifacts"][args.model]
    model = joblib.load(model_path)
    reference_profile = json.loads(
        (PROJECT_ROOT / baseline_report["training_reference_profile"]).read_text(
            encoding="utf-8"
        )
    )
    dataset = pd.read_csv(DATA_PATH)
    demo_id = baseline_report["interpretation_demo_household_id"]
    selected = dataset.loc[
        dataset["hhid"].map(format_household_id).eq(demo_id)
    ]
    if len(selected) != 1:
        raise ValueError("The selected held-out demo household was not found uniquely.")
    household = selected.iloc[0]

    variable_descriptions = load_variable_descriptions()
    stata_value_labels = load_value_label_maps()
    priority_value_labels = load_priority_code_labels()
    payload = interpret_prediction(
        model=model,
        household_row=household,
        reference_values=reference_profile["values"],
        dataset_report=dataset_report,
        variable_descriptions=variable_descriptions,
        stata_value_labels=stata_value_labels,
        priority_value_labels=priority_value_labels,
        model_name=args.model,
    )
    payload["demo_context"] = {
        "selected_from": "The deterministic held-out test fold recorded by the baseline runner.",
        "observed_target_for_demo_only": {
            "class": json_value(household[dataset_report["target"]["name"]]),
            "source_status": str(household["target_source_status"]),
            "not_part_of_prediction_inputs": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Model: {args.model}")
    print(f"Household: {payload['household_id']}")
    print(f"Prediction: {payload['prediction']['predicted_label']}")
    print(f"Probability: {payload['prediction']['probability']:.4f}")
    print(f"Confidence margin: {payload['confidence']['score']:.4f}")
    print(f"Interpretation JSON: {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
