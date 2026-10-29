import json
import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.ui.financial_inclusion_service import (
    FEATURE_VALUE_LABEL_VARIABLES,
    PROJECT_ROOT,
    category_value_key,
    format_category_label,
    load_ui_resources,
    predict_household,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_fin_ui")

PUBLIC_DATA_PATH = PROJECT_ROOT / "data" / "public" / "household_aggregates.json"
REPORT_PATH = PROJECT_ROOT / "data" / "public" / "analytical_report.json"
MODEL_NAMES = ("random_forest", "logistic_regression")

FEATURE_LABELS = {
    "resident_member_count": "People who live in this household",
    "mean_resident_age_years": "Average age of household members (years)",
    "share_female_residents": "Share of household members who are women",
    "share_residents_age_15_64": "Share of members aged 15 to 64",
    "share_residents_ever_attended_school": "Share who have ever attended school",
    "share_residents_with_mobile_phone": "Share with a mobile phone",
    "share_residents_using_internet": "Share who used the internet",
    "share_residents_worked_own_farm": "Share who worked on their own farm",
    "share_residents_worked_nonfarm_wage": "Share in non-farm wage work",
    "share_residents_running_nonfarm_business": "Share running a non-farm business",
    "province_code": "Province",
    "district_code": "District",
    "urban_rural_code": "Residence area",
    "consumption_quintile": "Consumption quintile",
    "dwelling_type_code": "Dwelling type",
    "dwelling_owner_code": "Dwelling ownership",
    "owns_residential_house": "Does a household member own a residential house?",
    "drinking_water_source_code": "Main drinking-water source",
    "electricity_grid_connected": "Household electricity connection",
    "internet_at_home": "Does this household have an internet connection?",
    "toilet_type_code": "Household toilet type",
    "any_livestock": "Did any member own or raise livestock?",
    "access_service_use_often_share": "Share of services used often or regularly",
    "access_service_satisfied_share": "Share of services household members are satisfied with",
    "access_service_improving_share": "Share of services reported as improving",
    "access_median_travel_time_minutes": "Typical one-way travel time to services (minutes)",
    "household_any_loan_attempt": "Has any member tried to obtain a loan?",
    "household_any_member_owes_money": "Does any household member currently owe money?",
}

FORM_SECTIONS = (
    ("Household composition", (
        "resident_member_count",
        "mean_resident_age_years",
        "share_female_residents",
        "share_residents_age_15_64",
        "share_residents_ever_attended_school",
    )),
    ("Employment and economic activity", (
        "consumption_quintile",
        "share_residents_worked_own_farm",
        "share_residents_worked_nonfarm_wage",
        "share_residents_running_nonfarm_business",
    )),
    ("Housing and basic services", (
        "dwelling_type_code",
        "dwelling_owner_code",
        "owns_residential_house",
        "drinking_water_source_code",
        "electricity_grid_connected",
        "internet_at_home",
        "toilet_type_code",
        "any_livestock",
        "share_residents_with_mobile_phone",
        "share_residents_using_internet",
        "access_service_use_often_share",
        "access_service_satisfied_share",
        "access_service_improving_share",
        "access_median_travel_time_minutes",
    )),
    ("Financial access and borrowing", (
        "household_any_loan_attempt",
        "household_any_member_owes_money",
    )),
)

FEATURE_HELP = {
    "share_female_residents": "A proportion from 0 to 1, calculated from household members with a recorded sex response.",
    "share_residents_age_15_64": "A proportion from 0 to 1 among household members whose age is recorded.",
    "share_residents_ever_attended_school": "A proportion from 0 to 1 among household members with a recorded Yes/No answer.",
    "share_residents_with_mobile_phone": "A proportion from 0 to 1 among household members with a recorded Yes/No answer.",
    "share_residents_using_internet": "A proportion from 0 to 1 among household members with a recorded Yes/No answer.",
    "consumption_quintile": "A quintile divides households ranked by consumption into five equal-sized groups. Q1 is the lowest-consumption group and Q5 the highest.",
    "dwelling_owner_code": "Choose the survey-recorded owner category for the dwelling itself.",
    "owns_residential_house": "This asks whether any household member owns a residential house in this village or elsewhere; it is separate from ownership of the dwelling occupied here.",
    "drinking_water_source_code": "Select the main source household members use for drinking water.",
    "electricity_grid_connected": "The survey distinguishes grid electricity, solar electricity, both, and neither.",
    "toilet_type_code": "Select the toilet type used by the household. VIP means ventilated improved pit latrine.",
    "any_livestock": "Includes owning and/or raising livestock, as asked in the EICV7 questionnaire.",
    "access_service_use_often_share": "Share of recorded services used often or regularly, rather than not at all or sometimes.",
    "access_service_satisfied_share": "Share of recorded services for which household members reported being satisfied.",
    "access_service_improving_share": "Share of recorded services reported as improving.",
    "access_median_travel_time_minutes": "Median one-way travel time to services, converted to minutes; not a distance measure.",
    "household_any_loan_attempt": "Yes if any valid household response said a member tried to obtain a loan.",
    "household_any_member_owes_money": "Yes if any valid household response said a member owes money.",
}

SOURCE_VARIABLES = {
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


st.set_page_config(
    page_title="Rwanda Financial Vulnerability Intelligence",
    page_icon=":material/insights:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink:#172a25; --muted:#60736b; --paper:#f4f7f3; --line:#dce6de; --green:#176b4d; --mint:#e3f1e9; }
    .stApp { background:var(--paper); color:var(--ink); }
    .block-container { max-width:1240px; padding-top:1.5rem; padding-bottom:3rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:0 !important; }
    .eyebrow { color:var(--green); text-transform:uppercase; font-size:.75rem; font-weight:750; }
    .intro { color:var(--muted); max-width:900px; line-height:1.55; }
    @media (max-width:700px) { .block-container { padding-left:1rem; padding-right:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading EICV7 household indicators...")
def load_dataset_and_report() -> tuple[dict[str, pd.DataFrame], dict]:
    if not PUBLIC_DATA_PATH.is_file():
        raise FileNotFoundError(f"Public aggregate data is missing: {PUBLIC_DATA_PATH}")
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Public analytical report is missing: {REPORT_PATH}")
    public_bundle = json.loads(PUBLIC_DATA_PATH.read_text(encoding="utf-8"))
    if public_bundle.get("synthetic"):
        st.warning("Synthetic data is being displayed; these values are not EICV7 survey findings.")
    dimensions = public_bundle.get("dimensions", {})
    frames = {name: pd.DataFrame(rows) for name, rows in dimensions.items()}
    if not frames:
        raise ValueError("Public aggregate data contains no dimensions.")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return frames, report


@st.cache_resource(show_spinner="Loading saved model and feature metadata...")
def load_model_resources(model_name: str) -> dict:
    return load_ui_resources(model_name)


def display_code(value, field: str, report: dict) -> str:
    source = SOURCE_VARIABLES.get(field)
    key = str(int(value)) if pd.notna(value) and float(value).is_integer() else str(value)
    structure_path = PROJECT_ROOT / "data" / "processed" / "eicv7_priority_structure.json"
    if source and structure_path.is_file():
        code_labels = json.loads(structure_path.read_text(encoding="utf-8")).get("code_labels", {})
        if key in code_labels.get(source, {}):
            return code_labels[source][key]
    return f"Unverified {FEATURE_LABELS.get(field, field.replace('_', ' '))} category ({key})"


def account_access_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if frame.empty or group not in frame:
        return pd.DataFrame(columns=[group, "households", "with_account", "without_account", "account_access_share", "no_account_share"])
    summary = frame.copy()
    summary["no_account_share"] = 1 - summary["account_access_share"]
    return summary.sort_values("households", ascending=False).reset_index(drop=True)


def render_overview(report: dict) -> None:
    target = report.get("target", {}).get("distribution", {})
    yes_count = target.get("yes", {}).get("count")
    total = report.get("number_of_households", 0)
    st.header("Overview")
    st.caption("Verified EICV7 household aggregates. Figures below describe this dataset, not current administrative totals.")
    with st.container(horizontal=True):
        st.metric("Households", f"{total:,}", border=True)
        st.metric("Model predictors", f"{report.get('number_of_features', len(report.get('model_predictor_columns', [])))}", border=True)
        account_access_share = target.get("yes", {}).get("fraction_of_households")
        st.metric("Reported account access", f"{account_access_share:.1%}" if account_access_share is not None else "Not available", border=True)
        st.metric("Unresolved target records", f"{target.get('unresolved_missing', {}).get('count', 0):,}", border=True)

    st.subheader("Key findings")
    if yes_count is not None:
        st.write(
            f"{yes_count:,} of {total:,} households ({target['yes']['fraction_of_households']:.1%}) "
            "are recorded as having at least one member with a bank or Mobile Money account. "
            "This is a survey-derived observation; it is not a causal or current-year claim."
        )
    quality = report.get("data_quality", {})
    missing = quality.get("missingness", report.get("missingness_summary", {}))
    predictors = report.get("model_predictor_columns", [])
    missing_rows = [
        {"Predictor": FEATURE_LABELS.get(name, name), "Missing households": missing.get(name, {}).get("missing_count", 0), "Missing share": missing.get(name, {}).get("missing_fraction", 0)}
        for name in predictors if missing.get(name, {}).get("missing_count", 0)
    ]
    left, right = st.columns(2)
    with left:
        st.subheader("Data coverage")
        coverage = report.get("source_files", [])
        if coverage:
            coverage_frame = pd.DataFrame([
                {"Source": item["filename"], "Source rows": item.get("source_rows", 0), "Households without source row": item.get("coverage_against_household_base", {}).get("base_households_without_source_rows", 0)}
                for item in coverage
            ])
            st.dataframe(coverage_frame, hide_index=True, alt="Source-file row counts and household coverage")
        st.caption("EICV7 only. FinScope 2024 is not merged because the survey designs and units are not established as compatible.")
    with right:
        st.subheader("Data quality")
        st.write(f"Duplicate household rows flagged: **{quality.get('duplicate_household_rows', 0):,}** across {quality.get('duplicate_household_id_groups', 0):,} ID groups")
        invalid = quality.get("invalid_record_counts", {})
        st.write(f"Invalid feature values flagged: **{sum(invalid.values()):,}** across {len(invalid)} checks")
        outliers = quality.get("iqr_outlier_rows", {})
        st.write(f"Potential IQR outlier flags: **{sum(outliers.values()):,}** (diagnostic only; values were not removed)")
        if missing_rows:
            st.dataframe(pd.DataFrame(missing_rows).sort_values("Missing households", ascending=False), hide_index=True, alt="Missing values in model predictors")
        else:
            st.success("No missing model predictors recorded in the quality report.")

    evaluation = report.get("baseline_model_evaluation", {}).get("metrics", {})
    if evaluation:
        st.subheader("Held-out model comparison")
        rows = []
        for model, metrics in evaluation.items():
            rows.append({"Model": model.replace("_", " ").title(), "F1": metrics.get("f1_positive_class"), "ROC-AUC": metrics.get("roc_auc"), "Recall": metrics.get("recall_positive_class"), "Test households": metrics.get("test_households")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Actual held-out evaluation metrics for saved models")
        st.caption("Metrics are copied from the saved evaluation report; no evaluation is inferred or fabricated by the interface.")


def render_explore(frames: dict[str, pd.DataFrame], report: dict) -> None:
    st.header("Explore")
    st.caption("Compare observed account-access rates in EICV7 household groups. Small groups can be unstable and should not be ranked as official priorities.")
    group_choices = {
        "province_code": "Province",
        "district_code": "District",
        "urban_rural_code": "Residence area",
        "consumption_quintile": "Consumption quintile",
    }
    grouping = st.selectbox("Compare households by", list(group_choices), format_func=group_choices.get, key="explore_grouping", help="Choose the grouping used in the aggregate chart and downloadable table.")
    summary = account_access_summary(frames.get(grouping, pd.DataFrame()), grouping)
    if summary.empty:
        st.warning("No households with a resolved target match the selected filters.")
        return
    summary["Group"] = summary[grouping].map(lambda value: display_code(value, grouping, report))
    chart = alt.Chart(summary).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("no_account_share:Q", title="Share without reported bank/Mobile Money account", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Group:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("households:Q", title="Households", format=","), alt.Tooltip("with_account:Q", title="With account", format=","), alt.Tooltip("without_account:Q", title="Without account", format=","), alt.Tooltip("no_account_share:Q", title="Without-account share", format=".1%")],
        color=alt.value("#176b4d"),
    ).properties(height=max(180, 34 * min(len(summary), 18)))
    st.altair_chart(chart, width="stretch", alt="Observed share of households without a bank or Mobile Money account by selected group")
    st.dataframe(summary[["Group", "households", "with_account", "without_account", "account_access_share", "no_account_share"]].rename(columns={"households": "Households", "with_account": "With account", "without_account": "Without account", "account_access_share": "Account-access share", "no_account_share": "Without-account share"}), hide_index=True, alt="Aggregate account-access counts and shares")
    st.download_button("Download aggregate comparison CSV", summary.drop(columns=[grouping], errors="ignore").to_csv(index=False), file_name="eicv7_account_access_aggregate.csv", mime="text/csv", help="Exports grouped counts and rates only; no household identifiers or row-level records.")


def render_prediction(report: dict) -> None:
    st.header("Predict")
    st.caption("Model estimates describe associations learned from EICV7. They are not verified household facts, calibrated certainty, eligibility decisions, or causal effects.")
    registered = report.get("baseline_model_evaluation", {}).get("model_artifacts", {})
    available_names = [name for name in MODEL_NAMES if name in registered]
    if not available_names:
        st.error("No trained model is registered. Run the documented model-training command after building the processed dataset.")
        return
    model_name = st.selectbox("Prediction model", available_names, format_func=lambda name: name.replace("_", " ").title(), help="Choose a saved model that has actual held-out evaluation results.")
    try:
        resources = load_model_resources(model_name)
    except Exception as error:
        logger.exception("Unable to load prediction resources")
        st.error("Prediction is unavailable because the selected model or required metadata could not be loaded. Overview and Explore remain available.")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
        return

    st.caption("The model inputs below mirror the saved preprocessing schema. Defaults are the training-fold median (numeric) or mode (categorical), not a representative household.")
    defaults = resources["reference_values"]
    options_by_feature = resources["category_options"]
    geography = resources["geographic_code_pairs"]
    province_options = options_by_feature["province_code"]
    province_key = "predict_province_code"
    previous_province = st.session_state.get(province_key, defaults["province_code"])
    if previous_province not in province_options:
        previous_province = province_options[0]
    with st.container(border=True):
        st.subheader("Geographic information")
        province_column, district_column, residence_column = st.columns(3)
        with province_column:
            province_code = st.selectbox(
                FEATURE_LABELS["province_code"],
                province_options,
                index=province_options.index(previous_province),
                format_func=lambda value: format_category_label("province_code", value, resources),
                key=province_key,
                help="Choose the province or city recorded in the EICV7 survey.",
            )
        district_options = [
            district
            for district in options_by_feature["district_code"]
            if int(float(district)) in geography.get(category_value_key(province_code), [])
        ]
        district_key = "predict_district_code"
        previous_district = st.session_state.get(district_key, defaults["district_code"])
        if previous_district not in district_options:
            previous_district = district_options[0]
        with district_column:
            district_code = st.selectbox(
                FEATURE_LABELS["district_code"],
                district_options,
                index=district_options.index(previous_district),
                format_func=lambda value: format_category_label("district_code", value, resources),
                key=district_key,
                help="District choices are limited to those recorded under the selected province or city.",
            )
        residence_options = options_by_feature["urban_rural_code"]
        residence_key = "predict_urban_rural_code"
        previous_residence = st.session_state.get(
            residence_key, defaults["urban_rural_code"]
        )
        if previous_residence not in residence_options:
            previous_residence = residence_options[0]
        with residence_column:
            urban_rural_code = st.selectbox(
                FEATURE_LABELS["urban_rural_code"],
                residence_options,
                index=residence_options.index(previous_residence),
                format_func=lambda value: format_category_label(
                    "urban_rural_code", value, resources
                ),
                key=residence_key,
                help="Choose the survey-recorded residence area.",
            )

    inputs = {
        "province_code": province_code,
        "district_code": district_code,
        "urban_rural_code": urban_rural_code,
    }
    with st.form("prediction_form", border=True):
        for section_name, section_features in FORM_SECTIONS:
            st.subheader(section_name)
            columns = st.columns(2)
            for index, feature in enumerate(section_features):
                if feature not in resources["features"]:
                    continue
                label = FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
                key = f"predict_{feature}"
                default = st.session_state.get(key, defaults.get(feature))
                help_text = FEATURE_HELP.get(
                    feature,
                    "Use the household value represented in the EICV7-derived model feature.",
                )
                with columns[index % len(columns)]:
                    if feature in resources["categorical_features"]:
                        options = options_by_feature[feature]
                        if default not in options:
                            default = options[0]
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "format_func": lambda value, current_feature=feature: format_category_label(
                                current_feature, value, resources
                            ),
                        }
                        if key not in st.session_state:
                            widget_options["index"] = options.index(default)
                        inputs[feature] = st.selectbox(label, options, **widget_options)
                    elif feature.startswith("share_") or feature.endswith("_share"):
                        numeric_value = float(default) if pd.notna(default) else 0.0
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "min_value": 0.0,
                            "max_value": 1.0,
                            "step": 0.01,
                        }
                        if key not in st.session_state:
                            widget_options["value"] = min(1.0, max(0.0, numeric_value))
                        inputs[feature] = st.slider(label, **widget_options)
                    elif feature in {"household_any_loan_attempt", "household_any_member_owes_money"}:
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "format_func": lambda value: "Yes" if value else "No",
                        }
                        if key not in st.session_state:
                            widget_options["index"] = int(float(default) >= 0.5)
                        inputs[feature] = st.selectbox(label, [0, 1], **widget_options)
                    elif feature == "resident_member_count":
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "min_value": 1,
                            "max_value": 40,
                            "step": 1,
                        }
                        if key not in st.session_state:
                            widget_options["value"] = min(40, max(1, int(round(float(default) if pd.notna(default) else 1))))
                        inputs[feature] = st.number_input(label, **widget_options)
                    elif feature == "mean_resident_age_years":
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "min_value": 0.0,
                            "max_value": 120.0,
                            "step": 1.0,
                        }
                        if key not in st.session_state:
                            widget_options["value"] = min(120.0, max(0.0, float(default) if pd.notna(default) else 0.0))
                        inputs[feature] = st.number_input(label, **widget_options)
                    elif feature == "access_median_travel_time_minutes":
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "min_value": 0.0,
                            "max_value": 1440.0,
                            "step": 5.0,
                        }
                        if key not in st.session_state:
                            widget_options["value"] = min(1440.0, max(0.0, float(default) if pd.notna(default) else 0.0))
                        inputs[feature] = st.number_input(label, **widget_options)
                    else:
                        widget_options = {
                            "key": key,
                            "help": help_text,
                            "min_value": 0.0,
                            "step": 1.0,
                        }
                        if key not in st.session_state:
                            widget_options["value"] = max(0.0, float(default) if pd.notna(default) else 0.0)
                        inputs[feature] = st.number_input(label, **widget_options)
        submitted = st.form_submit_button("Estimate account-access pattern", type="primary", help="Validate all model features and produce one model-based estimate.")

    if submitted:
        try:
            st.session_state["prediction_payload"] = predict_household(inputs, "Manual profile", resources)
            st.session_state["prediction_error"] = None
        except (ValueError, TypeError) as error:
            st.session_state["prediction_payload"] = None
            st.session_state["prediction_error"] = str(error)
        except Exception as error:
            logger.exception("Prediction request failed")
            st.session_state["prediction_payload"] = None
            st.session_state["prediction_error"] = f"Prediction failed ({type(error).__name__}). Check model and report compatibility."

    if st.session_state.get("prediction_error"):
        st.error(st.session_state["prediction_error"])
    payload = st.session_state.get("prediction_payload")
    if not payload:
        return
    prediction = payload["prediction"]
    probabilities = prediction["probabilities"]
    st.subheader("Model estimate")
    st.info("This is a model prediction of reported bank or Mobile Money account access, not a verified fact about a household and not a poverty or financial-vulnerability classification.")
    st.write(f"Predicted pattern: **{prediction['predicted_label']}**")
    with st.container(horizontal=True):
        st.metric("Estimated probability of account access", f"{probabilities['has_bank_or_momo_account']:.1%}", border=True)
        st.metric("Estimated probability of no account access", f"{probabilities['no_bank_or_momo_account']:.1%}", border=True)
        st.metric("Class-probability margin", f"{payload['confidence']['score']:.1%}", border=True)
    st.caption("The margin is not a probability that the prediction is correct. The model has not been calibrated or externally validated.")
    factors = payload.get("key_factors", [])
    if factors:
        st.subheader("Factors associated with this model estimate")
        factor_data = pd.DataFrame([{"Feature": FEATURE_LABELS.get(factor["feature_name"], factor["feature_name"]), "Change vs. reference (percentage points)": factor["local_ablation_probability_delta"] * 100} for factor in factors])
        factor_chart = alt.Chart(factor_data).mark_bar().encode(x=alt.X("Change vs. reference (percentage points):Q", title="Change in class-1 score (percentage points)"), y=alt.Y("Feature:N", sort="-x", title=None), color=alt.condition(alt.datum["Change vs. reference (percentage points)"] >= 0, alt.value("#176b4d"), alt.value("#a4443e")), tooltip=["Feature", alt.Tooltip("Change vs. reference (percentage points):Q", format="+.2f")]).properties(height=max(150, len(factors) * 35))
        st.altair_chart(factor_chart, width="stretch", alt="Local one-feature-at-a-time model score sensitivity, not causal effects")
        st.caption("Each value replaces one input with its training-fold median or mode. These local sensitivities are not causal contributions and do not sum to the prediction.")
    st.download_button("Download prediction interpretation JSON", json.dumps(payload, indent=2, ensure_ascii=False), file_name="household_prediction_interpretation.json", mime="application/json", help="Includes the display reference and model explanation. Review before sharing.")


def render_interventions(frames: dict[str, pd.DataFrame], report: dict) -> None:
    st.header("Intervention intelligence")
    st.caption("Observed survey gaps can guide further investigation. The suggestions below are options for program design, not proven impacts or official poverty classifications.")
    grouping = st.selectbox("Identify observed gaps by", ["province_code", "district_code", "consumption_quintile", "urban_rural_code"], format_func={"province_code": "Province", "district_code": "District", "consumption_quintile": "Consumption quintile", "urban_rural_code": "Residence area"}.get, key="intervention_grouping", help="Choose the descriptive comparison for potential priority areas.")
    minimum = st.number_input("Minimum households per group", min_value=10, max_value=1000, value=50, step=10, key="intervention_minimum", help="Hide groups with fewer resolved observations to reduce unstable comparisons.")
    evidence = account_access_summary(frames.get(grouping, pd.DataFrame()), grouping)
    evidence = evidence.loc[evidence["households"] >= minimum].copy()
    if evidence.empty:
        st.info("No groups meet the minimum support threshold for this comparison.")
        return
    evidence["Group"] = evidence[grouping].map(lambda value: display_code(value, grouping, report))
    evidence = evidence.sort_values(["no_account_share", "households"], ascending=[False, False])
    st.subheader("Observed evidence")
    st.dataframe(evidence[["Group", "households", "with_account", "without_account", "account_access_share", "no_account_share"]].rename(columns={"households": "Households", "with_account": "With account", "without_account": "Without account", "account_access_share": "Account-access share", "no_account_share": "Without-account share"}), hide_index=True, alt="Observed account access comparisons for groups meeting the support threshold")
    st.subheader("Potential response and monitoring")
    st.write("**Potential intervention to assess:** locally co-design accessible account-opening support, mobile-money onboarding, and financial/digital capability sessions in groups with lower observed access. These options require feasibility and community consultation; this dataset does not evaluate intervention impacts.")
    st.write("**Monitoring indicators:** account ownership at follow-up; active account use; distance or time to an access point; participant reach by age, sex, disability, and location; service complaints; and cost per participant.")
    st.write("**Limitations:** account access is not a poverty measure; group differences are descriptive, may reflect sampling uncertainty and confounding, and should be checked with survey weights, design-aware uncertainty, current administrative evidence, and local validation before resource allocation.")
    st.download_button("Download aggregate priority evidence CSV", evidence.drop(columns=[grouping], errors="ignore").to_csv(index=False), file_name="eicv7_observed_priority_evidence.csv", mime="text/csv", help="Exports grouped evidence only, not household-level records.")


st.markdown('<div class="eyebrow">NISR 2026 Big Data Hackathon</div>', unsafe_allow_html=True)
st.title("Rwanda Financial Vulnerability Intelligence")
st.markdown('<p class="intro">EICV7-based evidence on household financial-service access. Observed survey indicators, model estimates, and proposed actions are clearly distinguished.</p>', unsafe_allow_html=True)
section = st.segmented_control("Platform section", ["Overview", "Explore", "Predict", "Intervention intelligence"], default="Overview", selection_mode="single", key="platform_section")

try:
    aggregate_frames, dataset_report = load_dataset_and_report()
except Exception as error:
    logger.exception("Unable to load analytical household data")
    st.error("Published aggregate data or its report is unavailable. Prepare public assets from an authorized processed dataset to enable analysis.")
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")
    if section == "Predict":
        st.info("Prediction requires the processed dataset, report, and a registered saved model.")
    st.stop()

if section == "Overview":
    render_overview(dataset_report)
elif section == "Explore":
    render_explore(aggregate_frames, dataset_report)
elif section == "Predict":
    render_prediction(dataset_report)
else:
    render_interventions(aggregate_frames, dataset_report)

st.divider()
st.caption("EICV7 survey observations are not official current-year statistics. Predictions and group associations are not causal claims or individual eligibility determinations. Household-level data is not offered for download.")
