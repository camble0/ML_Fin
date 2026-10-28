import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "EICV7"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "financial_inclusion_household.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "financial_inclusion_household.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "analytical_dataset_report.json"

TARGET_COLUMN = "hh_has_bank_or_momo_account"
TARGET_STATUS_COLUMN = "target_source_status"

CATEGORICAL_FEATURES = [
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
]

NUMERIC_FEATURES = [
    "resident_member_count",
    "mean_resident_age_years",
    "share_female_residents",
    "share_residents_age_15_64",
    "share_residents_ever_attended_school",
    "share_residents_with_mobile_phone",
    "share_residents_using_internet",
    "share_residents_worked_own_farm",
    "share_residents_worked_nonfarm_wage",
    "share_residents_running_nonfarm_business",
    "access_service_use_often_share",
    "access_service_satisfied_share",
    "access_service_improving_share",
    "access_median_travel_time_minutes",
    "household_any_loan_attempt",
    "household_any_member_owes_money",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def read_dta_columns(filename: str, columns: list[str]) -> pd.DataFrame:
    file_path = RAW_DIR / filename
    try:
        frame, _ = pyreadstat.read_dta(
            str(file_path), usecols=columns, apply_value_formats=False
        )
    except UnicodeDecodeError:
        frame, _ = pyreadstat.read_dta(
            str(file_path),
            usecols=columns,
            apply_value_formats=False,
            encoding="latin1",
        )
    return frame


def fraction_with_code(values: pd.Series, positive_codes: set[int]) -> float:
    observed = values.isin([1, 2])
    if not observed.any():
        return np.nan
    return float(values.loc[observed].isin(positive_codes).mean())


def aggregate_household_file() -> tuple[pd.DataFrame, dict]:
    columns = [
        "hhid",
        "clust",
        "province",
        "district",
        "ur",
        "strata_id",
        "weight",
        "pop_wt",
        "quintile",
        "s5aq2",
        "s5aq7",
        "s5bq6",
        "s5cq3",
        "s5cq14",
        "s5cq20a",
        "s5cq25",
        "s7aq4",
    ]
    frame = read_dta_columns("CS_S01_S5_S7_Household.dta", columns)
    missing_hhid = int(frame["hhid"].isna().sum())
    frame = frame.loc[frame["hhid"].notna()]
    if frame["hhid"].duplicated().any():
        raise ValueError("Household source hhid is not unique.")

    frame = frame.rename(
        columns={
            "province": "province_code",
            "district": "district_code",
            "ur": "urban_rural_code",
            "quintile": "consumption_quintile",
            "s5aq2": "dwelling_type_code",
            "s5aq7": "dwelling_owner_code",
            "s5bq6": "owns_residential_house",
            "s5cq3": "drinking_water_source_code",
            "s5cq14": "electricity_grid_connected",
            "s5cq20a": "internet_at_home",
            "s5cq25": "toilet_type_code",
            "s7aq4": "any_livestock",
        }
    )
    audit = {
        "source_rows": int(len(frame) + missing_hhid),
        "source_households": int(frame["hhid"].nunique()),
        "rows_missing_hhid": missing_hhid,
    }
    return frame, audit


def aggregate_person_file() -> tuple[pd.DataFrame, dict]:
    columns = [
        "hhid",
        "pid",
        "s1q1",
        "s1q3y",
        "s1q14",
        "s4aq1",
        "s4bq8",
        "s4bq15",
        "s6aq2",
        "s6aq5",
        "s6aq6",
    ]
    frame = read_dta_columns(
        "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta", columns
    )
    source_rows = len(frame)
    missing_hhid = int(frame["hhid"].isna().sum())
    missing_membership = int(frame["s1q14"].isna().sum())
    residents = frame.loc[frame["hhid"].notna() & frame["s1q14"].eq(1)]
    grouped = residents.groupby("hhid", sort=False)
    result = grouped.agg(
        resident_member_count=("pid", "count"),
        mean_resident_age_years=("s1q3y", "mean"),
        share_female_residents=("s1q1", lambda values: fraction_with_code(values, {2})),
        share_residents_ever_attended_school=(
            "s4aq1",
            lambda values: fraction_with_code(values, {1}),
        ),
        share_residents_with_mobile_phone=(
            "s4bq8",
            lambda values: fraction_with_code(values, {1}),
        ),
        share_residents_using_internet=(
            "s4bq15",
            lambda values: fraction_with_code(values, {1}),
        ),
        share_residents_worked_own_farm=(
            "s6aq2",
            lambda values: fraction_with_code(values, {1}),
        ),
        share_residents_worked_nonfarm_wage=(
            "s6aq5",
            lambda values: fraction_with_code(values, {1}),
        ),
        share_residents_running_nonfarm_business=(
            "s6aq6",
            lambda values: fraction_with_code(values, {1}),
        ),
    ).reset_index()
    ages = grouped["s1q3y"].agg(
        lambda values: float(values.between(15, 64).mean())
        if values.notna().any()
        else np.nan
    )
    result = result.merge(
        ages.rename("share_residents_age_15_64").reset_index(),
        on="hhid",
        how="left",
        validate="one_to_one",
    )
    audit = {
        "source_rows": int(source_rows),
        "rows_missing_hhid": missing_hhid,
        "rows_missing_membership_code": missing_membership,
        "resident_rows_used": int(len(residents)),
        "households_with_resident_records": int(result["hhid"].nunique()),
    }
    return result, audit


def aggregate_access_file() -> tuple[pd.DataFrame, dict]:
    columns = ["hhid", "s5fq1", "s5fq3a", "s5fq3b", "s5fq6", "s5fq7"]
    frame = read_dta_columns("CS_S5F_access_to_services.dta", columns)
    source_rows = len(frame)
    missing_hhid = int(frame["hhid"].isna().sum())
    frame = frame.loc[frame["hhid"].notna()]
    both_time_parts = frame["s5fq3a"].notna() & frame["s5fq3b"].notna()
    frame["travel_time_minutes"] = (
        frame["s5fq3a"].astype("float64") * 60
        + frame["s5fq3b"].astype("float64")
    ).where(both_time_parts)
    result = frame.groupby("hhid", sort=False).agg(
        access_service_item_count=("s5fq1", "size"),
        access_service_use_often_share=(
            "s5fq1",
            lambda values: fraction_with_code(values, {3, 4}),
        ),
        access_service_satisfied_share=(
            "s5fq6",
            lambda values: fraction_with_code(values, {1}),
        ),
        access_service_improving_share=(
            "s5fq7",
            lambda values: fraction_with_code(values, {3}),
        ),
        access_median_travel_time_minutes=("travel_time_minutes", "median"),
    ).reset_index()
    audit = {
        "source_rows": int(source_rows),
        "rows_missing_hhid": missing_hhid,
        "households_with_access_records": int(result["hhid"].nunique()),
        "rows_with_both_travel_time_components": int(both_time_parts.sum()),
    }
    return result, audit


def aggregate_credit_file() -> tuple[pd.DataFrame, dict]:
    columns = ["hhid", "pid", "s10aq4", "s10aq1", "s10aq3"]
    frame = read_dta_columns("CS_S10A1_A2_credits.dta", columns)
    source_rows = len(frame)
    missing_hhid = int(frame["hhid"].isna().sum())
    frame = frame.loc[frame["hhid"].notna()].copy()
    frame["credit_ids_missing_both"] = frame["pid"].isna() & frame["s10aq4"].isna()
    result = frame.groupby("hhid", sort=False).agg(
        household_any_loan_attempt=("s10aq1", lambda values: any_response_yes(values)),
        household_any_member_owes_money=("s10aq3", lambda values: any_response_yes(values)),
        credit_id_missing_row_count=("credit_ids_missing_both", "sum"),
        credit_source_row_count=("hhid", "size"),
    ).reset_index()
    audit = {
        "source_rows": int(source_rows),
        "rows_missing_hhid": missing_hhid,
        "rows_missing_both_member_and_loan_id": int(frame["credit_ids_missing_both"].sum()),
        "households_with_credit_records": int(result["hhid"].nunique()),
    }
    return result, audit


def any_response_yes(values: pd.Series) -> float:
    observed = values.isin([1, 2])
    if not observed.any():
        return np.nan
    return float(values.loc[observed].eq(1).any())


def build_target() -> tuple[pd.DataFrame, dict]:
    frame = read_dta_columns(
        "CS_S10C_Savings.dta", ["hhid", "s10cq1", "s10cq4"]
    )
    source_rows = len(frame)
    missing_hhid = int(frame["hhid"].isna().sum())
    frame = frame.loc[frame["hhid"].notna()]
    grouped = frame.groupby("hhid", sort=False)
    flags = grouped.agg(
        q4_any_yes=("s10cq4", lambda values: values.eq(1).any()),
        q4_any_no=("s10cq4", lambda values: values.eq(2).any()),
        q1_any_yes=("s10cq1", lambda values: values.eq(1).any()),
        q1_any_no=("s10cq1", lambda values: values.eq(2).any()),
        q4_observation_count=("s10cq4", "count"),
        q1_observation_count=("s10cq1", "count"),
    )

    target = pd.Series(pd.NA, index=flags.index, dtype="Int64")
    direct_yes = flags["q4_any_yes"]
    direct_no = ~direct_yes & flags["q4_any_no"]
    structural_no = (
        ~direct_yes
        & ~flags["q4_any_no"]
        & flags["q1_any_no"]
        & ~flags["q1_any_yes"]
    )
    target.loc[direct_yes] = 1
    target.loc[direct_no | structural_no] = 0

    source_status = pd.Series("unresolved_missing", index=flags.index, dtype="string")
    source_status.loc[direct_yes] = "direct_s10cq4_yes"
    source_status.loc[direct_no] = "direct_s10cq4_no"
    source_status.loc[structural_no] = "structural_no_from_s10cq1"
    result = flags.assign(
        **{
            TARGET_COLUMN: target,
            TARGET_STATUS_COLUMN: source_status,
            "target_s10cq4_conflict": flags["q4_any_yes"] & flags["q4_any_no"],
        }
    )
    result = result.drop(
        columns=["q4_any_yes", "q4_any_no", "q1_any_yes", "q1_any_no"]
    ).reset_index()

    audit = {
        "source_rows": int(source_rows),
        "rows_missing_hhid": missing_hhid,
        "households_with_savings_records": int(result["hhid"].nunique()),
        "source_q4_observations": int(frame["s10cq4"].notna().sum()),
        "households_with_conflicting_q4_responses": int(
            result["target_s10cq4_conflict"].sum()
        ),
        "unknown_nonmissing_s10cq4_codes": sorted(
            int(value)
            for value in frame.loc[
                frame["s10cq4"].notna() & ~frame["s10cq4"].isin([1, 2]),
                "s10cq4",
            ].unique()
        ),
    }
    return result, audit


def source_coverage(base: pd.DataFrame, source: pd.DataFrame) -> dict:
    base_ids = set(base["hhid"].tolist())
    source_ids = set(source["hhid"].tolist())
    return {
        "base_households_without_source_rows": int(len(base_ids - source_ids)),
        "source_households_outside_base": int(len(source_ids - base_ids)),
    }


def merge_one_to_one(
    base: pd.DataFrame, source: pd.DataFrame, source_name: str
) -> tuple[pd.DataFrame, dict]:
    coverage = source_coverage(base, source)
    merged = base.merge(
        source,
        on="hhid",
        how="left",
        validate="one_to_one",
        suffixes=("", f"_{source_name}"),
    )
    return merged, coverage


def missingness_summary(frame: pd.DataFrame) -> dict:
    return {
        column: {
            "missing_count": int(frame[column].isna().sum()),
            "missing_fraction": float(frame[column].isna().mean()),
        }
        for column in frame.columns
    }


def data_quality_summary(frame: pd.DataFrame) -> dict:
    invalid_record_counts = {}
    range_checks = {
        "resident_member_count": (1, 40),
        "mean_resident_age_years": (0, 120),
        "access_median_travel_time_minutes": (0, 1440),
    }
    for column, (minimum, maximum) in range_checks.items():
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = values.notna() & (~np.isfinite(values) | values.lt(minimum) | values.gt(maximum))
            invalid_record_counts[column] = int(invalid.sum())
    for column in NUMERIC_FEATURES:
        if column not in frame or not (column.startswith("share_") or column.endswith("_share")):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (~np.isfinite(values) | values.lt(0) | values.gt(1))
        invalid_record_counts[column] = int(invalid.sum())
    if TARGET_COLUMN in frame:
        target = frame[TARGET_COLUMN]
        invalid_record_counts[TARGET_COLUMN] = int((target.notna() & ~target.isin([0, 1])).sum())

    outlier_rows = {}
    for column in NUMERIC_FEATURES:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.dropna()
        if len(observed) < 4:
            outlier_rows[column] = 0
            continue
        lower_quartile, upper_quartile = observed.quantile([0.25, 0.75])
        spread = upper_quartile - lower_quartile
        outlier_rows[column] = int(
            ((values < lower_quartile - 1.5 * spread) | (values > upper_quartile + 1.5 * spread)).sum()
        )

    return {
        "method": "Missingness is counted per column. Invalid values use explicit feature bounds. Potential numeric outliers are flagged by the 1.5-IQR rule and are not removed or corrected.",
        "duplicate_household_rows": int(frame["hhid"].duplicated(keep=False).sum()) if "hhid" in frame else None,
        "duplicate_household_id_groups": int(frame.loc[frame["hhid"].duplicated(keep=False), "hhid"].nunique()) if "hhid" in frame else None,
        "invalid_record_counts": invalid_record_counts,
        "iqr_outlier_rows": outlier_rows,
        "missingness": missingness_summary(frame),
        "records_changed_by_audit": 0,
    }


def target_distribution(frame: pd.DataFrame) -> dict:
    target = frame[TARGET_COLUMN]
    total = len(frame)
    yes_count = int(target.eq(1).sum())
    no_count = int(target.eq(0).sum())
    missing_count = int(target.isna().sum())
    return {
        "yes": {"count": yes_count, "fraction_of_households": yes_count / total},
        "no": {"count": no_count, "fraction_of_households": no_count / total},
        "unresolved_missing": {
            "count": missing_count,
            "fraction_of_households": missing_count / total,
        },
        "resolved_households": yes_count + no_count,
    }


def build_report(
    dataset: pd.DataFrame,
    source_audits: dict,
    source_coverage_checks: dict,
    parquet_status: dict,
) -> dict:
    distribution = target_distribution(dataset)
    status_counts = {
        str(key): int(value)
        for key, value in dataset[TARGET_STATUS_COLUMN]
        .value_counts(dropna=False)
        .items()
    }
    return {
        "title": "Initial EICV7 household financial-inclusion dataset",
        "source_structure_report": "data/processed/eicv7_priority_structure.json",
        "target": {
            "name": TARGET_COLUMN,
            "definition": "At least one household member has a bank or Momo account.",
            "source": "CS_S10C_Savings.dta: s10cq4 (Yes=1, No=2), aggregated by hhid.",
            "derivation": "Target Yes if any s10cq4 response is Yes; otherwise No if any s10cq4 response is No. If s10cq4 is entirely unanswered and s10cq1 is exclusively No, record a structural No. Other unresolved cases remain missing.",
            "structural_no_assumption": "In the observed data, all 753 households with no s10cq4 response have s10cq1=No; s10cq1 is labelled ownership of bank account/financial institution/Momo/Tontine. This skip interpretation is explicit and should be confirmed against the questionnaire.",
            "household_source_status_counts": status_counts,
            "distribution": distribution,
            "q4_conflicting_response_households": source_audits["savings"]["households_with_conflicting_q4_responses"],
            "q4_conflict_rule": "A household with both Yes and No account rows is positive if any member/account row is Yes; conflict count remains available for audit and is excluded from predictors.",
            "alternatives_not_selected": [
                "s10cq1 is broader and yielded 14,301 Yes versus 753 No households, making it less discriminating.",
                "s9d5q1/s9d5q2 describe VUP financial-service loan application/approval and are missing for 14,721 of 15,054 households; they are too program-specific and sparse for the first general inclusion target.",
                "s10aq1/s10aq3 describe loan-seeking and debt, not account access. They are retained only as distinct aggregated predictors."
            ],
        },
        "number_of_households": int(len(dataset)),
        "number_of_features": int(len(MODEL_FEATURES)),
        "model_predictor_columns": MODEL_FEATURES,
        "categorical_predictors": CATEGORICAL_FEATURES,
        "numeric_predictors": NUMERIC_FEATURES,
        "source_files": [
            {
                "filename": filename,
                "purpose": purpose,
                **source_audits[key],
                "coverage_against_household_base": source_coverage_checks[key],
            }
            for key, filename, purpose in [
                ("household", "CS_S01_S5_S7_Household.dta", "Base household context, design fields, dwelling, utilities, and quintile."),
                ("person", "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta", "Resident-member demographics, education, digital access, and selected work indicators."),
                ("access", "CS_S5F_access_to_services.dta", "Household-level aggregates over basic-service items."),
                ("credit", "CS_S10A1_A2_credits.dta", "Household aggregates of loan-seeking and debt responses."),
                ("savings", "CS_S10C_Savings.dta", "Household target and target-source provenance."),
            ]
        ],
        "feature_transformations": [
            {
                "group": "household_context",
                "source_file": "CS_S01_S5_S7_Household.dta",
                "transformations": [
                    {"original_variables": ["province", "district", "ur", "quintile"], "resulting_features": ["province_code", "district_code", "urban_rural_code", "consumption_quintile"], "transformation": "Retain Stata codes as categorical values; one-hot encode in the model."},
                    {"original_variables": ["s5aq2", "s5aq7", "s5bq6", "s5cq3", "s5cq14", "s5cq20a", "s5cq25", "s7aq4"], "resulting_features": CATEGORICAL_FEATURES[4:], "transformation": "Retain interpretable dwelling, ownership, water, electricity, internet, toilet, and livestock response codes as categorical values."},
                    {"original_variables": ["hhid", "clust", "strata_id", "weight", "pop_wt"], "resulting_features": ["hhid", "clust", "strata_id", "weight", "pop_wt"], "transformation": "Preserve identifiers/design fields in the table; never include these as model predictors. clust is used only for grouped splitting."}
                ]
            },
            {
                "group": "person_demographics_education_digital_work",
                "source_file": "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person.dta",
                "filter": "Keep rows with s1q14=1 (household membership Yes), then aggregate by hhid.",
                "transformations": [
                    {"original_variables": ["pid"], "resulting_features": ["resident_member_count"], "transformation": "Count nonmissing resident person IDs per household."},
                    {"original_variables": ["s1q3y"], "resulting_features": ["mean_resident_age_years", "share_residents_age_15_64"], "transformation": "Mean age and share aged 15-64 among members with age observed."},
                    {"original_variables": ["s1q1"], "resulting_features": ["share_female_residents"], "transformation": "Fraction of known sex responses coded Female (2)."},
                    {"original_variables": ["s4aq1", "s4bq8", "s4bq15"], "resulting_features": ["share_residents_ever_attended_school", "share_residents_with_mobile_phone", "share_residents_using_internet"], "transformation": "Within-household share of valid Yes (1) responses; denominator is nonmissing Yes/No responses."},
                    {"original_variables": ["s6aq2", "s6aq5", "s6aq6"], "resulting_features": ["share_residents_worked_own_farm", "share_residents_worked_nonfarm_wage", "share_residents_running_nonfarm_business"], "transformation": "Within-household share of valid Yes (1) responses for each activity."}
                ]
            },
            {
                "group": "basic_service_access",
                "source_file": "CS_S5F_access_to_services.dta",
                "transformations": [
                    {"original_variables": ["s5fq1"], "resulting_features": ["access_service_use_often_share"], "transformation": "Share of valid service items coded Often (3) or Regularly (4)."},
                    {"original_variables": ["s5fq6", "s5fq7"], "resulting_features": ["access_service_satisfied_share", "access_service_improving_share"], "transformation": "Share of valid items coded Satisfied (1) and Improving (3), respectively."},
                    {"original_variables": ["s5fq3a", "s5fq3b"], "resulting_features": ["access_median_travel_time_minutes"], "transformation": "Convert hours*60 + minutes only where both source components are present, then take household median."},
                    {"original_variables": ["s5fq1"], "resulting_features": ["access_service_item_count"], "transformation": "Retain observed item count as a data-quality field; exclude from predictors."}
                ]
            },
            {
                "group": "credit_context",
                "source_file": "CS_S10A1_A2_credits.dta",
                "transformations": [
                    {"original_variables": ["s10aq1"], "resulting_features": ["household_any_loan_attempt"], "transformation": "Per hhid, 1 if any valid row is Yes; 0 if valid responses exist and none is Yes; otherwise missing."},
                    {"original_variables": ["s10aq3"], "resulting_features": ["household_any_member_owes_money"], "transformation": "Per hhid, 1 if any valid row is Yes; 0 if valid responses exist and none is Yes; otherwise missing."},
                    {"original_variables": ["pid", "s10aq4"], "resulting_features": ["credit_id_missing_row_count"], "transformation": "Count records where both member and loan IDs are missing; retain responses by hhid and exclude this diagnostic count from predictors."}
                ]
            },
            {
                "group": "financial_inclusion_target",
                "source_file": "CS_S10C_Savings.dta",
                "transformations": [
                    {"original_variables": ["s10cq4", "s10cq1"], "resulting_features": [TARGET_COLUMN, TARGET_STATUS_COLUMN, "target_s10cq4_conflict"], "transformation": "Use any s10cq4 Yes as household Yes; otherwise any s10cq4 No as No; if q4 is entirely unanswered and q1 is exclusively No, mark a structural No and label its source. Leave remaining unresolved cases missing. Mark households with both q4 codes for audit."}
                ]
            }
        ],
        "excluded_from_predictors": [
            {"variables": ["s10cq1", "s10cq4"], "reason": "Direct target construction/source-screen questions; target provenance and conflict flags are also excluded."},
            {"variables": ["s10cq3", "s10cq5", "s10cq6", "s10cq7", "s10cq8", "pid"], "reason": "Account IDs or account/tontine details from the target module; not loaded as predictors to avoid target-adjacent leakage."},
            {"variables": ["hhid", "clust", "strata_id", "weight", "pop_wt"], "reason": "Identifiers, split groups, and preserved survey-design fields; not model predictors."},
            {"variables": ["s10aq2", "s10aq4", "s10aq6", "s10aq7", "s10aq8", "s10aq9"], "reason": "Loan denial/source/purpose/collateral/repayment details are excluded from the initial feature set; several are conditional or post-loan details."},
            {"variables": ["s9d5q1-s9d5q7"], "reason": "VUP-specific financial-loan module was not selected as a general inclusion predictor or target because it is sparse and program-specific."}
        ],
        "credit_feature_caveat": "Aggregated credit indicators are contemporaneous financial behaviors, not causes. They do not directly define the bank/Momo target but may act as adjacent financial-inclusion proxies; interpret model scores accordingly.",
        "known_unresolved_source_issues": [
            {
                "source_file": "CS_S10C_Savings.dta",
                "issue": "The validated structural report found one nonmissing hhid-pid pair that does not match the person roster.",
                "handling": "The pipeline aggregates the target by hhid and does not use savings pid for linkage. The unmatched reference is neither repaired nor used to fabricate a person match; raw data remains untouched."
            }
        ],
        "survey_weight_decision": "Preserve weight and pop_wt unchanged in the CSV. Do not pass them as estimator sample weights until the survey design and target population estimand are confirmed. Exclude them from predictors.",
        "missingness_summary": missingness_summary(dataset),
        "data_quality": data_quality_summary(dataset),
        "source_audits": source_audits,
        "household_coverage_checks": source_coverage_checks,
        "raw_data_modified": False,
        "finscope_merged": False,
        "csv_output": str(OUTPUT_CSV.relative_to(PROJECT_ROOT)),
        "parquet_output": parquet_status,
        "baseline_status": "Not yet run.",
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    household, household_audit = aggregate_household_file()
    person, person_audit = aggregate_person_file()
    access, access_audit = aggregate_access_file()
    credit, credit_audit = aggregate_credit_file()
    target, savings_audit = build_target()

    base_hhids = set(household["hhid"].tolist())
    aggregated_sources = {
        "person": person,
        "access": access,
        "credit": credit,
        "savings": target,
    }
    source_coverage_checks = {"household": source_coverage(household, household)}
    source_coverage_checks.update(
        {
            name: source_coverage(household, source)
            for name, source in aggregated_sources.items()
        }
    )
    source_audits = {
        "household": household_audit,
        "person": person_audit,
        "access": access_audit,
        "credit": credit_audit,
        "savings": savings_audit,
    }

    dataset = household
    for source_name, source in aggregated_sources.items():
        dataset, _ = merge_one_to_one(dataset, source, source_name)

    if dataset["hhid"].isna().any() or dataset["hhid"].duplicated().any():
        raise ValueError("Final dataset must contain exactly one nonmissing row per hhid.")
    if set(dataset["hhid"].tolist()) != base_hhids:
        raise ValueError("Final household IDs differ from the household base file.")

    dataset = dataset.sort_values("hhid", kind="stable").reset_index(drop=True)
    dataset.to_csv(OUTPUT_CSV, index=False)

    parquet_status = {"written": False, "path": str(OUTPUT_PARQUET.relative_to(PROJECT_ROOT))}
    try:
        dataset.to_parquet(OUTPUT_PARQUET, index=False)
        parquet_status["written"] = True
    except (ImportError, ModuleNotFoundError) as error:
        parquet_status["reason"] = f"No Parquet engine is installed: {error}"

    report = build_report(
        dataset,
        source_audits,
        source_coverage_checks,
        parquet_status,
    )
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Households: {len(dataset):,}")
    print(f"Predictor features: {len(MODEL_FEATURES)}")
    print(f"Target distribution: {report['target']['distribution']}")
    print(f"CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Parquet: {parquet_status}")


if __name__ == "__main__":
    main()
