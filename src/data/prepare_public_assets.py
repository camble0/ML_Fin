import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "financial_inclusion_household.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "analytical_dataset_report.json"
STRUCTURE_PATH = PROJECT_ROOT / "data" / "processed" / "eicv7_priority_structure.json"
PUBLIC_DATA_DIR = PROJECT_ROOT / "data" / "public"
MINIMUM_GROUP_SIZE = 50
DIMENSIONS = (
    "province_code",
    "district_code",
    "urban_rural_code",
    "consumption_quintile",
)


def build_public_assets() -> tuple[Path, Path]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Processed household dataset is missing: {DATASET_PATH}")
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Analytical report is missing: {REPORT_PATH}")
    if not STRUCTURE_PATH.is_file():
        raise FileNotFoundError(f"Survey code-label structure is missing: {STRUCTURE_PATH}")

    frame = pd.read_csv(DATASET_PATH)
    required = {"hhid", "hh_has_bank_or_momo_account", *DIMENSIONS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")

    resolved = frame.loc[frame["hh_has_bank_or_momo_account"].isin([0, 1])]
    dimensions = {}
    suppressed_groups = {}
    for dimension in DIMENSIONS:
        summary = (
            resolved.groupby(dimension, dropna=False)["hh_has_bank_or_momo_account"]
            .agg(households="size", with_account="sum")
            .reset_index()
        )
        summary["without_account"] = summary["households"] - summary["with_account"]
        summary["account_access_share"] = summary["with_account"] / summary["households"]
        summary = summary.loc[summary["households"] >= MINIMUM_GROUP_SIZE]
        suppressed_groups[dimension] = int(
            resolved[dimension].nunique(dropna=False) - len(summary)
        )
        dimensions[dimension] = [
            {
                dimension: value.item() if hasattr(value, "item") else value,
                "households": int(row.households),
                "with_account": int(row.with_account),
                "without_account": int(row.without_account),
                "account_access_share": float(row.account_access_share),
            }
            for value, row in zip(summary[dimension], summary.itertuples(index=False))
        ]

    public_bundle = {
        "source": "EICV7 processed household survey",
        "synthetic": False,
        "minimum_group_size": MINIMUM_GROUP_SIZE,
        "suppressed_groups": suppressed_groups,
        "dimensions": dimensions,
    }
    report = deepcopy(json.loads(REPORT_PATH.read_text(encoding="utf-8")))
    report.get("baseline_model_evaluation", {}).pop(
        "interpretation_demo_household_id", None
    )
    missingness = report.get("data_quality", {}).get("missingness", {})
    missingness.pop("hhid", None)
    missingness.pop("clust", None)
    report["public_release"] = {
        "unit_level_records_included": False,
        "minimum_group_size": MINIMUM_GROUP_SIZE,
        "synthetic_data": False,
    }

    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = PUBLIC_DATA_DIR / "household_aggregates.json"
    report_path = PUBLIC_DATA_DIR / "analytical_report.json"
    bundle_path.write_text(json.dumps(public_bundle, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return bundle_path, report_path


if __name__ == "__main__":
    bundle_path, report_path = build_public_assets()
    print(f"Wrote {bundle_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")
