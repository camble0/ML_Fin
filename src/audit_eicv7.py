import json
from pathlib import Path

import pyreadstat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "EICV7"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "eicv7_metadata_inventory.json"
HOUSEHOLD_LABEL_VARIABLES = {
    "province",
    "district",
    "ur",
    "quintile",
    "s5aq2",
    "s5aq7",
    "s5bq6",
    "s5cq3",
    "s5cq14",
    "s5cq20a",
    "s5cq25",
    "s7aq4",
}


def inspect_file(file_path: Path) -> dict:
    try:
        empty_frame, metadata = pyreadstat.read_dta(str(file_path), metadataonly=True)
        metadata_decoding = "default"
    except UnicodeDecodeError:
        empty_frame, metadata = pyreadstat.read_dta(
            str(file_path), metadataonly=True, encoding="latin1"
        )
        metadata_decoding = "latin1 fallback"
    del empty_frame

    variable_names = metadata.column_names
    variable_labels = metadata.column_labels or []
    storage_types = metadata.readstat_variable_types or {}
    stata_formats = metadata.original_variable_types or {}
    value_label_sets = metadata.variable_to_label or {}
    value_labels = metadata.value_labels or {}
    missing_ranges = metadata.missing_ranges or {}

    variables = []
    for index, name in enumerate(variable_names):
        label_set = value_label_sets.get(name)
        labels = value_labels.get(label_set, {})
        variable = {
            "name": name,
            "label": variable_labels[index] if index < len(variable_labels) else None,
            "storage_type": storage_types.get(name),
            "stata_format": stata_formats.get(name),
            "value_label_set": label_set,
            "missing_ranges": missing_ranges.get(name),
        }
        if (
            file_path.name == "CS_S01_S5_S7_Household.dta"
            and name in HOUSEHOLD_LABEL_VARIABLES
            and labels
        ):
            variable["value_labels"] = {
                str(value): label for value, label in labels.items()
            }
        variables.append(variable)

    return {
        "filename": file_path.name,
        "observations": metadata.number_rows,
        "variable_count": metadata.number_columns,
        "file_label": metadata.file_label,
        "file_encoding": metadata.file_encoding,
        "metadata_decoding": metadata_decoding,
        "notes": metadata.notes,
        "variables": variables,
    }


def inspect_geographic_codes(file_path: Path) -> dict:
    try:
        frame, _ = pyreadstat.read_dta(
            str(file_path),
            usecols=["province", "district"],
            apply_value_formats=False,
        )
    except UnicodeDecodeError:
        frame, _ = pyreadstat.read_dta(
            str(file_path),
            usecols=["province", "district"],
            apply_value_formats=False,
            encoding="latin1",
        )

    pairs_by_province = {}
    for province, district in frame.dropna().itertuples(index=False, name=None):
        pairs_by_province.setdefault(str(int(province)), set()).add(int(district))
    return {
        "source_file": file_path.name,
        "province_district_codes": {
            province: sorted(districts)
            for province, districts in sorted(pairs_by_province.items())
        },
    }


def main() -> None:
    files = sorted(DATA_DIR.glob("CS_*.dta"))
    inventory = {
        "source_directory": str(DATA_DIR.relative_to(PROJECT_ROOT)),
        "files": [],
    }

    print(f"Found {len(files)} EICV7 Stata files.")
    for file_path in files:
        try:
            file_inventory = inspect_file(file_path)
            inventory["files"].append(file_inventory)
            print(
                f"{file_path.name}: {file_inventory['observations']:,} observations, "
                f"{file_inventory['variable_count']:,} variables"
            )
        except Exception as error:
            inventory["files"].append(
                {
                    "filename": file_path.name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"ERROR {file_path.name}: {type(error).__name__}: {error}")

    household_file = DATA_DIR / "CS_S01_S5_S7_Household.dta"
    if household_file.is_file():
        inventory["geographic_codes"] = inspect_geographic_codes(household_file)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nInventory written to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
