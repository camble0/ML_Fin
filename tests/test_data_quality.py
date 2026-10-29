import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.build_financial_inclusion_dataset import data_quality_summary
from src.ui.financial_inclusion_service import (
    format_category_label,
    load_ui_resources,
    validate_inputs,
)


class DataQualityTests(unittest.TestCase):
    def test_quality_summary_flags_issues_without_mutation(self):
        frame = pd.DataFrame(
            {
                "hhid": [1, 1, 2, 3, 4],
                "resident_member_count": [2, 2, 0, 3, 4],
                "mean_resident_age_years": [30, 30, 130, 40, 35],
                "share_residents_with_mobile_phone": [0.5, 0.5, 1.2, 0.2, 0.4],
                "access_median_travel_time_minutes": [20, 20, 30, 40, 4000],
                "hh_has_bank_or_momo_account": [1, 1, 2, 0, 1],
            }
        )
        before = frame.copy(deep=True)

        summary = data_quality_summary(frame)

        self.assertEqual(summary["duplicate_household_rows"], 2)
        self.assertEqual(summary["duplicate_household_id_groups"], 1)
        self.assertEqual(summary["invalid_record_counts"]["resident_member_count"], 1)
        self.assertEqual(summary["invalid_record_counts"]["mean_resident_age_years"], 1)
        self.assertEqual(summary["invalid_record_counts"]["share_residents_with_mobile_phone"], 1)
        self.assertEqual(summary["invalid_record_counts"]["access_median_travel_time_minutes"], 1)
        self.assertEqual(summary["invalid_record_counts"]["hh_has_bank_or_momo_account"], 1)
        self.assertEqual(summary["iqr_outlier_rows"]["access_median_travel_time_minutes"], 1)
        self.assertEqual(summary["records_changed_by_audit"], 0)
        pd.testing.assert_frame_equal(frame, before)

    def test_prediction_validation_rejects_out_of_range_shares_and_categories(self):
        resources = {
            "features": ["share_example", "residence_code"],
            "categorical_features": ["residence_code"],
            "category_options": {"residence_code": [1, 2]},
        }
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_inputs({"share_example": 1.2, "residence_code": 1}, resources)
        with self.assertRaisesRegex(ValueError, "supported value"):
            validate_inputs({"share_example": 0.5, "residence_code": 3}, resources)

    def test_missing_model_registration_and_public_report_fail_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "not registered"):
            load_ui_resources("not_registered")
        with patch(
            "src.ui.financial_inclusion_service.PUBLIC_REPORT_PATH",
            Path("/missing/analytical_report.json"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Published analytical report is missing"):
                load_ui_resources("random_forest")

    def test_model_resources_do_not_load_household_rows(self):
        resources = load_ui_resources("random_forest")
        self.assertNotIn("dataset", resources)
        self.assertIn("model", resources)

    def test_all_model_categories_have_verified_labels_and_keep_raw_values(self):
        resources = load_ui_resources("random_forest")
        for feature in resources["categorical_features"]:
            for value in resources["category_options"][feature]:
                label = format_category_label(feature, value, resources)
                self.assertTrue(label.strip())
                self.assertNotIn("Code ", label)
                self.assertIn(value, resources["category_options"][feature])

        labels = resources["value_labels"]
        self.assertEqual(labels["s5bq6"], {"1": "Yes", "2": "No"})
        self.assertEqual(labels["s5cq20a"], {"1": "Yes", "2": "No"})
        self.assertEqual(labels["s7aq4"], {"1": "Yes", "2": "No"})
        self.assertEqual(
            labels["s5cq14"],
            {
                "1": "Yes on grid",
                "2": "Yes from solar",
                "3": "Yes both on grid  and from solar",
                "4": "No",
            },
        )
        expected_source_labels = {
            "s5aq2": {
                "1": "House occupied by one household",
                "2": "House occupied by multiple Households",
                "3": "Multi-storied building occupied by one household",
                "4": "Multi-storied building occupied by many households",
                "5": "Several buildings in a compound occupied by one household",
                "6": "Several buildings in a compound occupied by several households",
                "7": "Other (Specify)",
            },
            "s5aq7": {
                "1": "Head",
                "2": "Spouse",
                "3": "Spouse with head",
                "4": "Other household member",
                "5": "Relative",
                "6": "Non relative",
                "7": "State",
                "8": "Private company",
                "9": "Other",
            },
            "s5bq6": {"1": "Yes", "2": "No"},
            "s5cq3": {
                "1": "Piped into Dwelling",
                "2": "Piped into compound, yard/pot",
                "3": "Piped to neighbor HH3",
                "4": "Public tap/standpipe",
                "5": "Tube Well/Borehole",
                "6": "Protected spring",
                "7": "Unprotected spring",
                "8": "Protected well",
                "9": "Unprotected well",
                "10": "Rain water",
                "11": "Tanker Truck",
                "12": "River/Stream/Irrigation Channel",
                "13": "Lake/Pond/Surface water",
                "14": "Bottled water",
                "99": "Other specify)",
            },
            "s5cq14": {
                "1": "Yes on grid",
                "2": "Yes from solar",
                "3": "Yes both on grid  and from solar",
                "4": "No",
            },
            "s5cq20a": {"1": "Yes", "2": "No"},
            "s5cq25": {
                "1": "Flush to piped sewer system",
                "2": "Flush to septic tank",
                "3": "Flush to pit latrine",
                "4": "Ventilated improved pit (VIP) latrine",
                "5": "Pit Latrine with constructed floor slab",
                "6": "Composting toilet",
                "7": "Pit latrine without constructed floor slab",
                "8": "Other (specify)",
                "9": "No toilet /Bush/Field",
            },
            "s7aq4": {"1": "Yes", "2": "No"},
        }
        self.assertEqual(
            {name: labels[name] for name in expected_source_labels},
            expected_source_labels,
        )

    def test_labels_do_not_change_model_inputs_or_prediction_schema(self):
        resources = load_ui_resources("random_forest")
        inputs = dict(resources["reference_values"])
        for feature in resources["categorical_features"]:
            inputs[feature] = resources["category_options"][feature][0]
        inputs["province_code"] = 1.0
        inputs["district_code"] = resources["geographic_code_pairs"]["1"][0]

        feature_frame = validate_inputs(inputs, resources)

        self.assertEqual(feature_frame.columns.tolist(), resources["features"])
        for feature in resources["categorical_features"]:
            self.assertEqual(feature_frame.iloc[0][feature], inputs[feature])
            self.assertIsInstance(
                format_category_label(feature, inputs[feature], resources), str
            )
        self.assertTrue(all(np.issubdtype(dtype, np.number) for dtype in feature_frame.dtypes))
        probabilities = resources["model"].predict_proba(feature_frame)
        self.assertEqual(probabilities.shape, (1, 2))

    def test_every_supported_province_district_pair_is_valid(self):
        resources = load_ui_resources("random_forest")
        inputs = dict(resources["reference_values"])
        for province, districts in resources["geographic_code_pairs"].items():
            for district in districts:
                inputs["province_code"] = float(province)
                inputs["district_code"] = float(district)
                validated = validate_inputs(inputs, resources)
                self.assertEqual(validated.iloc[0]["province_code"], float(province))
                self.assertEqual(validated.iloc[0]["district_code"], float(district))

        inputs["province_code"] = 1.0
        inputs["district_code"] = 21.0
        with self.assertRaisesRegex(ValueError, "does not belong"):
            validate_inputs(inputs, resources)
