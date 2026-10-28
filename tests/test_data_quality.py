import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.build_financial_inclusion_dataset import data_quality_summary
from src.ui.financial_inclusion_service import load_ui_resources, validate_inputs


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
