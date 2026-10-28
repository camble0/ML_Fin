import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "public" / "household_aggregates.json"


@unittest.skipUnless(DATA_PATH.is_file(), "Processed EICV7 data is not installed")
class StreamlitWorkflowTests(unittest.TestCase):
  def test_all_sections_and_prediction_workflow(self):
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    self.assertFalse(app.exception)
    self.assertEqual(app.title[0].value, "Rwanda Financial Vulnerability Intelligence")
    self.assertEqual(len(app.metric), 4)

    section = app.get_by_key("platform_section")
    for view in ("Explore", "Intervention intelligence", "Predict", "Overview"):
      section.set_value(view).run()
      self.assertFalse(app.exception)

    section.set_value("Explore").run()
    self.assertEqual(app.selectbox(key="explore_grouping").value, "province_code")

    section.set_value("Intervention intelligence").run()
    self.assertIsNotNone(app.get_by_key("intervention_grouping"))
    self.assertEqual(app.number_input(key="intervention_minimum").value, 50)

    section.set_value("Predict").run()
    self.assertIsNotNone(app.get_by_key("predict_resident_member_count"))
    share_input = app.get_by_key("predict_share_residents_with_mobile_phone")
    self.assertEqual((share_input.min, share_input.max), (0.0, 1.0))
    self.assertTrue(all(widget.label.strip() for widget in app.slider))

    app.button[0].click().run()
    self.assertFalse(app.exception)
    self.assertFalse(app.error)
    self.assertEqual(
      [metric.label for metric in app.metric][-3:],
      [
        "Estimated probability of account access",
        "Estimated probability of no account access",
        "Class-probability margin",
      ],
    )
    payload = app.session_state["prediction_payload"]
    probability_values = payload["prediction"]["probabilities"].values()
    self.assertTrue(all(0 <= value <= 1 for value in probability_values))
    self.assertAlmostEqual(sum(probability_values), 1.0)
    self.assertEqual(payload["household_id"], "Manual profile")
    self.assertNotEqual(payload["household_id"], "300162")
