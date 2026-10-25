import unittest
from unittest.mock import Mock, patch

import app
import train_model


class RainfallModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = next((path for path in train_model.DATA_CANDIDATES if path.is_file()), None)
        if cls.dataset is None:
            raise unittest.SkipTest("weatherAUS.csv is needed for dataset validation")
        cls.data = train_model.load_clean_data(cls.dataset)

    def test_csv_is_cleaned_and_uses_expected_features(self):
        self.assertGreater(len(self.data), 100_000)
        self.assertEqual(self.data["target"].nunique(), 2)
        self.assertTrue(self.data["Date"].notna().all())
        self.assertEqual(train_model.FEATURES, app.FEATURES)

    def test_temporal_split_keeps_later_dates_out_of_training(self):
        X_train, X_test, y_train, y_test, start = train_model.chronological_split(self.data)
        self.assertLess(self.data.loc[X_train.index, "Date"].max(), start)
        self.assertGreaterEqual(self.data.loc[X_test.index, "Date"].min(), start)
        self.assertEqual(y_train.nunique(), 2)
        self.assertEqual(y_test.nunique(), 2)

    def test_saved_model_loads_and_outputs_a_valid_prediction(self):
        bundle, error = app.load_model()
        self.assertIsNone(error)
        self.assertEqual(bundle["features"], app.FEATURES)
        prediction, probability = app.prediction_for(bundle, {
            "min_temp": 19.2, "max_temp": 27.0,
            "rainfall": 0.0, "humidity": 72.0,
        })
        self.assertIn(prediction, {"Rain likely", "Rain less likely"})
        self.assertIsNotNone(probability)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)


class OpenMeteoParsingTests(unittest.TestCase):
    @patch("app.requests.get")
    def test_forecast_uses_local_three_pm_humidity(self, get):
        response = Mock()
        response.json.return_value = {
            "daily": {
                "time": ["2026-10-26"],
                "temperature_2m_min": [18.0],
                "temperature_2m_max": [28.0],
                "rain_sum": [1.4],
            },
            "hourly": {
                "time": ["2026-10-26T14:00", "2026-10-26T15:00"],
                "relative_humidity_2m": [48, 76],
            },
        }
        get.return_value = response
        result = app.get_weather("Bengaluru")
        self.assertEqual(result["humidity"], 76.0)
        self.assertEqual(result["rainfall"], 1.4)
        self.assertEqual(get.call_args.kwargs["params"]["timezone"], "Asia/Kolkata")

    @patch("app.requests.get")
    def test_missing_three_pm_value_is_not_fabricated(self, get):
        response = Mock()
        response.json.return_value = {
            "daily": {
                "time": ["2026-10-26"], "temperature_2m_min": [18],
                "temperature_2m_max": [28], "rain_sum": [0],
            },
            "hourly": {"time": ["2026-10-26T14:00"], "relative_humidity_2m": [50]},
        }
        get.return_value = response
        with self.assertRaises(ValueError):
            app.get_weather("Bengaluru")


if __name__ == "__main__":
    unittest.main()
