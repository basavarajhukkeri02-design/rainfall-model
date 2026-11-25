import tempfile
import unittest
import secrets
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import app
import joblib
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo
from werkzeug.security import check_password_hash, generate_password_hash


class FakePipeline:
    classes_ = [0, 1]

    def __init__(self, probability=0.8):
        self.probability = probability
        self.received_columns = None

    def predict_proba(self, features):
        self.received_columns = list(features.columns)
        return [[1 - self.probability, self.probability]]


MODEL_THRESHOLD = 0.17524123715314846


def fake_bundle(probability=0.8):
    return {
        "pipeline": FakePipeline(probability),
        "features": list(app.FEATURES),
        "threshold": MODEL_THRESHOLD,
        "positive_label": 1,
    }


def weather_for(district, features=None):
    values = features or {
        "Rainfall_Lag_1": 1.0, "Rainfall_Lag_2": 2.0,
        "Rainfall_Lag_3": 3.0, "Rainfall_Lag_7": 0.0,
        "Rolling_Mean_3": 2.0, "Rolling_Sum_7": 12.0,
        "Month": 11, "Day_Of_Year": 324, "District": district,
    }
    return {
        "district": district, "forecast_date": "2026-11-20",
        "prediction_date": "2026-11-21", "min_temp": 19.0,
        "max_temp": 29.0, "rainfall": 2.5, "humidity": 72.0,
        "current_temp": 24.0, "weather_code": 0,
        "weather_condition": "sunny", "weather_condition_label": "Sunny",
        "forecast_days": [
            {"date": (datetime(2026, 11, 20) + timedelta(days=offset)).date().isoformat(),
             "min_temp": 19.0, "max_temp": 29.0, "rainfall": 2.5,
             "weather_code": 0, "weather_condition": "sunny", "weather_condition_label": "Sunny"}
            for offset in range(7)
        ],
        "karnataka_history_features": dict(values),
        "karnataka_history_error": None,
    }


def open_meteo_payload(daily_overrides=None, hourly_overrides=None):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    today = now.date()
    dates = [(today - timedelta(days=offset)).isoformat() for offset in range(7, -7, -1)]
    daily = {
        "time": dates,
        "temperature_2m_min": [18.0] * len(dates),
        "temperature_2m_max": [30.0] * len(dates),
        "rain_sum": [float(index) for index in range(len(dates))],
        "weather_code": [0] * len(dates),
    }
    times = sorted({now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00"), f"{today.isoformat()}T15:00"})
    hourly = {"time": times, "relative_humidity_2m": [70.0] * len(times), "temperature_2m": [25.0] * len(times)}
    daily.update(daily_overrides or {})
    hourly.update(hourly_overrides or {})
    return {"daily": daily, "hourly": hourly}


class WeatherResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or open_meteo_payload()
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(f"HTTP {self.status_code}", response=response)

    def json(self):
        return self.payload


class AgriRainAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        app.app.config.update(TESTING=True, DATABASE=str(Path(self.temp_dir.name) / "test.db"))
        app.init_db()
        with app._weather_cache_lock:
            app._weather_cache.clear()
        self.client = app.app.test_client()
        with app.database_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email,password_hash,created_at) VALUES (?,?,?)",
                ("owner@example.com", generate_password_hash("password123"), "2026-10-26T10:00:00+05:30"),
            )
            self.user_id = cursor.lastrowid
        with self.client.session_transaction() as user_session:
            user_session["user_id"] = self.user_id
            user_session["user_email"] = "owner@example.com"
            user_session["_csrf_token"] = secrets.token_urlsafe(32)
        with app._translation_rate_lock:
            app._translation_request_times.clear()
        self.weather = weather_for("Mysuru")

    def tearDown(self):
        self.temp_dir.cleanup()
        app.app.config.update(TESTING=False, DATABASE=str(app.DATABASE))

    def count_records(self):
        with app.database_connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]

    def post_form(self, path, data):
        with self.client.session_transaction() as user_session:
            token = user_session.get("_csrf_token")
            if not token:
                token = secrets.token_urlsafe(32)
                user_session["_csrf_token"] = token
        payload = dict(data)
        payload["_csrf_token"] = token
        return self.client.post(path, data=payload)

    def post_translation(self, data):
        with self.client.session_transaction() as user_session:
            token = user_session["_csrf_token"]
        return self.client.post(
            "/api/translate", json=data,
            headers={"X-CSRF-Token": token},
        )

    @patch("app.load_model", return_value=(fake_bundle(), None))
    @patch("app.get_weather")
    def test_prediction_is_saved_once_and_refresh_does_not_resubmit(self, get_weather, _model):
        get_weather.return_value = self.weather
        response = self.post_form("/dashboard", {"district": "Mysuru"})
        response = self.client.get(response.location, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rain likely", response.data)
        self.assertEqual(self.count_records(), 1)
        self.client.get("/dashboard")
        self.assertEqual(self.count_records(), 1)

    @patch("app.load_model", return_value=(fake_bundle(0.25), None))
    @patch("app.get_weather")
    def test_prediction_page_shows_weather_date_target_date_probability_and_source(self, get_weather, _model):
        get_weather.return_value = self.weather
        response = self.post_form("/dashboard", {"district": "Mysuru"})
        page = self.client.get(response.location, follow_redirects=True)

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Rain likely", page.data)
        self.assertIn(b"Forecast values are for 2026-11-20", page.data)
        self.assertIn(b"Estimate for: 2026-11-21", page.data)
        self.assertIn(b"Estimated model probability: 25.0%", page.data)
        self.assertIn(b"Current weather (reference)", page.data)
        self.assertIn(b"previous seven days of district rainfall", page.data)

    @patch("app.load_model", return_value=(fake_bundle(), None))
    @patch("app.get_weather")
    def test_dashboard_renders_values_week_forecast_icons_and_navigation(self, get_weather, _model):
        get_weather.return_value = self.weather
        page = self.client.get("/dashboard?district=Mysuru")
        self.assertEqual(page.status_code, 200)
        for label in (b"Minimum Temperature", b"Maximum Temperature", b"Rainfall", b"Humidity"):
            self.assertIn(label, page.data)
        self.assertIn(b"weather-icon--sunny", page.data)
        self.assertEqual(len(re.findall(rb'<article class="forecast-day', page.data)), 7)
        self.assertIn(b"Rainfall Prediction", page.data)
        self.assertIn(b"History", page.data)
        self.assertIn(b"Logout", page.data)
        css = self.client.get("/static/css/style.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(b"max-width:620px", css.data)
        self.assertIn(b"weather-icon--clear-night", css.data)
        css.close()

    @patch("app.load_model", return_value=(fake_bundle(), None))
    @patch("app.get_weather")
    def test_missing_rainfall_history_is_explained_without_showing_a_result(self, get_weather, _model):
        missing_history = weather_for("Mysuru")
        missing_history["karnataka_history_features"] = None
        missing_history["karnataka_history_error"] = "Open-Meteo did not return valid rainfall for all seven prior days"
        get_weather.return_value = missing_history

        response = self.post_form("/dashboard", {"district": "Mysuru"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"complete recent rainfall history is missing or invalid", response.data)
        self.assertNotIn(b"Estimated model probability:", response.data)
        self.assertEqual(self.count_records(), 0)

    @patch("app.get_weather")
    def test_end_to_end_real_model_prediction_is_saved(self, get_weather):
        get_weather.return_value = weather_for("Bengaluru Urban")
        response = self.post_form("/dashboard", {"district": "Bengaluru Urban"})
        self.assertEqual(response.status_code, 302)
        page = self.client.get(response.location, follow_redirects=True)
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Rain likely", page.data)
        with app.database_connection() as connection:
            record = connection.execute(
                "SELECT user_id, rainfall_probability, humidity FROM prediction_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(record["user_id"], self.user_id)
        self.assertGreaterEqual(record["rainfall_probability"], 0)
        self.assertLessEqual(record["rainfall_probability"], 1)
        self.assertEqual(record["humidity"], self.weather["humidity"])

    def test_model_loader_uses_karnataka_bundle_schema_and_threshold(self):
        bundle, error = app.load_model()
        self.assertIsNone(error)
        self.assertEqual(app.MODEL_PATH, app.BASE_DIR / "karnataka_model" / "rainfall_model.pkl")
        self.assertEqual(bundle["features"], [
            "Rainfall_Lag_1", "Rainfall_Lag_2", "Rainfall_Lag_3", "Rainfall_Lag_7",
            "Rolling_Mean_3", "Rolling_Sum_7", "Month", "Day_Of_Year", "District",
        ])
        self.assertEqual(bundle["threshold"], MODEL_THRESHOLD)
        self.assertEqual(bundle["positive_label"], 1)
        self.assertEqual(list(bundle["pipeline"].feature_names_in_), bundle["features"])

    def test_threshold_and_feature_order_are_applied_from_bundle(self):
        for probability, expected in (
            (MODEL_THRESHOLD - 1e-8, "Rain less likely"),
            (MODEL_THRESHOLD, "Rain likely"),
            (MODEL_THRESHOLD + 1e-8, "Rain likely"),
        ):
            with self.subTest(probability=probability):
                bundle = fake_bundle(probability)
                result, actual_probability = app.prediction_for(bundle, weather_for("Mysuru"))
                self.assertEqual(result, expected)
                self.assertAlmostEqual(actual_probability, probability)
                self.assertEqual(bundle["pipeline"].received_columns, bundle["features"])

    def test_prediction_rejects_missing_nonfinite_and_mismatched_history(self):
        weather = weather_for("Mysuru")
        weather["karnataka_history_features"].pop("Rainfall_Lag_7")
        with self.assertRaisesRegex(ValueError, "features do not match"):
            app.prediction_for(fake_bundle(), weather)

        weather = weather_for("Mysuru")
        weather["karnataka_history_features"]["Rainfall_Lag_1"] = np.nan
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            app.prediction_for(fake_bundle(), weather)

        weather = weather_for("Mysuru")
        weather["karnataka_history_features"]["District"] = "Ballari"
        with self.assertRaisesRegex(ValueError, "does not match"):
            app.prediction_for(fake_bundle(), weather)

    def test_history_builder_rejects_a_missing_prior_day(self):
        feature_date = "2026-11-20"
        dates = [(datetime.fromisoformat(feature_date).date() - timedelta(days=offset)).isoformat() for offset in range(1, 8)]
        dates.remove("2026-11-17")
        features, error = app.build_karnataka_history_features(dates, [0.0] * len(dates), feature_date, "Mysuru")
        self.assertIsNone(features)
        self.assertIn("all seven prior days", error)

    def test_weather_adapter_rejects_an_outdated_forecast_response(self):
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        old_day = today - timedelta(days=2)
        dates = [(old_day - timedelta(days=offset)).isoformat() for offset in range(7, -1, -1)]
        payload = {
            "daily": {
                "time": dates, "temperature_2m_min": [18.0] * 8,
                "temperature_2m_max": [30.0] * 8, "rain_sum": [1.0] * 8,
            },
            "hourly": {"time": [], "relative_humidity_2m": []},
        }

        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return payload

        with patch("app.requests.get", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "Today's forecast date is missing"):
                app.get_weather("Mysuru")

    def test_all_31_districts_render_and_run_through_flask(self):
        districts = set(app.DISTRICTS)
        self.assertEqual(len(districts), 31)
        bundle = joblib.load(app.MODEL_PATH)
        encoder = bundle["pipeline"].named_steps["preprocessor"].named_transformers_["district"].named_steps["encoder"]
        self.assertEqual(districts, set(encoder.categories_[0]))
        with patch("app.get_weather", side_effect=lambda district: weather_for(district)):
            page = self.client.get("/dashboard")
            self.assertEqual(page.status_code, 200)
            select = page.get_data(as_text=True).split('<select id="district"', 1)[1].split("</select>", 1)[0]
            rendered = set(re.findall(r'<option value="([^"]+)"', select))
            self.assertEqual(rendered, districts)

            for district in sorted(districts):
                with self.subTest(district=district):
                    response = self.post_form("/dashboard", {"district": district})
                    self.assertEqual(response.status_code, 302)

        with app.database_connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT district FROM prediction_history WHERE user_id = ?", (self.user_id,)
            ).fetchall()
        self.assertEqual({row[0] for row in rows}, districts)

    def test_default_district_is_bengaluru_urban(self):
        with patch("app.get_weather", return_value=weather_for("Bengaluru Urban")):
            response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<option value="Bengaluru Urban" selected>', response.data)

    def test_karnataka_model_predicts_latest_prepared_row_for_each_district(self):
        data = pd.read_csv(
            app.BASE_DIR / "karnataka_ml_features.csv",
            usecols=["Date", *app.FEATURES],
            engine="c",
        )
        bundle, error = app.load_model()
        self.assertIsNone(error)
        self.assertEqual(set(data["District"].unique()), set(app.DISTRICTS))
        for district, group in data.groupby("District"):
            with self.subTest(district=district):
                latest = group.sort_values("Date").iloc[-1]
                inputs = {name: latest[name] for name in bundle["features"]}
                weather = weather_for(district, inputs)
                result, probability = app.prediction_for(bundle, weather)
                self.assertIn(result, {"Rain likely", "Rain less likely"})
                self.assertTrue(np.isfinite(probability))
                self.assertGreaterEqual(probability, 0)
                self.assertLessEqual(probability, 1)

    def test_open_meteo_adapter_uses_each_district_location_and_builds_history(self):
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        response = WeatherResponse()
        for district, coordinates in app.DISTRICTS.items():
            with self.subTest(district=district), patch("app.requests.get", return_value=response) as get:
                weather = app.get_weather(district)
                params = get.call_args.kwargs["params"]
                self.assertEqual((params["latitude"], params["longitude"]), coordinates)
                self.assertEqual(params["past_days"], 7)
                self.assertEqual(params["forecast_days"], 7)
                self.assertEqual(params["timezone"], "Asia/Kolkata")
                self.assertEqual(weather["district"], district)
                self.assertEqual(weather["forecast_date"], today.isoformat())
                self.assertEqual(weather["weather_code"], 0)
                local_hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
                expected_condition = "clear-night" if local_hour < 6 or local_hour >= 18 else "sunny"
                self.assertEqual(weather["weather_condition"], expected_condition)
                self.assertEqual(weather["current_temp"], 25.0)
                self.assertEqual(len(weather["forecast_days"]), 7)
                self.assertEqual(weather["prediction_date"], (today + timedelta(days=1)).isoformat())
                self.assertEqual(weather["karnataka_history_features"]["District"], district)
                self.assertEqual(list(weather["karnataka_history_features"]), app.FEATURES)
                self.assertIsNone(weather["karnataka_history_error"])

    def test_open_meteo_weather_codes_map_to_all_animated_icon_conditions(self):
        expected = {
            0: ("sunny", "Sunny"),
            2: ("partly-cloudy", "Partly cloudy"),
            3: ("cloudy", "Cloudy"),
            61: ("rainy", "Rain"),
            95: ("thunderstorm", "Thunderstorm"),
        }
        for code, condition in expected.items():
            with self.subTest(code=code):
                self.assertEqual(app.weather_condition_for_code(code), condition)
        self.assertEqual(app.weather_condition_for_code(0, is_night=True), ("clear-night", "Clear night"))

    def test_forecast_week_is_date_ordered_and_contains_weather_metrics(self):
        with patch("app.requests.get", return_value=WeatherResponse()):
            weather = app.get_weather("Mysuru")
        days = weather["forecast_days"]
        self.assertEqual(len(days), 7)
        self.assertEqual(days[0]["date"], weather["forecast_date"])
        self.assertEqual(days[-1]["date"], (datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=6)).isoformat())
        self.assertTrue(all({"min_temp", "max_temp", "rainfall", "weather_condition"} <= day.keys() for day in days))

    def test_open_meteo_adapter_uses_forecast_day_weather_code(self):
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        dates = [(today - timedelta(days=offset)).isoformat() for offset in range(7, -2, -1)]
        codes = [0] * len(dates)
        codes[dates.index(today.isoformat())] = 95
        payload = open_meteo_payload({"weather_code": codes})
        with patch("app.requests.get", return_value=WeatherResponse(payload)):
            weather = app.get_weather("Bengaluru Urban")
        self.assertEqual(weather["weather_code"], 95)
        self.assertEqual(weather["weather_condition"], "thunderstorm")
        self.assertEqual(weather["weather_condition_label"], "Thunderstorm")

    def test_weather_retry_and_duplicate_suppression_for_multiple_districts(self):
        districts = ("Bagalkot", "Mysuru", "Yadgir")
        for district in districts:
            with self.subTest(district=district):
                response = WeatherResponse()
                with patch("app.requests.get", side_effect=[requests.Timeout("temporary"), response]) as get, \
                     patch("app.time.sleep"):
                    first = app.get_weather(district)
                    second = app.get_weather(district)
                self.assertEqual(get.call_count, 2)
                self.assertEqual(first["district"], district)
                self.assertEqual(first["min_temp"], 18.0)
                self.assertEqual(first["max_temp"], 30.0)
                self.assertTrue(np.isfinite(first["rainfall"]))
                self.assertEqual(first["humidity"], 70.0)
                self.assertEqual(first, second)

    def test_weather_retries_temporary_http_503_and_does_not_cache_failures(self):
        for district in ("Belagavi", "Tumakuru"):
            with self.subTest(district=district):
                with patch("app.requests.get", side_effect=[WeatherResponse(status_code=503), WeatherResponse()]) as get, \
                     patch("app.time.sleep"):
                    weather = app.get_weather(district)
                self.assertEqual(get.call_count, 2)
                self.assertEqual(weather["district"], district)

    def test_weather_retries_are_bounded_after_connection_failures(self):
        with patch("app.requests.get", side_effect=requests.ConnectionError("offline")) as get, \
             patch("app.time.sleep"):
            with self.assertRaises(requests.ConnectionError):
                app.get_weather("Kolar")
        self.assertEqual(get.call_count, 3)
        self.assertNotIn("Kolar", app._weather_cache)

    def test_weather_rejects_invalid_values_before_display(self):
        cases = (
            ({"temperature_2m_min": [31.0] * 9}, "minimum temperature"),
            ({"rain_sum": [-1.0] * 9}, "outside its valid range"),
            ({"temperature_2m_min": [float("nan")] * 9}, "missing number"),
        )
        for index, (overrides, message) in enumerate(cases):
            district = ("Hassan", "Raichur", "Udupi")[index]
            with self.subTest(district=district), patch(
                "app.requests.get", return_value=WeatherResponse(open_meteo_payload(overrides))
            ):
                with self.assertRaisesRegex(ValueError, message):
                    app.get_weather(district)
                self.assertNotIn(district, app._weather_cache)

    def test_missing_humidity_is_not_returned_as_partial_weather(self):
        response = WeatherResponse(open_meteo_payload(hourly_overrides={"relative_humidity_2m": []}))
        with patch("app.requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "3 PM humidity forecast is unavailable"):
                app.get_weather("Koppal")
        self.assertNotIn("Koppal", app._weather_cache)

    def test_concurrent_same_district_reads_share_one_api_request(self):
        from concurrent.futures import ThreadPoolExecutor

        with patch("app.requests.get", return_value=WeatherResponse()) as get:
            with ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(app.get_weather, ["Mysuru"] * 6))
        self.assertEqual(get.call_count, 1)
        self.assertTrue(all(result["district"] == "Mysuru" for result in results))

    def test_incomplete_or_outdated_history_is_not_saved_as_a_prediction(self):
        weather = weather_for("Mysuru")
        weather["karnataka_history_features"] = None
        weather["karnataka_history_error"] = "Open-Meteo did not return valid rainfall for all seven prior days"
        with patch("app.get_weather", return_value=weather):
            response = self.post_form("/dashboard", {"district": "Mysuru"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"recent rainfall history is missing or invalid", response.data)
        self.assertEqual(self.count_records(), 0)

    @patch("app.get_weather")
    def test_invalid_district_is_not_saved(self, get_weather):
        get_weather.return_value = self.weather
        response = self.post_form("/dashboard", {"district": "Not a district"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.count_records(), 0)

    @patch("app.load_model", return_value=(None, "Model unavailable"))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_weather_timeout_is_shown_without_server_error(self, _weather, _model):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Weather forecast is temporarily unavailable", response.data)

    @patch("app.load_model", return_value=(fake_bundle(), None))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_manual_weather_values_do_not_replace_missing_history(self, _weather, _model):
        response = self.post_form("/dashboard", {
            "district": "Mysuru", "observed_min_temp": "19.0",
            "observed_max_temp": "29.0", "observed_rainfall": "2.5",
            "observed_humidity": "72",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"seven complete prior rainfall days are required", response.data)
        self.assertEqual(self.count_records(), 0)

    @patch("app.load_model", return_value=(fake_bundle(), None))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_missing_manual_observation_does_not_predict(self, _weather, _model):
        response = self.post_form("/dashboard", {
            "district": "Mysuru", "observed_min_temp": "19.0",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"seven complete prior rainfall days are required", response.data)
        self.assertEqual(self.count_records(), 0)

    def test_history_clear_requires_confirmation(self):
        with app.database_connection() as connection:
            other_id = connection.execute(
                "INSERT INTO users (email,password_hash,created_at) VALUES (?,?,?)",
                ("other@example.com", generate_password_hash("password123"), "2026-10-26T10:00:00+05:30"),
            ).lastrowid
            connection.execute(
                "INSERT INTO prediction_history (user_id,district,prediction,created_at) VALUES (?,?,?,?)",
                (self.user_id, "Mysuru", "Rain likely", "2026-10-25T10:00:00+05:30"),
            )
            connection.execute(
                "INSERT INTO prediction_history (user_id,district,prediction,created_at) VALUES (?,?,?,?)",
                (other_id, "Ballari", "Rain less likely", "2026-10-25T11:00:00+05:30"),
            )
        self.post_form("/clear-history", {"confirm": "no"})
        self.assertEqual(self.count_records(), 2)
        self.post_form("/clear-history", {"confirm": "yes"})
        self.assertEqual(self.count_records(), 1)

    @patch("app.get_weather")
    def test_home_and_dashboard_render_as_separate_pages(self, get_weather):
        get_weather.return_value = self.weather
        landing = self.client.get("/")
        self.assertEqual(landing.status_code, 200)
        self.assertIn(b"Smart Rainfall</span><span>Prediction for a</span><span>Better Tomorrow.", landing.data)
        self.assertNotIn(b"Weather dashboard", landing.data)
        self.assertNotIn(b"prediction-form", landing.data)
        self.assertIn(b"Bengaluru Urban", landing.data)
        self.assertIn(b"data-live-clock", landing.data)
        self.assertEqual(self.client.get("/home").status_code, 302)
        with patch("app.get_weather", return_value=self.weather):
            dashboard = self.client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Weather dashboard", dashboard.data)
        self.assertNotIn(b"landing-title", dashboard.data)

    @patch("app.get_weather")
    def test_dashboard_history_and_about_pages_render(self, get_weather):
        get_weather.return_value = self.weather
        home = self.client.get("/dashboard")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"YOUR FIELD PROFILE", home.data)
        self.assertIn(b"owner@example.com", home.data)
        self.assertEqual(self.client.get("/history").status_code, 200)
        self.assertEqual(self.client.get("/about").status_code, 200)

    def test_language_selector_is_available_in_header_and_settings(self):
        with self.client.session_transaction() as user_session:
            user_session["ui_language"] = "kn"
        dashboard = self.client.get("/dashboard")
        settings = self.client.get("/settings")
        self.assertIn(b'<html lang="kn"', dashboard.data)
        self.assertIn(b'data-open-language-modal', dashboard.data)
        self.assertIn(b'name="language"', dashboard.data)
        self.assertIn(b"data-language-form", settings.data)
        self.assertIn(app.translate("Language preference", "kn").encode(), settings.data)

    def test_all_site_languages_remain_selectable(self):
        for code in app.LANGUAGES:
            with self.subTest(language=code):
                response = self.post_form("/set-language", {"language": code, "next": "/settings"})
                self.assertEqual(response.status_code, 302)
                page = self.client.get("/settings")
                self.assertIn(f'<html lang="{code}"'.encode(), page.data)
                self.assertIn(f'name="language" value="{code}"'.encode(), page.data)
                self.assertIn(app.LANGUAGES[code].encode("utf-8"), page.data)
                if code == "ur":
                    self.assertIn(b'dir="rtl"', page.data)

    def test_language_menu_has_mobile_viewport_limits(self):
        css = (app.BASE_DIR / "static" / "css" / "style.css").read_text(encoding="utf-8")
        template = self.client.get("/settings").data
        self.assertIn(b'id="language-modal"', template)
        self.assertIn(b'role="dialog" aria-modal="true"', template)
        self.assertIn("inset:0; display:grid; width:100vw; height:100vh; height:100dvh", css)
        self.assertIn("width:min(90vw,380px)", css)
        self.assertIn("overflow-y:auto", css)
        # The centered panel resolves to 337.5px at 375px and 380px at 1440px.
        self.assertEqual(min(0.9 * 375, 380), 337.5)
        self.assertEqual(min(0.9 * 1440, 380), 380)

    def test_translator_page_is_authenticated_and_shows_supported_languages(self):
        response = self.client.get("/translator")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Language Translator", response.data)
        for language in ("Kannada", "English", "Hindi", "Telugu", "Tamil", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Odia", "Urdu"):
            self.assertIn(language.encode(), response.data)

    @patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "test-server-key"})
    @patch("app.requests.post")
    def test_requested_indian_language_pairs_are_sent_to_google_and_saved(self, post):
        post.return_value.json.return_value = {
            "data": {"translations": [{"translatedText": "mocked test result"}]}
        }
        pairs = [
            ("kn", "en", "à²®à²³à³†", "Rain"),
            ("en", "kn", "Rain", "à²®à²³à³†"),
            ("hi", "kn", "à¤¬à¤¾à¤°à¤¿à¤¶", "à²®à²³à³†"),
            ("kn", "hi", "à²®à²³à³†", "à¤¬à¤¾à¤°à¤¿à¤¶"),
            ("en", "te", "Rain", "à°µà°°à±à°·à°‚"),
            ("te", "en", "à°µà°°à±à°·à°‚", "Rain"),
        ]
        for source, target, phrase, _expected_example in pairs:
            with self.subTest(source=source, target=target):
                response = self.post_translation({"text": phrase, "source": source, "target": target})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["translated_text"], "mocked test result")
                self.assertEqual(post.call_args.kwargs["json"]["source"], source)
                self.assertEqual(post.call_args.kwargs["json"]["target"], target)
                self.assertEqual(post.call_args.kwargs["params"]["key"], "test-server-key")
                self.assertNotIn("test-server-key", response.get_data(as_text=True))
        with app.database_connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM translation_history").fetchone()[0], len(pairs))

    @patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "test-server-key"})
    @patch("app.requests.post")
    def test_all_supported_translation_languages_are_accepted(self, post):
        post.return_value.json.return_value = {
            "data": {"translations": [{"translatedText": "translated"}]}
        }
        with patch.object(app, "TRANSLATION_RATE_LIMIT", len(app.TRANSLATION_LANGUAGES) + 1):
            for code in app.TRANSLATION_LANGUAGES:
                with self.subTest(language=code):
                    target = "kn" if code == "en" else "en"
                    response = self.post_translation({"text": "rain", "source": code, "target": target})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(post.call_args.kwargs["json"]["source"], code)

    @patch.dict("os.environ", {}, clear=False)
    def test_translation_without_provider_key_returns_setup_error_without_fake_result(self):
        import os
        old_key = os.environ.pop("GOOGLE_TRANSLATE_API_KEY", None)
        try:
            response = self.post_translation({"text": "Rain", "source": "en", "target": "kn"})
        finally:
            if old_key is not None:
                os.environ["GOOGLE_TRANSLATE_API_KEY"] = old_key
        self.assertEqual(response.status_code, 503)
        self.assertIn("GOOGLE_TRANSLATE_API_KEY", response.json["error"])
        self.assertNotIn("translated_text", response.json)

    def test_translation_input_language_and_csrf_validation(self):
        token = self.client.session_transaction
        with token() as user_session:
            csrf = user_session["_csrf_token"]
        invalid = self.client.post(
            "/api/translate", json={"text": "hello", "source": "en", "target": "xx"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(invalid.status_code, 400)
        too_long = self.client.post(
            "/api/translate", json={"text": "x" * (app.TRANSLATION_MAX_CHARS + 1), "source": "en", "target": "kn"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(too_long.status_code, 400)
        missing_csrf = self.client.post(
            "/api/translate", json={"text": "hello", "source": "en", "target": "kn"},
        )
        self.assertEqual(missing_csrf.status_code, 400)

    @patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "test-server-key"})
    @patch("app.requests.post")
    def test_translation_endpoint_rate_limit_and_provider_failure(self, post):
        post.return_value.json.return_value = {
            "data": {"translations": [{"translatedText": "à²®à²³à³†"}]}
        }
        with patch.object(app, "TRANSLATION_RATE_LIMIT", 1):
            first = self.post_translation({"text": "rain", "source": "en", "target": "kn"})
            second = self.post_translation({"text": "rain", "source": "en", "target": "kn"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(post.call_count, 1)

    def test_translation_history_clear_is_account_scoped(self):
        with app.database_connection() as connection:
            other_id = connection.execute(
                "INSERT INTO users (email,password_hash,created_at) VALUES (?,?,?)",
                ("other@example.com", generate_password_hash("password123"), "2026-10-26T10:00:00+05:30"),
            ).lastrowid
            connection.executemany(
                """INSERT INTO translation_history
                   (user_id,source_language,target_language,source_text,translated_text,created_at)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (self.user_id, "en", "kn", "rain", "à²®à²³à³†", "2026-10-26T10:00:00+05:30"),
                    (other_id, "en", "hi", "rain", "à¤¬à¤¾à¤°à¤¿à¤¶", "2026-10-26T10:00:00+05:30"),
                ],
            )
        self.post_form("/clear-translation-history", {"confirm": "yes"})
        with app.database_connection() as connection:
            remaining = connection.execute(
                "SELECT user_id FROM translation_history"
            ).fetchall()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["user_id"], other_id)

    def test_registration_stores_a_hash_and_login_authenticates(self):
        with self.client.session_transaction() as user_session:
            user_session.clear()
        response = self.post_form("/register", {
            "email": "new@example.com", "password": "securepass1",
            "confirm_password": "securepass1",
        })
        self.assertEqual(response.status_code, 302)
        with app.database_connection() as connection:
            user = connection.execute(
                "SELECT password_hash FROM users WHERE email = ?", ("new@example.com",)
            ).fetchone()
        self.assertTrue(check_password_hash(user["password_hash"], "securepass1"))
        self.assertNotEqual(user["password_hash"], "securepass1")
        response = self.post_form("/login", {
            "email": "NEW@example.com", "password": "securepass1",
        })
        self.assertEqual(response.status_code, 302)
        with self.client.session_transaction() as user_session:
            self.assertEqual(user_session["user_email"], "new@example.com")

    def test_private_pages_redirect_to_login_when_signed_out(self):
        with self.client.session_transaction() as user_session:
            user_session.clear()
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.location)

    def test_logout_returns_user_to_login(self):
        response = self.post_form("/logout", {})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.location)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_main_page_clear_history_redirects_back_home(self):
        with app.database_connection() as connection:
            connection.execute(
                "INSERT INTO prediction_history (user_id,district,prediction,created_at) VALUES (?,?,?,?)",
                (self.user_id, "Mysuru", "Rain likely", "2026-10-25T10:00:00+05:30"),
            )
        response = self.post_form(
            "/clear-history", {"confirm": "yes", "return_to": "home"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/dashboard")
        self.assertEqual(self.count_records(), 0)

    def test_history_only_shows_the_signed_in_users_records(self):
        with app.database_connection() as connection:
            other = connection.execute(
                "INSERT INTO users (email,password_hash,created_at) VALUES (?,?,?)",
                ("other@example.com", generate_password_hash("password123"), "2026-10-26T10:00:00+05:30"),
            ).lastrowid
            connection.executemany(
                "INSERT INTO prediction_history (user_id,district,prediction,created_at) VALUES (?,?,?,?)",
                [
                    (self.user_id, "Mysuru", "Rain likely", "2026-10-25T10:00:00+05:30"),
                    (other, "Ballari", "Rain less likely", "2026-10-25T11:00:00+05:30"),
                ],
            )
        response = self.client.get("/history")
        self.assertIn(b"Mysuru", response.data)
        self.assertNotIn(b"Ballari", response.data)

    def test_state_changing_post_without_csrf_token_is_rejected(self):
        response = self.client.post("/clear-history", data={"confirm": "yes"})
        self.assertEqual(response.status_code, 400)

    def test_login_query_payload_cannot_bypass_authentication(self):
        with self.client.session_transaction() as user_session:
            user_session.clear()
        response = self.post_form("/login", {
            "email": "' OR 1=1 --", "password": "password123",
        })
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as user_session:
            self.assertNotIn("user_id", user_session)


if __name__ == "__main__":
    unittest.main()
