import tempfile
import unittest
import secrets
from pathlib import Path
from unittest.mock import patch

import app
import requests
from werkzeug.security import check_password_hash, generate_password_hash


class FakePipeline:
    classes_ = [0, 1]

    def predict(self, _features):
        return [1]

    def predict_proba(self, _features):
        return [[0.2, 0.8]]


class AgriRainAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        app.app.config.update(TESTING=True, DATABASE=str(Path(self.temp_dir.name) / "test.db"))
        app.init_db()
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
        self.weather = {
            "forecast_date": "2026-10-25", "min_temp": 19.0,
            "max_temp": 29.0, "rainfall": 2.5, "humidity": 72.0,
        }

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

    @patch("app.load_model", return_value=({"pipeline": FakePipeline(), "features": app.FEATURES}, None))
    @patch("app.get_weather")
    def test_prediction_is_saved_once_and_refresh_does_not_resubmit(self, get_weather, _model):
        get_weather.return_value = self.weather
        response = self.post_form("/", {"district": "Mysuru"})
        response = self.client.get(response.location, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rain likely", response.data)
        self.assertEqual(self.count_records(), 1)
        self.client.get("/")
        self.assertEqual(self.count_records(), 1)

    @patch("app.get_weather")
    def test_end_to_end_real_model_prediction_is_saved(self, get_weather):
        get_weather.return_value = self.weather
        response = self.post_form("/", {"district": "Bengaluru"})
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

    @patch("app.get_weather")
    def test_invalid_district_is_not_saved(self, get_weather):
        get_weather.return_value = self.weather
        response = self.post_form("/", {"district": "Not a district"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.count_records(), 0)

    @patch("app.load_model", return_value=(None, "Model unavailable"))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_weather_timeout_is_shown_without_server_error(self, _weather, _model):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Weather forecast is temporarily unavailable", response.data)

    @patch("app.load_model", return_value=({"pipeline": FakePipeline(), "features": app.FEATURES}, None))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_manual_observations_allow_prediction_when_forecast_is_down(self, _weather, _model):
        response = self.post_form("/", {
            "district": "Mysuru", "observed_min_temp": "19.0",
            "observed_max_temp": "29.0", "observed_rainfall": "2.5",
            "observed_humidity": "72",
        })
        self.assertEqual(response.status_code, 302)
        page = self.client.get(response.location, follow_redirects=True)
        self.assertIn(b"Rain likely", page.data)
        self.assertIn(b"user-entered observations", page.data)
        self.assertEqual(self.count_records(), 1)

    @patch("app.load_model", return_value=({"pipeline": FakePipeline(), "features": app.FEATURES}, None))
    @patch("app.get_weather", side_effect=requests.Timeout)
    def test_missing_manual_observation_does_not_predict(self, _weather, _model):
        response = self.post_form("/", {
            "district": "Mysuru", "observed_min_temp": "19.0",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"all four valid weather values", response.data)
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
    def test_home_history_and_about_pages_render(self, get_weather):
        get_weather.return_value = self.weather
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"YOUR FIELD PROFILE", home.data)
        self.assertIn(b"owner@example.com", home.data)
        self.assertEqual(self.client.get("/history").status_code, 200)
        self.assertEqual(self.client.get("/about").status_code, 200)

    def test_translator_page_is_authenticated_and_shows_supported_languages(self):
        response = self.client.get("/translator")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Language Translator", response.data)
        for language in ("Kannada", "English", "Hindi", "Telugu", "Tamil", "Malayalam", "Marathi", "Bengali", "Gujarati", "Punjabi", "Urdu"):
            self.assertIn(language.encode(), response.data)

    @patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "test-server-key"})
    @patch("app.requests.post")
    def test_requested_indian_language_pairs_are_sent_to_google_and_saved(self, post):
        post.return_value.json.return_value = {
            "data": {"translations": [{"translatedText": "mocked test result"}]}
        }
        pairs = [
            ("kn", "en", "ಮಳೆ", "Rain"),
            ("en", "kn", "Rain", "ಮಳೆ"),
            ("hi", "kn", "बारिश", "ಮಳೆ"),
            ("kn", "hi", "ಮಳೆ", "बारिश"),
            ("en", "te", "Rain", "వర్షం"),
            ("te", "en", "వర్షం", "Rain"),
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
            "data": {"translations": [{"translatedText": "ಮಳೆ"}]}
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
                    (self.user_id, "en", "kn", "rain", "ಮಳೆ", "2026-10-26T10:00:00+05:30"),
                    (other_id, "en", "hi", "rain", "बारिश", "2026-10-26T10:00:00+05:30"),
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
        self.assertEqual(self.client.get("/").status_code, 302)

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
        self.assertEqual(response.location, "/")
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
