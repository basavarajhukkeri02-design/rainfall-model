"""AgriRain Flask application. Run with `python app.py` for local development."""

from datetime import datetime, timedelta
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from collections import defaultdict, deque
import html
import os
from pathlib import Path
import secrets
import sqlite3
import re
import threading
import time
import webbrowser
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, abort, flash as flask_flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from translations import LANGUAGES, translate


BASE_DIR = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get("AGRIRAIN_DATABASE", BASE_DIR / "prediction_history.db"))
MODEL_PATH = BASE_DIR / "karnataka_model" / "rainfall_model.pkl"
OPEN_METEO_URL = os.environ.get(
    "OPEN_METEO_API_URL", "https://api.open-meteo.com/v1/forecast"
)
KARNATAKA_HISTORY_FEATURES = [
    "Rainfall_Lag_1", "Rainfall_Lag_2", "Rainfall_Lag_3", "Rainfall_Lag_7",
    "Rolling_Mean_3", "Rolling_Sum_7", "Month", "Day_Of_Year", "District",
]
FEATURES = KARNATAKA_HISTORY_FEATURES
TRANSLATION_URL = "https://translation.googleapis.com/language/translate/v2"
TRANSLATION_MAX_CHARS = 1000
TRANSLATION_RATE_LIMIT = 10
TRANSLATION_RATE_WINDOW = 60
TRANSLATION_LANGUAGES = {
    "kn": "ಕನ್ನಡ", "en": "English", "hi": "हिन्दी", "te": "తెలుగు",
    "ta": "தமிழ்", "ml": "മലയാളം", "mr": "मराठी", "bn": "বাংলা",
    "gu": "ગુજરાતી", "pa": "ਪੰਜਾਬੀ", "ur": "اردو",
}
DISTRICTS = {
    # Keys match the district labels used by the Karnataka training dataset.
    # Coordinates point to representative district headquarters/towns.
    "Bagalkot": (16.1725, 75.6557),
    "Ballari": (15.1394, 76.9214),
    "Belagavi": (15.8497, 74.4977),
    "Bengaluru Rural": (13.2847, 77.6078),
    "Bengaluru Urban": (12.9700, 77.6536),
    "Bidar": (17.9149, 77.5046),
    "Chamrajnagar": (12.0526, 77.2865),
    "Chikkaballapura": (13.4324, 77.7280),
    "Chikkamagaluru": (13.3153, 75.7754),
    "Chitradurga": (14.2306, 76.3985),
    "Dakshina Kannada": (12.9141, 74.8560),
    "Davangere": (14.4663, 75.9238),
    "Dharwad": (15.4589, 75.0078),
    "Gadag": (15.4325, 75.6380),
    "Hassan": (13.0097, 76.1029),
    "Haveri": (14.6610, 75.4345),
    "Kalaburagi": (17.3297, 76.8343),
    "Kodagu": (12.4240, 75.8070),
    "Kolar": (13.1357, 78.1326),
    "Koppal": (15.3465, 76.1670),
    "Mandya": (12.5222, 76.9009),
    "Mysuru": (12.2958, 76.6394),
    "Raichur": (16.2120, 77.3439),
    "Ramanagara": (12.7094, 77.2807),
    "Shivamogga": (13.9299, 75.5681),
    "Tumakuru": (13.3390, 77.1140),
    "Udupi": (13.3408, 74.7421),
    "Uttara Kannada": (14.8185, 74.1416),
    "Vijayanagara": (15.2689, 76.3909),
    "Vijayapura": (16.8302, 75.7100),
    "Yadgir": (16.7604, 77.1428),
}
DISTRICT_LABELS = {
    "Chamrajnagar": "Chamarajanagar",
    "Davangere": "Davanagere",
}
WEATHER_CACHE_TTL_SECONDS = 20
_weather_cache = {}
_weather_cache_lock = threading.Lock()
_weather_district_locks = {district: threading.Lock() for district in DISTRICTS}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["DATABASE"] = str(DATABASE)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
)
_translation_request_times = defaultdict(deque)
_translation_rate_lock = threading.Lock()


def flash(message, category="message", *args, **kwargs):
    """Localize server-generated notices using the visitor's session language."""
    language = session.get("ui_language", "en")
    flask_flash(translate(message, language), category, *args, **kwargs)


def get_db():
    """Open a row-based SQLite connection for one operation."""
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(app.config["DATABASE"])
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def database_connection():
    """Commit a successful operation and always release its SQLite handle."""
    connection = get_db()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@app.before_request
def protect_post_requests():
    """Require a session-bound CSRF token for every state-changing form POST."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)
    if request.method == "POST":
        supplied = (
            request.headers.get("X-CSRF-Token", "")
            if request.is_json else request.form.get("_csrf_token", "")
        )
        expected = session.get("_csrf_token", "")
        if not supplied or not secrets.compare_digest(supplied, expected):
            if request.is_json:
                return jsonify(error=translate("Your session expired. Refresh the page and try again.", session.get("ui_language", "en"))), 400
            abort(400)


@app.context_processor
def inject_csrf_token():
    language = session.get("ui_language", "en")
    return {
        "csrf_token": session.get("_csrf_token", ""),
        "ui_language": language,
        "ui_languages": LANGUAGES,
        "_": lambda message: translate(message, language),
    }


def init_db():
    """Create account/history tables and migrate the original anonymous history."""
    with database_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                district TEXT NOT NULL,
                prediction TEXT NOT NULL,
                rainfall_probability REAL,
                temperature_min REAL,
                temperature_max REAL,
                expected_rainfall REAL,
                humidity REAL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(prediction_history)")}
        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE prediction_history ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE"
            )
        old_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchone()
        if old_table:
            connection.execute("""
                INSERT INTO prediction_history (district, prediction, created_at)
                SELECT h.district, h.prediction, h.created_at FROM history h
                WHERE NOT EXISTS (
                    SELECT 1 FROM prediction_history p
                    WHERE p.district = h.district AND p.prediction = h.prediction
                      AND p.created_at = h.created_at
                )
            """)


def load_model():
    """Load and validate the Karnataka model used for district predictions."""
    if not MODEL_PATH.is_file():
        return None, "The Karnataka rainfall model is missing."
    try:
        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict) or bundle.get("features") != FEATURES:
            raise ValueError("Model bundle has an unexpected feature order.")
        pipeline = bundle.get("pipeline")
        threshold = bundle.get("threshold")
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Model bundle has an invalid classification threshold.")
        if bundle.get("positive_label") != 1:
            raise ValueError("Model bundle has an unexpected positive label.")
        if not hasattr(pipeline, "predict_proba"):
            raise ValueError("Model bundle does not contain a usable pipeline.")
        pipeline_features = getattr(pipeline, "feature_names_in_", None)
        if pipeline_features is not None and list(pipeline_features) != FEATURES:
            raise ValueError("Model pipeline has an unexpected feature order.")
        return bundle, None
    except Exception:
        app.logger.exception("Could not load the Karnataka rainfall model")
        return None, "The Karnataka rainfall model could not be loaded."


def get_weather(district):
    """Return a short-lived district forecast, coalescing duplicate requests."""
    if district not in DISTRICTS:
        raise KeyError(f"Unknown Karnataka district: {district}")

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    with _weather_cache_lock:
        district_lock = _weather_district_locks[district]
        cached = _weather_cache.get(district)
        if cached and time.monotonic() - cached["stored_at"] < WEATHER_CACHE_TTL_SECONDS:
            if cached["weather"].get("forecast_date") == today:
                return deepcopy(cached["weather"])

    # Concurrent requests for the same district share one upstream request.
    with district_lock:
        with _weather_cache_lock:
            cached = _weather_cache.get(district)
            if cached and time.monotonic() - cached["stored_at"] < WEATHER_CACHE_TTL_SECONDS:
                if cached["weather"].get("forecast_date") == today:
                    return deepcopy(cached["weather"])

        weather = _fetch_weather(district)
        with _weather_cache_lock:
            _weather_cache[district] = {
                "stored_at": time.monotonic(),
                "weather": deepcopy(weather),
            }
        return weather


def _fetch_weather(district):
    """Fetch and validate one complete Open-Meteo forecast response."""
    latitude, longitude = DISTRICTS[district]
    params = {
        "latitude": latitude, "longitude": longitude,
        "daily": "temperature_2m_min,temperature_2m_max,rain_sum,weather_code",
        "hourly": "temperature_2m,relative_humidity_2m", "past_days": 7, "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }
    response = None
    for attempt in range(3):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=(4, 12))
            response.raise_for_status()
            break
        except requests.HTTPError as error:
            status = response.status_code if response is not None else None
            if (status != 429 and (status is None or status < 500)) or attempt == 2:
                raise
            app.logger.warning(
                "Open-Meteo transient HTTP %s for %s; retry %s/2",
                status, district, attempt + 1,
            )
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == 2:
                raise
            app.logger.warning(
                "Open-Meteo network failure for %s; retry %s/2",
                district, attempt + 1,
            )
        time.sleep(0.2 * (2 ** attempt))

    app.logger.info("Open-Meteo response status=%s district=%s", response.status_code, district)
    payload = response.json()
    try:
        daily, hourly = payload["daily"], payload["hourly"]
        daily_dates = daily["time"]
        if not isinstance(daily_dates, list) or not all(isinstance(day, str) for day in daily_dates):
            raise ValueError("Open-Meteo response contains invalid daily dates")
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        forecast_date = today if today in daily_dates else None
        if forecast_date is None:
            raise ValueError("Today's forecast date is missing")
        forecast_index = daily_dates.index(forecast_date)
        humidity_times = hourly["time"]
        humidity_values = hourly["relative_humidity_2m"]
        min_temp = daily["temperature_2m_min"][forecast_index]
        max_temp = daily["temperature_2m_max"][forecast_index]
        rainfall = daily["rain_sum"][forecast_index]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Open-Meteo response is missing required forecast fields") from error
    target_hour = f"{forecast_date}T15:00"
    try:
        humidity_index = humidity_times.index(target_hour)
        humidity_3pm = humidity_values[humidity_index]
    except (ValueError, IndexError, TypeError) as error:
        raise ValueError("3 PM humidity forecast is unavailable") from error
    current_hour = datetime.now(ZoneInfo("Asia/Kolkata")).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    current_temp = None
    try:
        temperature_values = hourly.get("temperature_2m", [])
        current_temp = float(temperature_values[humidity_times.index(current_hour)])
        if not np.isfinite(current_temp):
            current_temp = None
    except (ValueError, IndexError, TypeError):
        pass
    weather_code = None
    try:
        raw_weather_code = daily.get("weather_code", [])[forecast_index]
        if raw_weather_code is not None:
            weather_code = int(raw_weather_code)
    except (IndexError, TypeError, ValueError):
        pass
    local_hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    weather_condition, weather_condition_label = weather_condition_for_code(
        weather_code, is_night=local_hour < 6 or local_hour >= 18
    )
    result = {
        "district": district,
        "forecast_date": forecast_date,
        "prediction_date": (datetime.strptime(forecast_date, "%Y-%m-%d").date() + timedelta(days=1)).isoformat(),
        "weather_code": weather_code,
        "weather_condition": weather_condition,
        "weather_condition_label": weather_condition_label,
        "current_temp": current_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "rainfall": rainfall,
        "humidity": humidity_3pm,
    }
    for key in ("min_temp", "max_temp", "rainfall", "humidity"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError("Forecast contains a missing number")
    if result["min_temp"] > result["max_temp"]:
        raise ValueError("Forecast minimum temperature exceeds its maximum")
    if result["rainfall"] < 0 or not 0 <= result["humidity"] <= 100:
        raise ValueError("Forecast rainfall or humidity is outside its valid range")
    history_features, history_error = build_karnataka_history_features(
        daily_dates, daily.get("rain_sum"), forecast_date, district
    )
    result["karnataka_history_features"] = history_features
    result["karnataka_history_error"] = history_error
    forecasts = []
    daily_min = daily.get("temperature_2m_min", [])
    daily_max = daily.get("temperature_2m_max", [])
    daily_rain = daily.get("rain_sum", [])
    daily_codes = daily.get("weather_code", [])
    for index in range(forecast_index, min(forecast_index + 7, len(daily_dates))):
        try:
            low, high, rain = float(daily_min[index]), float(daily_max[index]), float(daily_rain[index])
            if not all(np.isfinite(value) for value in (low, high, rain)) or low > high or rain < 0:
                continue
            code = int(daily_codes[index]) if daily_codes[index] is not None else None
            condition, label = weather_condition_for_code(code)
        except (IndexError, TypeError, ValueError):
            continue
        forecasts.append({
            "date": daily_dates[index], "min_temp": low, "max_temp": high,
            "rainfall": rain, "weather_code": code,
            "weather_condition": condition, "weather_condition_label": label,
        })
    result["forecast_days"] = forecasts
    return result


def weather_condition_for_code(weather_code, is_night=False):
    """Map Open-Meteo WMO codes to lightweight weather-icon states."""
    if weather_code == 0:
        if is_night:
            return "clear-night", "Clear night"
        return "sunny", "Sunny"
    if weather_code in {1, 2}:
        return "partly-cloudy", "Partly cloudy"
    if weather_code == 3:
        return "cloudy", "Cloudy"
    if weather_code in {45, 48}:
        return "cloudy", "Fog"
    if weather_code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "rainy", "Rain"
    if weather_code in {71, 73, 75, 77, 85, 86}:
        return "snowy", "Snow"
    if weather_code in {95, 96, 99}:
        return "thunderstorm", "Thunderstorm"
    return "unknown", "Weather condition unavailable"


def build_karnataka_history_features(daily_dates, daily_rainfall, feature_date, district):
    """Build training-ordered lag inputs only when all seven prior days are valid."""
    if (
        not isinstance(daily_dates, list)
        or not all(isinstance(day, str) for day in daily_dates)
        or not isinstance(daily_rainfall, list)
    ):
        return None, "Daily rainfall history is missing"
    if len(daily_dates) != len(daily_rainfall) or len(set(daily_dates)) != len(daily_dates):
        return None, "Daily rainfall history has inconsistent dates or values"
    rainfall_by_date = {}
    for day, value in zip(daily_dates, daily_rainfall):
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(amount) and amount >= 0:
            rainfall_by_date[day] = amount

    try:
        base_date = datetime.strptime(feature_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None, "Forecast date is invalid"

    prior_dates = [(base_date - timedelta(days=offset)).isoformat() for offset in range(1, 8)]
    if any(day not in rainfall_by_date for day in prior_dates):
        return None, "Open-Meteo did not return valid rainfall for all seven prior days"

    # First element is the previous day, matching pandas shift(1) in training.
    prior_rainfall = [rainfall_by_date[day] for day in prior_dates]
    feature_values = {
        "Rainfall_Lag_1": prior_rainfall[0],
        "Rainfall_Lag_2": prior_rainfall[1],
        "Rainfall_Lag_3": prior_rainfall[2],
        "Rainfall_Lag_7": prior_rainfall[6],
        "Rolling_Mean_3": float(np.mean(prior_rainfall[:3])),
        "Rolling_Sum_7": float(np.sum(prior_rainfall)),
        "Month": base_date.month,
        "Day_Of_Year": base_date.timetuple().tm_yday,
        "District": district,
    }
    # Dict insertion order follows the model training feature order.
    return {name: feature_values[name] for name in KARNATAKA_HISTORY_FEATURES}, None


def prediction_for(bundle, weather):
    feature_values = weather.get("karnataka_history_features")
    if not isinstance(feature_values, dict):
        detail = weather.get("karnataka_history_error") or "Seven-day district rainfall history is unavailable"
        raise ValueError(detail)
    feature_order = bundle.get("features")
    if feature_order != FEATURES or list(feature_values) != feature_order:
        raise ValueError("Prediction features do not match the Karnataka model bundle")
    if feature_values.get("District") not in DISTRICTS:
        raise ValueError("Prediction contains an unknown district")
    if feature_values["District"] != weather.get("district"):
        raise ValueError("Rainfall history district does not match the selected district")
    numeric_features = [name for name in feature_order if name != "District"]
    try:
        for name in numeric_features:
            feature_values[name] = float(feature_values[name])
            if not np.isfinite(feature_values[name]):
                raise ValueError(f"Prediction feature {name} is missing or invalid")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Seven-day district rainfall features are incomplete or invalid") from error
    row = pd.DataFrame([[feature_values[name] for name in feature_order]], columns=feature_order)
    pipeline = bundle["pipeline"]
    classes = list(getattr(pipeline, "classes_", []))
    positive_label = bundle["positive_label"]
    if positive_label not in classes:
        raise ValueError("The model does not expose its configured positive class")
    probability = float(pipeline.predict_proba(row)[0][classes.index(positive_label)])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The model returned an invalid probability")
    label = int(probability >= float(bundle["threshold"]))
    return ("Rain likely" if label == positive_label else "Rain less likely"), probability


def read_manual_weather(form):
    """Read explicitly entered observations; never invent missing model inputs."""
    form_names = {
        "min_temp": "min_temp", "max_temp": "max_temp",
        "rainfall": "rainfall", "humidity": "humidity",
    }
    values = {}
    for key, name in form_names.items():
        raw = form.get(f"observed_{name}", "").strip()
        if not raw:
            raise ValueError("Manual observations are incomplete")
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError("Manual observations must be numeric") from error
        if not np.isfinite(value):
            raise ValueError("Manual observations must be finite numbers")
        values[key] = value
    if values["min_temp"] > values["max_temp"]:
        raise ValueError("Minimum temperature cannot exceed maximum temperature")
    if values["rainfall"] < 0 or not 0 <= values["humidity"] <= 100:
        raise ValueError("Rainfall must be nonnegative and humidity must be 0–100")
    return values


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return redirect(url_for("login"))
        with database_connection() as connection:
            user_exists = connection.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        if not user_exists:
            language = session.get("ui_language", "en")
            session.clear()
            session["ui_language"] = language
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def recent_predictions(user_id, limit=5):
    with database_connection() as connection:
        return connection.execute(
            "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()


def profile_summary(user_id):
    with database_connection() as connection:
        return connection.execute("""
            SELECT u.email, u.created_at, COUNT(p.id) AS prediction_count
            FROM users AS u
            LEFT JOIN prediction_history AS p ON p.user_id = u.id
            WHERE u.id = ?
            GROUP BY u.id, u.email, u.created_at
        """, (user_id,)).fetchone()


def translation_rate_limited(user_id):
    """Allow at most ten translations per user in a rolling one-minute window."""
    now = time.monotonic()
    with _translation_rate_lock:
        requests_for_user = _translation_request_times[user_id]
        while requests_for_user and now - requests_for_user[0] >= TRANSLATION_RATE_WINDOW:
            requests_for_user.popleft()
        if len(requests_for_user) >= TRANSLATION_RATE_LIMIT:
            return True
        requests_for_user.append(now)
        return False


def translate_with_google(text, source, target):
    """Call Google Cloud Translation Basic without exposing its key to clients."""
    api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Translation is not configured yet. Add a Google Cloud Translation API key to the server as GOOGLE_TRANSLATE_API_KEY.")

    response = requests.post(
        TRANSLATION_URL,
        params={"key": api_key},
        json={"q": text, "source": source, "target": target, "format": "text"},
        timeout=(4, 15),
    )
    if response.status_code in {400, 401, 403, 429}:
        app.logger.warning("Google Translation rejected a request (HTTP %s)", response.status_code)
        raise ValueError("The translation provider rejected this request. Check API access, billing, and quota settings.")
    response.raise_for_status()
    try:
        translated = response.json()["data"]["translations"][0]["translatedText"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ValueError("The translation provider returned an invalid response.") from error
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("The translation provider returned an empty result.")
    return html.unescape(translated)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Enter your email address and password.", "error")
        else:
            with database_connection() as connection:
                user = connection.execute(
                    "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
                ).fetchone()
            if user and check_password_hash(user["password_hash"], password):
                language = session.get("ui_language", "en")
                session.clear()
                session["ui_language"] = language
                session["user_id"] = user["id"]
                session["user_email"] = user["email"]
                return redirect(url_for("home"))
            flash("Email or password is incorrect.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        valid_email = len(email) <= 254 and re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)
        if not valid_email:
            flash("Enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Use a password with at least 8 characters.", "error")
        elif len(password) > 128:
            flash("Password must be 128 characters or fewer.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                with database_connection() as connection:
                    connection.execute(
                        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                        (email, generate_password_hash(password), datetime.now().astimezone().isoformat(timespec="seconds")),
                    )
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
    return render_template("register.html")


@app.post("/logout")
@login_required
def logout():
    language = session.get("ui_language", "en")
    session.clear()
    session["ui_language"] = language
    flash(translate("You have been signed out.", language), "success")
    return redirect(url_for("login"))


@app.post("/set-language")
def set_language():
    language = request.form.get("language", "")
    if language not in LANGUAGES:
        abort(400)
    session["ui_language"] = language
    destination = request.form.get("next", "/")
    parsed = urlsplit(destination)
    if not destination.startswith("/") or destination.startswith("//") or parsed.scheme or parsed.netloc:
        destination = "/"
    return redirect(destination)


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    selected = request.form.get("district", "Bengaluru Urban") if request.method == "POST" else request.args.get("district", "Bengaluru Urban")
    invalid_district = selected not in DISTRICTS
    if invalid_district:
        flash("Please select a valid Karnataka district.", "error")
        selected = "Bengaluru Urban"

    weather = None
    model_bundle, model_error = load_model()
    prediction_result = session.pop("prediction_result", None)
    try:
        weather = get_weather(selected)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        app.logger.warning("Weather forecast unavailable for %s", selected, exc_info=True)
        flash("Weather forecast is temporarily unavailable. Please try again later.", "error")

    if request.method == "POST":
        if invalid_district:
            pass
        elif not request.form.get("district"):
            flash("Choose a district before requesting a prediction.", "error")
        elif model_bundle is None:
            flash(model_error, "error")
        elif weather is None:
            flash("A current forecast and seven complete prior rainfall days are required for prediction.", "error")
        else:
            try:
                prediction, probability = prediction_for(model_bundle, weather)
                with database_connection() as connection:
                    connection.execute("""
                        INSERT INTO prediction_history
                        (user_id, district, prediction, rainfall_probability, temperature_min,
                         temperature_max, expected_rainfall, humidity, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session["user_id"], selected, prediction, probability, weather["min_temp"],
                        weather["max_temp"], weather["rainfall"], weather["humidity"],
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                    ))
                session["prediction_result"] = {
                    "district": selected, "prediction": prediction,
                    "probability": probability,
                    "input_source": "Open-Meteo rainfall history for the previous seven days",
                    "prediction_date": weather["prediction_date"],
                    "weather": weather,
                }
                return redirect(url_for("home", district=selected))
            except (KeyError, TypeError, ValueError):
                app.logger.exception("Prediction could not be completed")
                flash("Prediction is unavailable because complete recent rainfall history is missing or invalid.", "error")

    return render_template(
        "index.html", districts=DISTRICTS, district_labels=DISTRICT_LABELS,
        district=selected, weather=weather,
        prediction_result=prediction_result, history=recent_predictions(session["user_id"]),
        model_error=model_error, profile=profile_summary(session["user_id"]),
    )


@app.route("/home")
@login_required
def home_landing():
    return render_template("home.html")


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


@app.route("/history")
@login_required
def history():
    page = max(request.args.get("page", 1, type=int), 1)
    page_size = 20
    with database_connection() as connection:
        user_id = session["user_id"]
        total = connection.execute(
            "SELECT COUNT(*) FROM prediction_history WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (user_id, page_size, (page - 1) * page_size),
        ).fetchall()
    return render_template(
        "history.html", records=rows, page=page,
        pages=max((total + page_size - 1) // page_size, 1), total=total,
    )


@app.post("/clear-history")
@login_required
def clear_history():
    if request.form.get("confirm") != "yes":
        flash("History was not cleared because confirmation was not received.", "error")
        return redirect(url_for("history"))
    with database_connection() as connection:
        connection.execute("DELETE FROM prediction_history WHERE user_id = ?", (session["user_id"],))
    flash("Prediction history cleared.", "success")
    destination = "home" if request.form.get("return_to") == "home" else "history"
    return redirect(url_for(destination))


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/translator")
@login_required
def translator():
    with database_connection() as connection:
        translations = connection.execute(
            """SELECT source_language, target_language, source_text, translated_text, created_at
               FROM translation_history WHERE user_id = ? ORDER BY id DESC LIMIT 10""",
            (session["user_id"],),
        ).fetchall()
    return render_template(
        "translator.html", languages=TRANSLATION_LANGUAGES,
        max_chars=TRANSLATION_MAX_CHARS, translations=translations,
        translation_configured=bool(os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip()),
    )


@app.post("/api/translate")
def api_translate():
    """Validate, rate-limit, call the provider, and store a user's translation."""
    language = session.get("ui_language", "en")

    def localized(message):
        return translate(message, language)

    user_id = session.get("user_id")
    if not user_id:
        return jsonify(error=localized("Please sign in to use the translator.")), 401
    with database_connection() as connection:
        if not connection.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            language = session.get("ui_language", "en")
            session.clear()
            session["ui_language"] = language
            return jsonify(error=localized("Your session has expired. Please sign in again.")), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error=localized("Send a valid translation request.")), 400
    text = payload.get("text")
    source = payload.get("source")
    target = payload.get("target")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error=localized("Enter a word or sentence to translate.")), 400
    text = text.strip()
    if len(text) > TRANSLATION_MAX_CHARS:
        limit_message = localized("Please keep the text to {limit} characters or fewer.").replace("{limit}", str(TRANSLATION_MAX_CHARS))
        return jsonify(error=limit_message), 400
    if (not isinstance(source, str) or not isinstance(target, str)
            or source not in TRANSLATION_LANGUAGES or target not in TRANSLATION_LANGUAGES):
        return jsonify(error=localized("Choose a supported source and target language.")), 400
    if source == target:
        return jsonify(error=localized("Choose two different languages.")), 400
    if translation_rate_limited(user_id):
        return jsonify(error=localized("You have reached the limit of 10 translations per minute. Please wait and try again.")), 429, {
            "Retry-After": str(TRANSLATION_RATE_WINDOW),
        }

    try:
        translated = translate_with_google(text, source, target)
    except RuntimeError:
        app.logger.warning("Google Translation is not configured")
        return jsonify(error=localized("Translation is not configured yet. Add a Google Cloud Translation API key to the server as GOOGLE_TRANSLATE_API_KEY.")), 503
    except ValueError as error:
        return jsonify(error=localized(str(error))), 502
    except requests.RequestException:
        app.logger.warning("Google Translation service is currently unreachable")
        return jsonify(error=localized("The translation service is temporarily unavailable. Please try again shortly.")), 503

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with database_connection() as connection:
        connection.execute(
            """INSERT INTO translation_history
               (user_id, source_language, target_language, source_text, translated_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, source, target, text, translated, created_at),
        )
    return jsonify(
        translated_text=translated,
        source_language=source,
        target_language=target,
        created_at=created_at,
    )


@app.post("/clear-translation-history")
@login_required
def clear_translation_history():
    if request.form.get("confirm") != "yes":
        flash("Translation history was not cleared because confirmation was not received.", "error")
        return redirect(url_for("translator"))
    with database_connection() as connection:
        connection.execute(
            "DELETE FROM translation_history WHERE user_id = ?", (session["user_id"],)
        )
    flash("Translation history cleared.", "success")
    return redirect(url_for("translator"))


@app.errorhandler(500)
def internal_error(_error):
    return render_template("error.html"), 500


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html"), 404


@app.errorhandler(400)
def bad_request(_error):
    return render_template("error.html"), 400


init_db()

if __name__ == "__main__":
    if os.environ.get("AGRIRAIN_OPEN_BROWSER", "1") == "1":
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/login")).start()
    app.run(debug=False)
