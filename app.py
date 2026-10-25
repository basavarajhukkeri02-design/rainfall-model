"""AgriRain Flask application. Run with `python app.py` for local development."""

from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import secrets
import sqlite3
import re
import threading
import webbrowser

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get("AGRIRAIN_DATABASE", BASE_DIR / "prediction_history.db"))
MODEL_PATH = BASE_DIR / "rainfall_model.pkl"
FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm"]
DISTRICTS = {
    "Bengaluru": (12.9716, 77.5946), "Belagavi": (15.8497, 74.4977),
    "Dharwad": (15.4589, 75.0078), "Vijayapura": (16.8302, 75.7100),
    "Kalaburagi": (17.3297, 76.8343), "Mysuru": (12.2958, 76.6394),
    "Shivamogga": (13.9299, 75.5681), "Ballari": (15.1394, 76.9214),
    "Mangaluru": (12.9141, 74.8560), "Tumakuru": (13.3379, 77.1173),
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["DATABASE"] = str(DATABASE)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
)


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
        supplied = request.form.get("_csrf_token", "")
        expected = session.get("_csrf_token", "")
        if not supplied or not secrets.compare_digest(supplied, expected):
            abort(400)


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": session.get("_csrf_token", "")}


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
    """Load only the model bundle produced by this project's training script."""
    if not MODEL_PATH.is_file():
        return None, "The trained model is missing. Run `python train_model.py` first."
    try:
        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict) or bundle.get("features") != FEATURES:
            raise ValueError("Model bundle has an unexpected feature order.")
        pipeline = bundle.get("pipeline")
        if not hasattr(pipeline, "predict"):
            raise ValueError("Model bundle does not contain a usable pipeline.")
        return bundle, None
    except Exception:
        app.logger.exception("Could not load the rainfall model")
        return None, "The trained model could not be loaded. Retrain it with `python train_model.py`."


def get_weather(district):
    """Return today's Open-Meteo forecast; missing or invalid values are errors."""
    latitude, longitude = DISTRICTS[district]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude, "longitude": longitude,
            "daily": "temperature_2m_min,temperature_2m_max,rain_sum",
            "hourly": "relative_humidity_2m", "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        },
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    daily, hourly = payload["daily"], payload["hourly"]
    forecast_date = daily["time"][0]
    humidity_times = hourly["time"]
    humidity_values = hourly["relative_humidity_2m"]
    target_hour = f"{forecast_date}T15:00"
    try:
        humidity_index = humidity_times.index(target_hour)
        humidity_3pm = humidity_values[humidity_index]
    except (ValueError, IndexError, TypeError) as error:
        raise ValueError("3 PM humidity forecast is unavailable") from error
    result = {
        "forecast_date": forecast_date,
        "min_temp": daily["temperature_2m_min"][0],
        "max_temp": daily["temperature_2m_max"][0],
        "rainfall": daily["rain_sum"][0],
        "humidity": humidity_3pm,
    }
    for key in ("min_temp", "max_temp", "rainfall", "humidity"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError("Forecast contains a missing number")
    return result


def prediction_for(bundle, weather):
    row = pd.DataFrame([[
        weather["min_temp"], weather["max_temp"],
        weather["rainfall"], weather["humidity"],
    ]], columns=FEATURES)
    pipeline = bundle["pipeline"]
    label = int(pipeline.predict(row)[0])
    classes = list(getattr(pipeline, "classes_", []))
    probability = None
    if hasattr(pipeline, "predict_proba") and 1 in classes:
        candidate = float(pipeline.predict_proba(row)[0][classes.index(1)])
        if np.isfinite(candidate) and 0 <= candidate <= 1:
            probability = candidate
    return ("Rain likely" if label == 1 else "Rain less likely"), probability


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
            session.clear()
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
                session.clear()
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
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    selected = request.form.get("district", "Bengaluru") if request.method == "POST" else request.args.get("district", "Bengaluru")
    invalid_district = selected not in DISTRICTS
    if invalid_district:
        flash("Please select a valid Karnataka district.", "error")
        selected = "Bengaluru"

    weather = None
    model_bundle, model_error = load_model()
    prediction_result = session.pop("prediction_result", None)
    try:
        weather = get_weather(selected)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        app.logger.info("Weather forecast unavailable for %s", selected, exc_info=True)
        flash("Weather forecast is temporarily unavailable. Please try again later.", "error")

    if request.method == "POST":
        if invalid_district:
            pass
        elif not request.form.get("district"):
            flash("Choose a district before requesting a prediction.", "error")
        elif weather is None:
            flash("No prediction was made because weather data is unavailable.", "error")
        elif model_bundle is None:
            flash(model_error, "error")
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
                    "probability": probability, "weather": weather,
                }
                return redirect(url_for("home", district=selected))
            except (ValueError, KeyError, TypeError, sqlite3.Error):
                app.logger.exception("Prediction could not be completed")
                flash("Prediction could not be saved. Please retry.", "error")

    return render_template(
        "index.html", districts=DISTRICTS, district=selected, weather=weather,
        prediction_result=prediction_result, history=recent_predictions(session["user_id"]),
        model_error=model_error, profile=profile_summary(session["user_id"]),
    )


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
