"""Read-only live Open-Meteo and Karnataka model validation for all districts.

Run with ``python validate_live_forecasts.py``. The report is written to stdout;
the app import uses an in-memory database so no project data is changed.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import os
from threading import Lock
from zoneinfo import ZoneInfo


# app.py initializes its database when imported. Keep that initialization in
# memory and suppress bytecode writes so this validation does not touch files.
os.environ["AGRIRAIN_DATABASE"] = ":memory:"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import app  # noqa: E402  (environment must be set before importing app)
app.app.logger.disabled = True


def inspect_daily_history(payload, today):
    daily = payload.get("daily", {})
    daily_dates = daily.get("time", [])
    daily_rainfall = daily.get("rain_sum", [])
    if not isinstance(daily_dates, list) or not isinstance(daily_rainfall, list):
        return None, None, False
    rainfall_by_date = dict(zip(daily_dates, daily_rainfall))
    prior_dates = [
        (today - timedelta(days=offset)).isoformat()
        for offset in range(1, 8)
    ]
    valid_count = 0
    for date in prior_dates:
        try:
            amount = float(rainfall_by_date[date])
        except (KeyError, TypeError, ValueError):
            continue
        if amount >= 0 and amount < float("inf"):
            valid_count += 1
    date_range = [daily_dates[0], daily_dates[-1]] if daily_dates else None
    return valid_count, date_range, today.isoformat() in daily_dates


def run_validation():
    model, model_error = app.load_model()
    if model_error:
        return {
            "run_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            "threshold": None,
            "summary": {"districts_tested": 0, "passed": 0, "failed": len(app.DISTRICTS)},
            "error": model_error,
            "districts": [],
        }

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    original_get = app.requests.get
    lock = Lock()
    responses = {}

    def tracked_get(url, *, params, timeout):
        response = original_get(url, params=params, timeout=timeout)
        coordinate_key = (float(params["latitude"]), float(params["longitude"]))
        with lock:
            responses[coordinate_key] = response
        return response

    def check_district(district):
        coordinates = app.DISTRICTS[district]
        try:
            weather = app.get_weather(district)
            response = responses.get(coordinates)
            if response is None:
                raise RuntimeError("No HTTP response was recorded")

            history_count, date_range, today_available = inspect_daily_history(
                response.json(), today
            )
            label, probability = app.prediction_for(model, weather)
            features_match = list(weather["karnataka_history_features"]) == model["features"]
            target_date_matches = (
                weather["prediction_date"]
                == (today + timedelta(days=1)).isoformat()
            )
            passed = all((
                response.status_code == 200,
                weather["forecast_date"] == today.isoformat(),
                history_count == 7,
                today_available,
                features_match,
                target_date_matches,
            ))
            return {
                "district": district,
                "http_status": response.status_code,
                "forecast_date": weather["forecast_date"],
                "prediction_date": weather["prediction_date"],
                "daily_date_range": date_range,
                "local_today_available": today_available,
                "valid_prior_rainfall_days": f"{history_count}/7",
                "model_features_match": features_match,
                "prediction": label,
                "probability": probability,
                "status": "passed" if passed else "failed",
                "error": None if passed else "Live response did not satisfy all validation checks",
            }
        except Exception as error:
            response = responses.get(coordinates)
            history_count = None
            date_range = None
            today_available = False
            if response is not None:
                try:
                    history_count, date_range, today_available = inspect_daily_history(
                        response.json(), today
                    )
                except (ValueError, TypeError):
                    pass
            return {
                "district": district,
                "http_status": response.status_code if response else None,
                "forecast_date": None,
                "prediction_date": None,
                "daily_date_range": date_range,
                "local_today_available": today_available,
                "valid_prior_rainfall_days": f"{history_count}/7" if history_count is not None else None,
                "model_features_match": False,
                "prediction": None,
                "probability": None,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }

    app.requests.get = tracked_get
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            districts = list(executor.map(check_district, app.DISTRICTS))
    finally:
        app.requests.get = original_get

    passed = sum(row["status"] == "passed" for row in districts)
    return {
        "run_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "forecast_date": today.isoformat(),
        "prediction_date": (today + timedelta(days=1)).isoformat(),
        "threshold": model["threshold"],
        "summary": {
            "districts_tested": len(districts),
            "http_200": sum(row["http_status"] == 200 for row in districts),
            "passed": passed,
            "failed": len(districts) - passed,
        },
        "districts": districts,
    }


if __name__ == "__main__":
    report = run_validation()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["summary"]["failed"] == 0 else 1)
