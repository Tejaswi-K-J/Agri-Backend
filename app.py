from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import statistics
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# =========================
# ENVIRONMENT VARIABLES
# =========================

API_KEY = os.getenv("DATA_GOV_API_KEY")

if not API_KEY:
    raise ValueError("DATA_GOV_API_KEY environment variable not set")

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# =========================
# MODEL AUTO DOWNLOAD
# =========================

MODEL_PATH = "karnataka_yield_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1f6ZblcYZTUogSh1nwDP4M31ynp7Towc6"

if not os.path.exists(MODEL_PATH):
    print("Downloading model file from Google Drive...")
    try:
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model file: {e}")

model = joblib.load(MODEL_PATH)

# =========================
# LOAD DATA
# =========================

crop_master = pd.read_csv("karnataka_crop_master_dataset_expanded.csv")

district_rainfall = {
    "Mandya": 806, "Mysuru": 798, "Hassan": 1031,
    "Chikkamagaluru": 1925, "Shivamogga": 1813,
    "Kodagu": 2718, "Udupi": 4119,
    "Dakshina Kannada": 3975, "Uttara Kannada": 2835,
    "Belagavi": 808, "Dharwad": 772,
    "Haveri": 753, "Tumakuru": 688,
    "Raichur": 621, "Bagalkote": 562,
    "Vijayapura": 578, "Kolar": 744,
    "Bengaluru Rural": 885, "Ballari": 636,
    "Gulbarga": 777
}

# =========================
# MANDI PRICE FUNCTIONS
# =========================

def fetch_karnataka_prices():
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[state]": "Karnataka",
        "limit": 300
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data.get("records", [])
    except Exception as e:
        print("Mandi API fetch error:", e)
        return []


def build_price_dictionary(records):
    price_dict = {}

    for record in records:
        commodity = record.get("commodity")
        price = record.get("modal_price")

        if not commodity or not price:
            continue

        try:
            price = float(price)
        except:
            continue

        # Outlier filtering
        if price < 100:
            continue
        if price > 30000:
            continue

        commodity_clean = commodity.lower().strip()

        if commodity_clean not in price_dict:
            price_dict[commodity_clean] = []

        price_dict[commodity_clean].append(price)

    # Median price
    for commodity in price_dict:
        price_dict[commodity] = statistics.median(price_dict[commodity])

    return price_dict


def get_price_for_crop(crop_name, price_dict):
    crop_name = crop_name.lower()

    for commodity in price_dict:

        # Avoid mixing dry & green chilli
        if "chilli" in crop_name:
            if "dry" in commodity and "green" in crop_name:
                continue

        if crop_name in commodity:
            return price_dict[commodity]

    return 0


# =========================
# HEALTH CHECK ROUTE
# =========================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Agri Finance Backend Running"})


# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    district = data["district"]
    season = data["season"]
    soil = data["soil"]
    rainfall = float(data["rainfall"])
    irrigation = data["irrigation"]
    land_area = float(data["land_area"])
    available_profit = float(data["profit"])

    district_mean = district_rainfall.get(district, 800)

    mandi_records = fetch_karnataka_prices()
    price_dict = build_price_dictionary(mandi_records)

    recommendations = []

    for _, crop in crop_master.iterrows():

        # Rule filtering
        if season.lower() == "kharif" and crop["season_kharif"] != 1:
            continue
        if season.lower() == "rabi" and crop["season_rabi"] != 1:
            continue

        if soil.lower() == "black" and crop["soil_black"] != 1:
            continue
        if soil.lower() == "red" and crop["soil_red"] != 1:
            continue
        if soil.lower() == "alluvial" and crop["soil_alluvial"] != 1:
            continue

        input_data = pd.DataFrame([{
            "district_mean_rainfall": district_mean,
            "rainfall_mm": rainfall,
            "crop_id": crop["crop_id"],
            "soil_black": 1 if soil.lower()=="black" else 0,
            "soil_red": 1 if soil.lower()=="red" else 0,
            "soil_alluvial": 1 if soil.lower()=="alluvial" else 0,
            "irrigation_rainfed": 1 if irrigation.lower()=="rainfed" else 0,
            "irrigation_borewell": 1 if irrigation.lower()=="borewell" else 0,
            "irrigation_canal": 1 if irrigation.lower()=="canal" else 0,
            "season_kharif": 1 if season.lower()=="kharif" else 0,
            "season_rabi": 1 if season.lower()=="rabi" else 0
        }])

        predicted_yield = model.predict(input_data)[0]

        price = get_price_for_crop(crop["crop_name"], price_dict)

        if price == 0:
            continue

        investment = crop["total_cost_per_acre"] * land_area
        revenue = predicted_yield * land_area * price
        expected_profit = revenue - investment

        roi = (expected_profit / investment) * 100 if investment > 0 else 0

        if roi > 200:
            risk = "High Reward (High Market Volatility)"
        elif roi > 80:
            risk = "Moderate"
        else:
            risk = "Stable"

        recommendations.append({
            "crop": crop["crop_name"],
            "predicted_yield": round(predicted_yield, 2),
            "market_price_per_quintal": round(price, 2),
            "investment": round(investment, 2),
            "expected_profit": round(expected_profit, 2),
            "roi_percent": round(roi, 2),
            "risk_level": risk
        })

    recommendations.sort(key=lambda x: x["expected_profit"], reverse=True)

    return jsonify({
        "recommendations": recommendations[:3]
    })


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    app.run()
