"""
=============================================================================
  LUNG CANCER DETECTION SYSTEM — Flask API
  Endpoint: POST /predict   →  JSON prediction
  Endpoint: GET  /health    →  service health check
  Endpoint: GET  /model-info →  model metadata
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load model artefacts once at startup ─────────────────────────────────────
model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
metadata      = joblib.load(os.path.join(MODEL_DIR, "model_metadata.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

print(f"✅ Model loaded: {metadata['model_name']}  "
      f"(Training accuracy: {metadata['accuracy']:.2%})")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")


def engineer_features(data: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training.
    Input dict uses raw form field names (upper-case).
    """
    df = pd.DataFrame([data])
    df["SMOKING_AGE"]      = df["SMOKING"] * df["AGE"] / 100
    df["RESPIRATORY_RISK"] = ((df["WHEEZING"] + df["SHORTNESS_OF_BREATH"] +
                                df["COUGHING"] + df["CHEST_PAIN"]) - 4) / 4
    df["SYSTEMIC_RISK"]    = ((df["CHRONIC_DISEASE"] + df["FATIGUE"] +
                                df["ANXIETY"]) - 3) / 3
    df["SOCIAL_RISK"]      = ((df["PEER_PRESSURE"] + df["ALCOHOL_CONSUMING"]) - 2) / 2
    return df[feature_names]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the HTML prediction form."""
    return render_template("index.html",
                           model_name=metadata["model_name"],
                           accuracy=f"{metadata['accuracy']:.2%}")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON or form data with patient symptoms and return prediction.

    Expected fields (JSON or form):
      GENDER (1=M, 2=F), AGE (int),
      SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
      FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING,
      SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN
      — all binary features: 1=No, 2=Yes
    """
    try:
        # Accept both JSON body and HTML form POST
        if request.is_json:
            raw = request.get_json()
        else:
            raw = request.form.to_dict()

        # Cast all values to float
        data = {k: float(v) for k, v in raw.items()}

        # Build feature frame with engineering
        X_input = engineer_features(data)
        X_scaled = scaler.transform(X_input)

        # Prediction
        prediction   = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0].tolist()
        confidence   = max(probabilities) * 100

        risk_label = "High Risk of Lung Cancer" if prediction == 1 else "Low Risk of Lung Cancer"
        risk_level = "high" if prediction == 1 else "low"

        # Key contributing factors (top 5 symptoms present)
        symptom_map = {
            "SMOKING": "Smoking", "WHEEZING": "Wheezing",
            "COUGHING": "Coughing", "CHEST_PAIN": "Chest Pain",
            "SHORTNESS_OF_BREATH": "Shortness of Breath",
            "SWALLOWING_DIFFICULTY": "Swallowing Difficulty",
            "CHRONIC_DISEASE": "Chronic Disease", "FATIGUE": "Fatigue",
            "ANXIETY": "Anxiety", "ALCOHOL_CONSUMING": "Alcohol Consuming"
        }
        factors = [label for field, label in symptom_map.items()
                   if data.get(field, 1) == 2][:5]

        response = {
            "prediction": prediction,
            "risk_label": risk_label,
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "probability_no_cancer": round(probabilities[0] * 100, 2),
            "probability_cancer":    round(probabilities[1] * 100, 2),
            "key_factors": factors,
            "model_used": metadata["model_name"],
            "model_accuracy": f"{metadata['accuracy']:.2%}"
        }
        return jsonify(response)

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Service health check."""
    return jsonify({
        "status": "healthy",
        "model": metadata["model_name"],
        "accuracy": f"{metadata['accuracy']:.2%}",
        "features": len(feature_names)
    })


@app.route("/model-info")
def model_info():
    """Return model metadata."""
    return jsonify({
        "model_name": metadata["model_name"],
        "training_accuracy": f"{metadata['accuracy']:.2%}",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "description": (
            "Lung Cancer Risk Prediction Model trained on clinical "
            "symptom data using an advanced ensemble approach."
        )
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "★"*50)
    print("   LUNG CANCER DETECTION API — Starting …")
    print(f"   Model   : {metadata['model_name']}")
    print(f"   Accuracy: {metadata['accuracy']:.2%}")
    print("   URL     : http://127.0.0.1:5000")
    print("★"*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
