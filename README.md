# 🫁 Lung Cancer Detection System

AI-powered lung cancer risk prediction using clinical symptom data.  
**Best model accuracy: 95.13% | ROC-AUC: 0.991**

---

## 📁 Project Structure

```
lung_cancer_detection/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── lung_cancer_data.csv    # Generated dataset (3000 samples)
├── models/
│   ├── best_model.pkl          # Saved best ML model (AdaBoost)
│   ├── scaler.pkl              # StandardScaler for feature normalization
│   ├── feature_names.pkl       # Feature list for inference
│   └── model_metadata.pkl      # Model name, accuracy, feature info
├── plots/
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_age_distribution.png
│   ├── 05_model_comparison.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   ├── 08_feature_importance.png
│   └── model_results.csv
├── templates/
│   └── index.html              # HTML prediction frontend
├── train.py                    # Model training pipeline
├── app.py                      # Flask API server
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
cd data && python generate_dataset.py && cd ..
```

### 3. Train models
```bash
python train.py
```
This will:
- Run EDA and generate visualizations in `plots/`
- Train 5 base models + 2 ensemble models
- Perform hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Evaluate all models and print a comparison table
- Save the best model to `models/best_model.pkl`

### 4. Start the API server
```bash
python app.py
```
Open your browser at **http://127.0.0.1:5000**

---

## 📊 Model Results

| Model              | Accuracy | Precision | Recall | F1     | ROC-AUC |
|--------------------|----------|-----------|--------|--------|---------|
| AdaBoost ✅         | **95.13%** | 0.931   | 0.967  | 0.948  | 0.991   |
| Stacking Ensemble ✅| **95.13%** | 0.930   | 0.966  | 0.948  | 0.991   |
| Gradient Boosting  | 94.79%   | 0.930     | 0.956  | 0.943  | 0.990   |
| Logistic Regression| 94.45%   | 0.926     | 0.952  | 0.939  | 0.989   |
| Random Forest      | 93.95%   | 0.921     | 0.949  | 0.935  | 0.987   |
| SVM                | 93.28%   | 0.915     | 0.941  | 0.928  | 0.985   |

---

## 🌐 API Reference

### `POST /predict`
Predict lung cancer risk from patient symptoms.

**Request body (JSON):**
```json
{
  "GENDER": 1,
  "AGE": 65,
  "SMOKING": 2,
  "YELLOW_FINGERS": 2,
  "ANXIETY": 1,
  "PEER_PRESSURE": 1,
  "CHRONIC_DISEASE": 2,
  "FATIGUE": 2,
  "ALLERGY": 1,
  "WHEEZING": 2,
  "ALCOHOL_CONSUMING": 1,
  "COUGHING": 2,
  "SHORTNESS_OF_BREATH": 2,
  "SWALLOWING_DIFFICULTY": 1,
  "CHEST_PAIN": 2
}
```
> **Encoding:** Gender: 1=Male, 2=Female | All symptoms: 1=No, 2=Yes

**Response:**
```json
{
  "prediction": 1,
  "risk_label": "High Risk of Lung Cancer",
  "risk_level": "high",
  "confidence": 94.7,
  "probability_cancer": 94.7,
  "probability_no_cancer": 5.3,
  "key_factors": ["Smoking", "Chronic Disease", "Wheezing", "Coughing", "Chest Pain"],
  "model_used": "AdaBoost",
  "model_accuracy": "95.13%"
}
```

### `GET /health`
```json
{ "status": "healthy", "model": "AdaBoost", "accuracy": "95.13%", "features": 19 }
```

### `GET /model-info`
Returns full model metadata and feature list.

---

## 🧪 Sample cURL Test

```bash
# High-risk patient
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GENDER":1,"AGE":65,"SMOKING":2,"YELLOW_FINGERS":2,
    "ANXIETY":2,"PEER_PRESSURE":1,"CHRONIC_DISEASE":2,
    "FATIGUE":2,"ALLERGY":1,"WHEEZING":2,
    "ALCOHOL_CONSUMING":2,"COUGHING":2,
    "SHORTNESS_OF_BREATH":2,"SWALLOWING_DIFFICULTY":2,"CHEST_PAIN":2
  }'

# Low-risk patient
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GENDER":2,"AGE":28,"SMOKING":1,"YELLOW_FINGERS":1,
    "ANXIETY":1,"PEER_PRESSURE":1,"CHRONIC_DISEASE":1,
    "FATIGUE":1,"ALLERGY":1,"WHEEZING":1,
    "ALCOHOL_CONSUMING":1,"COUGHING":1,
    "SHORTNESS_OF_BREATH":1,"SWALLOWING_DIFFICULTY":1,"CHEST_PAIN":1
  }'
```

---

## 🔬 Why AdaBoost Performs Best

AdaBoost (Adaptive Boosting) achieved the highest accuracy because:

1. **Sequential error correction** — Each tree focuses on samples misclassified by previous trees, progressively reducing bias on hard cases.
2. **Low variance** — The boosting strategy naturally regularizes the ensemble, preventing overfitting even on tabular clinical data.
3. **Optimal for structured/tabular data** — Binary symptom features (1/2 encoding) are well-suited to shallow decision stumps.
4. **ROC-AUC of 0.991** — Near-perfect discrimination between cancer and non-cancer patients.

---

## ⚠️ Disclaimer

This system is for **research and screening purposes only**.  
It does **not** replace professional medical diagnosis.  
Always consult a qualified physician for medical decisions.
"# lung-cancer-detection" 
