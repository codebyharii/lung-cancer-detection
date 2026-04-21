"""
Generate a realistic lung cancer dataset matching Kaggle lung cancer prediction format.
Features are based on clinical risk factors with realistic correlations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)
N = 3000  # Large enough for robust training

def generate_lung_cancer_dataset(n_samples=N):
    """
    Generates a clinically realistic lung cancer dataset.
    
    Feature encoding:
      Gender: M=1, F=2
      All YES/NO features: YES=2, NO=1
      Target: YES=1, NO=0
    """
    
    # --- Base population ---
    age = np.random.randint(20, 85, n_samples)
    gender = np.random.choice([1, 2], n_samples, p=[0.55, 0.45])   # 1=M, 2=F
    smoking = np.random.choice([1, 2], n_samples, p=[0.45, 0.55])  # 2=YES

    # Correlated risk factors
    # Anxiety: higher in smokers and older people
    anxiety_prob = 0.3 + 0.2*(smoking==2) + 0.1*(age>50)
    anxiety = (np.random.rand(n_samples) < anxiety_prob).astype(int) + 1

    # Peer pressure: mostly younger population
    peer_pressure_prob = 0.4 - 0.005*np.clip(age-20, 0, 40)
    peer_pressure = (np.random.rand(n_samples) < peer_pressure_prob).astype(int) + 1

    # Chronic disease: increases with age
    chronic_prob = 0.1 + 0.005*(age-20)
    chronic_disease = (np.random.rand(n_samples) < np.clip(chronic_prob,0,0.9)).astype(int) + 1

    # Fatigue: correlated with chronic disease and age
    fatigue_prob = 0.3 + 0.2*(chronic_disease==2) + 0.1*(age>60)
    fatigue = (np.random.rand(n_samples) < np.clip(fatigue_prob,0,0.95)).astype(int) + 1

    # Allergy
    allergy = np.random.choice([1, 2], n_samples, p=[0.55, 0.45])

    # Wheezing: higher in smokers
    wheeze_prob = 0.2 + 0.35*(smoking==2) + 0.1*(chronic_disease==2)
    wheezing = (np.random.rand(n_samples) < np.clip(wheeze_prob,0,0.95)).astype(int) + 1

    # Alcohol consuming
    alcohol_prob = 0.35 + 0.1*(smoking==2)
    alcohol = (np.random.rand(n_samples) < np.clip(alcohol_prob,0,0.9)).astype(int) + 1

    # Coughing: smokers + chronic disease
    cough_prob = 0.25 + 0.3*(smoking==2) + 0.15*(chronic_disease==2)
    coughing = (np.random.rand(n_samples) < np.clip(cough_prob,0,0.95)).astype(int) + 1

    # Shortness of breath: smoking, wheezing, chronic
    sob_prob = 0.2 + 0.2*(smoking==2) + 0.2*(wheezing==2) + 0.15*(chronic_disease==2)
    shortness_of_breath = (np.random.rand(n_samples) < np.clip(sob_prob,0,0.95)).astype(int) + 1

    # Swallowing difficulty: older, chronic disease
    swallow_prob = 0.1 + 0.005*(age-30) + 0.15*(chronic_disease==2)
    swallowing_difficulty = (np.random.rand(n_samples) < np.clip(swallow_prob,0,0.8)).astype(int) + 1

    # Chest pain: smoking, shortness of breath
    chest_prob = 0.15 + 0.2*(smoking==2) + 0.2*(shortness_of_breath==2) + 0.1*(wheezing==2)
    chest_pain = (np.random.rand(n_samples) < np.clip(chest_prob,0,0.95)).astype(int) + 1

    # --- Lung cancer label: weighted risk score ---
    risk_score = (
        0.30 * (smoking == 2) +
        0.15 * (age > 55) / 1.0 +
        0.10 * (chronic_disease == 2) +
        0.10 * (wheezing == 2) +
        0.08 * (coughing == 2) +
        0.07 * (shortness_of_breath == 2) +
        0.06 * (chest_pain == 2) +
        0.05 * (swallowing_difficulty == 2) +
        0.04 * (fatigue == 2) +
        0.03 * (alcohol == 2) +
        0.02 * (anxiety == 2)
    )
    # Add noise
    risk_score += np.random.normal(0, 0.05, n_samples)
    
    # Use 0.48 threshold → ~55% positive class (realistic imbalance)
    lung_cancer = (risk_score > 0.48).astype(int)

    df = pd.DataFrame({
        "GENDER": gender,
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": np.random.choice([1,2], n_samples, p=[0.6,0.4]),
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC_DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL_CONSUMING": alcohol,
        "COUGHING": coughing,
        "SHORTNESS_OF_BREATH": shortness_of_breath,
        "SWALLOWING_DIFFICULTY": swallowing_difficulty,
        "CHEST_PAIN": chest_pain,
        "LUNG_CANCER": lung_cancer
    })

    return df

if __name__ == "__main__":
    df = generate_lung_cancer_dataset()
    df.to_csv("lung_cancer_data.csv", index=False)
    print(f"Dataset generated: {df.shape}")
    print(df["LUNG_CANCER"].value_counts())
    print(df.head())
