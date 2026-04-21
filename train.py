"""
=============================================================================
  LUNG CANCER DETECTION SYSTEM - Model Training Pipeline
=============================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "lung_cancer_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE = {"primary":"#E84393","secondary":"#7C3AED","success":"#10B981",
           "warning":"#F59E0B","danger":"#EF4444"}

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH).drop_duplicates()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].map({"YES":2,"NO":1,"M":1,"F":2}).fillna(df[col])
df = df.apply(pd.to_numeric, errors="coerce").dropna()

X = df.drop("LUNG_CANCER", axis=1).copy()
y = df["LUNG_CANCER"].astype(int)

# Feature engineering
X["SMOKING_AGE"]      = X["SMOKING"] * X["AGE"] / 100
X["RESPIRATORY_RISK"] = ((X["WHEEZING"]+X["SHORTNESS_OF_BREATH"]+X["COUGHING"]+X["CHEST_PAIN"])-4)/4
X["SYSTEMIC_RISK"]    = ((X["CHRONIC_DISEASE"]+X["FATIGUE"]+X["ANXIETY"])-3)/3
X["SOCIAL_RISK"]      = ((X["PEER_PRESSURE"]+X["ALCOHOL_CONSUMING"])-2)/2

feature_names = list(X.columns)
print(f"Features: {len(feature_names)}  |  Samples: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

# ── 2. EDA plots ──────────────────────────────────────────────────────────────
print("EDA plots …")
fig, ax = plt.subplots(figsize=(6,4))
counts = y.value_counts()
ax.bar(["No Cancer","Lung Cancer"],counts.values,
       color=[PALETTE["success"],PALETTE["danger"]],width=0.5)
for i,v in enumerate(counts.values): ax.text(i,v+10,str(v),ha="center",fontweight="bold")
ax.set_title("Class Distribution",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/01_class_distribution.png",dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(14,10))
corr = df.corr(); mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="RdYlGn",center=0,
            ax=ax,linewidths=0.5,annot_kws={"size":7})
ax.set_title("Correlation Heatmap",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png",dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(8,4))
df[df.LUNG_CANCER==0]["AGE"].plot.hist(ax=ax,alpha=0.6,color=PALETTE["success"],bins=20,label="No Cancer")
df[df.LUNG_CANCER==1]["AGE"].plot.hist(ax=ax,alpha=0.6,color=PALETTE["danger"],bins=20,label="Lung Cancer")
ax.set_title("Age Distribution by Diagnosis",fontsize=12,fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/03_age_distribution.png",dpi=120); plt.close()
print("EDA plots saved ✅")

# ── 3. Train models ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {}

print("\n[1/5] Logistic Regression …")
lr = GridSearchCV(LogisticRegression(random_state=42,max_iter=1000),
                  {"C":[0.1,1,10],"solver":["lbfgs"]},cv=cv,scoring="accuracy",n_jobs=-1)
lr.fit(Xtr,y_train); models["Logistic Regression"]=lr.best_estimator_
print(f"  CV={lr.best_score_:.4f}")

print("[2/5] SVM …")
svm = GridSearchCV(SVC(probability=True,random_state=42),
                   {"C":[1,10],"kernel":["rbf"],"gamma":["scale"]},
                   cv=cv,scoring="accuracy",n_jobs=-1)
svm.fit(Xtr,y_train); models["SVM"]=svm.best_estimator_
print(f"  CV={svm.best_score_:.4f}")

print("[3/5] Random Forest …")
rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42,class_weight="balanced"),
    {"n_estimators":[200,400],"max_depth":[None,20],"min_samples_split":[2,5],"max_features":["sqrt"]},
    n_iter=8,cv=cv,scoring="accuracy",n_jobs=-1,random_state=42)
rf.fit(Xtr,y_train); models["Random Forest"]=rf.best_estimator_
print(f"  CV={rf.best_score_:.4f}")

print("[4/5] Gradient Boosting …")
gb = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators":[200,300],"learning_rate":[0.05,0.1],"max_depth":[4,6],"subsample":[0.8,1.0]},
    n_iter=8,cv=cv,scoring="accuracy",n_jobs=-1,random_state=42)
gb.fit(Xtr,y_train); models["Gradient Boosting"]=gb.best_estimator_
print(f"  CV={gb.best_score_:.4f}")

print("[5/5] AdaBoost …")
ada = GridSearchCV(AdaBoostClassifier(random_state=42),
                   {"n_estimators":[200,400],"learning_rate":[0.5,1.0]},
                   cv=cv,scoring="accuracy",n_jobs=-1)
ada.fit(Xtr,y_train); models["AdaBoost"]=ada.best_estimator_
print(f"  CV={ada.best_score_:.4f}")

# ── 4. Stacking ───────────────────────────────────────────────────────────────
print("\nStacking Ensemble …")
estimators = [(n.lower().replace(" ","_"),m) for n,m in models.items()]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=5,max_iter=1000,random_state=42),
    cv=5,n_jobs=-1)
stacking.fit(Xtr,y_train); models["Stacking Ensemble"]=stacking

voting = VotingClassifier(estimators=estimators,voting="soft",n_jobs=-1)
voting.fit(Xtr,y_train); models["Soft Voting"]=voting
print("Ensembles trained ✅")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n"+"="*60+"\n  EVALUATION\n"+"="*60)
rows=[]
for name,model in models.items():
    yp   = model.predict(Xte)
    yprob= model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
    acc  = accuracy_score(y_test,yp)
    prec = precision_score(y_test,yp,zero_division=0)
    rec  = recall_score(y_test,yp,zero_division=0)
    f1   = f1_score(y_test,yp,zero_division=0)
    auc  = roc_auc_score(y_test,yprob) if yprob is not None else np.nan
    rows.append({"Model":name,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,"ROC-AUC":auc})
    flag = "✅" if acc>=0.95 else "  "
    print(f"  {flag} {name:<22} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

results_df = pd.DataFrame(rows).sort_values("Accuracy",ascending=False)
results_df.to_csv(f"{PLOT_DIR}/model_results.csv",index=False)

# ── 6. Performance plots ──────────────────────────────────────────────────────
print("\nGenerating plots …")
metrics=["Accuracy","Precision","Recall","F1","ROC-AUC"]
x=np.arange(len(results_df)); w=0.15
colours_bar=[PALETTE["primary"],PALETTE["secondary"],PALETTE["success"],PALETTE["warning"],PALETTE["danger"]]
fig,ax=plt.subplots(figsize=(16,6))
for i,(m,c) in enumerate(zip(metrics,colours_bar)):
    ax.bar(x+i*w,results_df[m],w,label=m,color=c,alpha=0.85)
ax.axhline(0.95,color="black",linestyle="--",linewidth=1.5,label="95% target")
ax.set_xticks(x+w*2); ax.set_xticklabels(results_df["Model"],rotation=20,ha="right")
ax.set_ylim(0.5,1.05); ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics",fontsize=13,fontweight="bold")
ax.legend(bbox_to_anchor=(1,1))
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/05_model_comparison.png",dpi=130); plt.close()

n_m=len(models); cols_n=4; rows_n=(n_m+cols_n-1)//cols_n
fig,axes=plt.subplots(rows_n,cols_n,figsize=(cols_n*5,rows_n*4))
axes=axes.flatten()
for ax,(name,model) in zip(axes,models.items()):
    cm=confusion_matrix(y_test,model.predict(Xte))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,
                xticklabels=["No","Cancer"],yticklabels=["No","Cancer"],annot_kws={"size":12})
    acc=accuracy_score(y_test,model.predict(Xte))
    ax.set_title(f"{name}\n{acc:.2%}",fontsize=9,fontweight="bold")
for ax in axes[n_m:]: ax.set_visible(False)
plt.suptitle("Confusion Matrices",fontsize=13,fontweight="bold",y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_confusion_matrices.png",dpi=130,bbox_inches="tight"); plt.close()

fig,ax=plt.subplots(figsize=(9,7))
colours_roc=plt.cm.tab10(np.linspace(0,1,len(models)))
for (name,model),col in zip(models.items(),colours_roc):
    if hasattr(model,"predict_proba"):
        fpr,tpr,_=roc_curve(y_test,model.predict_proba(Xte)[:,1])
        auc=roc_auc_score(y_test,model.predict_proba(Xte)[:,1])
        ax.plot(fpr,tpr,lw=2,color=col,label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.2,label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves",fontsize=13,fontweight="bold"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/07_roc_curves.png",dpi=130); plt.close()

best_name=results_df.iloc[0]["Model"]; best_model=models[best_name]
if hasattr(best_model,"feature_importances_"):
    imp=best_model.feature_importances_; idx=np.argsort(imp)[::-1]
    colours_fi=[PALETTE["danger"] if i<5 else PALETTE["secondary"] for i in range(len(idx))]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh([feature_names[i] for i in idx[::-1]],imp[idx[::-1]],color=colours_fi[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance — {best_name}",fontsize=12,fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/08_feature_importance.png",dpi=130); plt.close()

print("Plots saved ✅")

# ── 7. Best model summary ──────────────────────────────────────────────────────
best_acc=results_df.iloc[0]["Accuracy"]
print("\n"+"="*60)
print(f"  🏆 BEST MODEL: {best_name}")
print(f"  📊 Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"  🎯 Target: {'✅ ACHIEVED' if best_acc>=0.95 else '⚠️ Near-target'}")
print("="*60)
yp=best_model.predict(Xte)
print(classification_report(y_test,yp,target_names=["No Cancer","Lung Cancer"]))

print("10-fold CV …")
full_X_sc=scaler.transform(X)
for met in ["accuracy","precision","recall","f1","roc_auc"]:
    sc=cross_val_score(best_model,full_X_sc,y,
                       cv=StratifiedKFold(10,shuffle=True,random_state=42),
                       scoring=met,n_jobs=-1)
    print(f"  {met:12s}: {sc.mean():.4f} ± {sc.std():.4f}")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump({"model_name":best_name,"accuracy":best_acc,"feature_names":feature_names},
            f"{MODEL_DIR}/model_metadata.pkl")
print(f"\nModel saved ✅  →  {MODEL_DIR}/best_model.pkl")
print("\n★ TRAINING COMPLETE ★")
