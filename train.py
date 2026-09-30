"""Train and compare churn models. Run: python train.py"""
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
df = df.drop(columns="customerID")
y = (df.pop("Churn") == "Yes").astype(int)
X = df
num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

results, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(Xtr, ytr)
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
    results.append(dict(model=name, accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                        recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, proba),
                        confusion=confusion_matrix(yte, pred).tolist()))
    fitted[name] = pipe
    print(name, {k: round(v, 3) for k, v in results[-1].items() if isinstance(v, float)})

best = max(results, key=lambda r: r["roc_auc"])["model"]
imp = permutation_importance(fitted[best], Xte, yte, n_repeats=3, random_state=42, scoring="roc_auc")
importance = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])[:10]
joblib.dump(fitted[best], "model.joblib", compress=3)
json.dump({"best": best, "results": results, "importance": importance}, open("metrics.json", "w"), indent=1)
print("Best model:", best)
