# Customer Churn Prediction

Predicts which telecom customers are likely to leave, using the public IBM Telco dataset (7,043 customers).

**Tech:** Python, Pandas, Scikit-learn, Plotly, Streamlit

## What I did
Cleaned data (fixed blank `TotalCharges`, imputed and scaled numbers, one-hot encoded categories); trained Logistic Regression, Random Forest and Gradient Boosting; evaluated on a stratified 20% test set.

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 |
| Random Forest | 0.799 | 0.662 | 0.497 | 0.568 | 0.841 |
| **Gradient Boosting** | 0.806 | 0.674 | 0.524 | 0.589 | **0.843** |

## Run locally
```bash
pip install -r requirements.txt
python train.py   # trains models and saves model.joblib + metrics.json
streamlit run app.py
```

## Live demo
[Add your Streamlit link here]
