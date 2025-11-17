
# example_predict.py
import joblib
import pandas as pd

clf = joblib.load("clf_pipeline_joblib.pkl")
reg = joblib.load("reg_pipeline_joblib.pkl")

# load sample
df = pd.read_csv("mock_insurance_data.csv")
sample = df.drop(columns=["policy_id", "claim", "claim_amount", "claim_age"]).iloc[:5]
probs = clf.predict_proba(sample)[:, 1]
amounts = reg.predict(sample)
print("probabilities:", probs)
print("amounts:", amounts)
