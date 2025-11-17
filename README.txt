
Insurance Analysis Project - Generated Artifacts
Files included:
- mock_insurance_data.csv : Mock dataset (10,000 rows)
- clf_pipeline_joblib.pkl : Trained RandomForest classification pipeline (preprocessor + classifier)
- reg_pipeline_joblib.pkl : Trained RandomForest regression pipeline (preprocessor + regressor) (trained on claim==1 rows)
- sample_predictions.csv : Example sample predictions (10 rows)
- example_predict.py : Example script to load models and make predictions
- README.txt : This file

To load a model:
import joblib
clf = joblib.load("clf_pipeline_joblib.pkl")
reg = joblib.load("reg_pipeline_joblib.pkl")
df = pd.read_csv("mock_insurance_data.csv")
