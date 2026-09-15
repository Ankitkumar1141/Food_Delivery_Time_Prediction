import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

dagshub.init(repo_owner='Ankitkumar1141', 
             repo_name='Food_Delivery_Time_Prediction', 
             mlflow=True)

# Target MLflow tracking service
mlflow.set_tracking_uri("https://dagshub.com/Ankitkumar1141/Food_Delivery_Time_Prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer

# Fetch candidate model name from saved run info
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# Build path to staging candidate in registry
model_path = f"models:/{model_name}/{stage}"

# Load staging model artifact
model = mlflow.sklearn.load_model(model_path)

# Determine project root directory
root_path = Path(__file__).parent.parent

# Load persisted preprocessing transformer
preprocessor_path = root_path / "models" / "preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)


# Combine preprocessing step and regressor into evaluation pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                        argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):
    # Read evaluation dataset
    df = pd.read_csv(test_data_path)
    
    # Remove records with null attributes
    df.dropna(inplace=True)
    
    # Separate predictors and ground truth labels
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']
    
    # Generate inference predictions
    y_pred = model_pipe.predict(X)
    
    # Compute mean absolute error metric
    mean_error = mean_absolute_error(y,y_pred)
    
    # Validate against allowable error tolerance threshold
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The {model_name} model passed the performance test")
    