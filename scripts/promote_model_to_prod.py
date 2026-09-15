import mlflow
import dagshub
import json
from mlflow import MlflowClient

dagshub.init(repo_owner='Ankitkumar1141', 
             repo_name='Food_Delivery_Time_Prediction', 
             mlflow=True)

# Initialize remote MLflow tracking endpoint
mlflow.set_tracking_uri("https://dagshub.com/Ankitkumar1141/Food_Delivery_Time_Prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# Extract registered model name from local run configuration
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# Instantiate MLflow client to inspect staging versions
client = MlflowClient()

# Query latest version of the model currently residing in Staging
latest_versions = client.get_latest_versions(name=model_name,stages=[stage])

latest_model_version_staging = latest_versions[0].version

# Target environment for deployment promotion
promotion_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)