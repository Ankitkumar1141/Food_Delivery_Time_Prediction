import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json


# Configure DagsHub remote MLflow tracking
import dagshub
dagshub.init(repo_owner='Ankitkumar1141', 
             repo_name='Food_Delivery_Time_Prediction', 
             mlflow=True)

# Set MLflow tracking server location
mlflow.set_tracking_uri("https://dagshub.com/Ankitkumar1141/Food_Delivery_Time_Prediction.mlflow")

# Designate MLflow experiment name
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# Setup evaluation logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# Stream log handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Register stream handler
logger.addHandler(handler)

# Configure logging output format
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)


if __name__ == "__main__":
    # Base workspace directory
    root_path = Path(__file__).parent.parent.parent
    # Processed train and test dataset locations
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    # Serialized model artifact path
    model_path = root_path / "models" / "model.joblib"
    
    
    # Load transformed training split
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    # Load transformed testing split
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")
    
    # Partition features and target response vectors
    X_train, y_train = make_X_and_y(train_data,TARGET)
    X_test, y_test = make_X_and_y(test_data,TARGET)
    logger.info("Data split completed")
    
    # Ingest serialized model pipeline
    model = load_model(model_path)
    logger.info("Model Loaded successfully")
    
    
    # Generate predictions on training and test subsets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("prediction on data complete")
    
    # Compute Mean Absolute Error (MAE) for train and test splits
    train_mae = mean_absolute_error(y_train,y_train_pred)
    test_mae = mean_absolute_error(y_test,y_test_pred)
    logger.info("error calculated")
    
    # Calculate Coefficient of Determination (R2 Score)
    train_r2 = r2_score(y_train,y_train_pred)
    test_r2 = r2_score(y_test,y_test_pred)
    logger.info("r2 score calculated")
    
    # Perform 5-fold cross validation on training data
    cv_scores = cross_val_score(model,
                                X_train,
                                y_train,
                                cv=5,
                                scoring="neg_mean_absolute_error",
                                n_jobs=-1)
    logger.info("cross validation complete")
    
    # Aggregate mean cross-validation error score
    mean_cv_score = -(cv_scores.mean())
    
    # Record parameters, metrics, and artifacts within MLflow tracking run
    with mlflow.start_run() as run:
        # Assign contextual experiment metadata tags
        mlflow.set_tag("model","Food Delivery Time Regressor")

        # Track model hyperparameter configurations
        mlflow.log_params(model.get_params())

        # Log primary evaluation metrics
        mlflow.log_metric("train_mae",train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2",train_r2)
        mlflow.log_metric("test_r2",test_r2)
        mlflow.log_metric("mean_cv_score",-(cv_scores.mean()))

        # Record individual fold cross-validation performance scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})
        
        # Format dataset objects for MLflow data lineage tracking
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)
        
        # Track training and validation dataset lineage
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")
        
        # Infer schema signature from sample inputs and predictions
        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20,random_state=42),
                                    model_output=model.predict(X_train.sample(20,random_state=42)))
        
        # Log trained model artifact alongside schema signature
        mlflow.sklearn.log_model(model,"delivery_time_pred_model",signature=model_signature)

        # Upload auxiliary stacking regressor artifact
        mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
        
        # Upload target transformation transformer artifact
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        
        # Upload fitted preprocessing pipeline artifact
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")
        
        # Retrieve remote artifact storage path for this active run
        artifact_uri = mlflow.get_artifact_uri()
        
        logger.info("Mlflow logging complete and model logged")
        
    # Extract unique execution run identifier 
    run_id = run.info.run_id
    model_name = "delivery_time_pred_model"
    
    # Persist run details to local metadata file for downstream stages
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")
    
    
    
    
    