import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.ensemble import StackingRegressor

TARGET = "time_taken"

# Initialize logging facility
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Setup stream log handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Register handler
logger.addHandler(handler)

# Set logging output format
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df

def read_params(file_path):
    with open(file_path,"r") as f:
        params_file = yaml.safe_load(f)
    
    return params_file


def save_model(model, save_dir: Path, model_name: str):
    # Formulate destination model path
    save_location = save_dir / model_name
    # Serialize model artifact
    joblib.dump(value=model,filename=save_location)
    
    
def save_transformer(transformer, save_dir: Path, transformer_name: str):
    # Formulate destination transformer path
    save_location = save_dir / transformer_name
    # Serialize transformer artifact
    joblib.dump(transformer, save_location)
    
    
def train_model(model, X_train: pd.DataFrame, y_train):
    # Fit estimator on training observations
    model.fit(X_train,y_train)
    return model


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y



if __name__ == "__main__":
    # Base workspace directory
    root_path = Path(__file__).parent.parent.parent
    # Path to preprocessed training dataset
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    # Hyperparameters configuration file
    params_file_path = root_path / "params.yaml"
    
    # Load processed feature matrix
    training_data = load_data(data_path)
    logger.info("Training Data read successfully")
    
    # Separate predictors and target labels
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")
    
    # Ingest modeling hyperparameters
    model_params = read_params(params_file_path)['Train']
    
    # Random Forest estimator parameters
    rf_params = model_params['Random_Forest']
    logger.info("random forest parameters read")
    
    # Instantiate Random Forest regressor
    rf = RandomForestRegressor(**rf_params)
    logger.info("built random forest model")
    
    # LightGBM estimator parameters
    lgbm_params = model_params["LightGBM"]
    logger.info("Light GBM parameters read")
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("built Light GBM model")
    
    # Instantiate linear regression meta-learner
    lr = LinearRegression()
    logger.info("Meta model built")
    
    # Power transformer for target normalisation
    power_transform = PowerTransformer()
    logger.info("Target Transformer built")
    
    # Construct ensemble StackingRegressor
    stacking_reg = StackingRegressor(estimators=[("rf_model",rf),
                                                 ("lgbm_model",lgbm)],
                                     final_estimator=lr,
                                     cv=5,n_jobs=-1)
    logger.info("Stacking regressor built")
    
    # Encapsulate stacking regressor with target transformation
    model = TransformedTargetRegressor(regressor=stacking_reg,
                                       transformer=power_transform)
    logger.info("Models wrapped inside wrapper")
    
    # Execute model training workflow
    train_model(model,X_train,y_train)
    logger.info("Model training completed")
    
    # Output artifact file naming
    model_filename = "model.joblib"
    # Destination directory for model artifacts
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)
    
    # Extract underlying ensemble and transformer components
    stacking_model = model.regressor_
    transformer = model.transformer_

    # Serialize complete wrapped model pipeline
    save_model(model=model,
            save_dir=model_save_dir,
            model_name=model_filename)
    logger.info("Trained model saved to location")
    
    # Serialize individual stacking regressor
    stacking_filename = "stacking_regressor.joblib"
    save_model(model=stacking_model,
            save_dir=model_save_dir,
            model_name=stacking_filename)
    logger.info("Trained model saved to location")
    
    # Serialize target power transformer
    transformer_filename = "power_transformer.joblib"
    transformer_save_dir = model_save_dir
    save_transformer(transformer, transformer_save_dir, transformer_filename)
    logger.info("Transformer saved to location")
