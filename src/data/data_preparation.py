import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = "time_taken"

# Configure modular logger
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

# Console log stream handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Attach handler to module logger
logger.addHandler(handler)

# Setup log formatting pattern
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df



def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(data, 
                                             test_size=test_size, 
                                             random_state=random_state)
    
    return train_data, test_data

def read_params(file_path):
    with open(file_path,"r") as f:
        params_file = yaml.safe_load(f)
    
    return params_file
        
def save_data(data: pd.DataFrame, save_path: Path) -> None:
    data.to_csv(save_path, index=False)
    
    
if __name__ == "__main__":
    # Resolve project directory structure and file locations
    root_path = Path(__file__).parent.parent.parent
    # Source cleaned dataset path
    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"
    # Target directory for partitioned splits
    save_data_dir = root_path / "data" / "interim"
    # Ensure interim directory exists
    save_data_dir.mkdir(exist_ok=True,parents=True)
    # Output file paths for training and testing subsets
    train_filename = "train.csv"
    test_filename = "test.csv"
    save_train_path = save_data_dir / train_filename
    save_test_path = save_data_dir / test_filename
    # Pipeline hyperparameters specification
    params_file_path = root_path / "params.yaml"
    
    # Ingest cleaned tabular dataset
    df = load_data(data_path)
    logger.info("Data Loaded Successfully")
    
    # Parse preparation parameters from configuration
    parameters = read_params(params_file_path)['Data_Preparation']
    test_size = parameters['test_size']
    random_state = parameters['random_state']
    logger.info("parameters read successfully")
    
    # Partition dataset into train and test splits
    train_data, test_data = split_data(df,test_size=test_size,random_state=random_state)
    logger.info("Dataset split into train and test data")
    
    # Persist train and test splits to disk
    data_subsets = [train_data,test_data]
    data_paths = [save_train_path,save_test_path]
    filename_list = [train_filename,test_filename]
    for filename , path, data in zip(filename_list, data_paths, data_subsets):
        save_data(data=data, save_path=path)
        logger.info(f"{filename.replace(".csv","")} data saved to location")
