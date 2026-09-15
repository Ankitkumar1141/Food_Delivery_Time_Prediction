import pandas as pd
import requests
from pathlib import Path
import pytest

# Define path to raw test input data
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# Extract an arbitrary clean sample observation for API payload verification
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# Separate target label to construct request feature payload
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()

@pytest.mark.parametrize(argnames="url, data",
                         argvalues=[("http://127.0.0.1:8000/predict", data)])
def test_predict_endpoint(url,data):
    # Dispatch inference POST request to API server
    response = requests.post(url=url,json=data)
    # Validate successful HTTP 200 OK status
    assert response.status_code == 200, "Prediction endpoint not giving response"
