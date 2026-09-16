import pandas as pd
import requests
from pathlib import Path

# Root and raw dataset path configurations
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "swiggy.csv"

# Remote or local prediction service URL
predict_url = "http://13.201.83.141//predict"

# Extract single non-null record for integration testing
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# Format features dictionary by excluding label column
data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
print(data)

# Dispatch HTTP POST request to prediction endpoint
response = requests.post(url=predict_url,json=data)

print("The status code for response is", response.status_code)

if response.status_code == 200:
    print(f"The prediction value by the API is {float(response.text):.2f} min")
else:
    print("Error:", response.status_code)