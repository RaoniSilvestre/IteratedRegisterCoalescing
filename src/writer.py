import json
from typing import Dict, List

LivenessData = Dict[str, List[List[int]]]

def save_liveness_to_json(data: LivenessData, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4) 
    print(f"Data loaded into: {filename}")

def load_liveness_from_json(filename: str) -> LivenessData:
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from: {filename}")
    return data
