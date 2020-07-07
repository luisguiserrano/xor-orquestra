import numpy as np
import json
import torch

# Build the dataset
def build_dataset():
    features = [[0,0],[0,1],[1,0],[1,1]]
    labels = [0,1,1,0]
    
    result = {}
    result["features"] = features
    result["labels"] = labels

    return result
