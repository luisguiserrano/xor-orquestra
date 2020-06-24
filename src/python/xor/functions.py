import numpy as np
import torch
from torch import nn
from torch import optim
import json
from json import JSONEncoder

# Build the dataset
def build_dataset():
    features = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float32)
    features = torch.from_numpy(features)

    labels = np.array([0,1,1,0]).astype(np.float32).reshape(-1,1)
    labels = torch.from_numpy(labels)

    dataset = (features, labels)
    return features, labels

# Build and train the model
def build_and_train_model(hidden_layer = 8, epochs = 5000, lr = 0.01):
    
    # Join the features and labels
    features, labels = build_dataset()
    data = [d for d in zip(features, labels)]

    #Build the model
    model = nn.Sequential(nn.Linear(2, hidden_layer),
                          nn.ReLU(),
                          nn.Linear(hidden_layer, 1))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)

    running_losses = []

    # Train the model
    for e in range(epochs):
        running_loss = 0
        for data_point in data:
            feature = data_point[0]
            label = data_point[1]
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if e % (epochs/10) == 0:
            #print(f"Epoch: {e}, Training loss: {running_loss}")
            running_losses.append(running_loss)
    return model

def predict(model):
    features, _ = build_dataset()
    sigmoid = nn.Sigmoid()
    predictions = []
    for f in features:
        predictions.append(sigmoid(model(f)).detach().numpy())
    return predictions

    
def read_json(filename) -> dict:
    """
    Loads data from JSON.
    Args:
        filename (str): the file to load the data from.
    Returns:
        data (dict): data that was loaded from the file.
    """
    data = {}
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except IOError:
            print(f'Error: Could not open {filename}')

    return data


def save_json(result, filename) -> None:
    """
    Saves data as JSON.
    Args:
        result (ditc): of data to save.
        filenames (str): file name to save the data in
            (should have a '.json' extension).
    """
    try:
        with open(filename,'w') as f:
            result["schema"] = "orquestra-v1-data"
            f.write(json.dumps(result, indent=2)) 

    except IOError:
        print(f'Error: Could not open {filename}')
