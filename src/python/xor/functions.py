import numpy as np
import torch
from torch import nn
from torch import optim
import json
from xor.utils import NumpyArrayEncoder
import logging

# Build the dataset
def build_dataset():

    features = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(np.float32)
    features = torch.from_numpy(features)

    labels = np.array([0,1,1,0]).astype(np.float32).reshape(-1,1)
    labels = torch.from_numpy(labels)

    dataset = (features, labels)
    return dataset

# Build and train the model
def build_and_train_model(features, labels, hidden_layer = '8', epochs = '5000', lr = '0.01'):
    logging.warning('This is a warning message')

    features = torch.from_numpy(np.array(features).astype(np.float32))
    labels = torch.from_numpy(np.array(labels).astype(np.float32))
    
    # Join the features and labels
    data = [d for d in zip(features, labels)]

    #Build the model
    model = nn.Sequential(nn.Linear(2, int(hidden_layer)),
                          nn.ReLU(),
                          nn.Linear(int(hidden_layer), 1))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr = np.float32(lr))

    running_losses = []

    # Train the model
    for e in range(int(epochs)):
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
        if e % (int(epochs)/10) == 0:
            #print(f"Epoch: {e}, Training loss: {running_loss}")
            running_losses.append(running_loss)
    return model

def predict(model, features, labels):
    sigmoid = nn.Sigmoid()
    predictions = []
    for f in features:
        predictions.append(sigmoid(model(f)).detach().numpy())
    predictions = np.array(predictions)
    return json.dumps(predictions, cls=NumpyArrayEncoder)
