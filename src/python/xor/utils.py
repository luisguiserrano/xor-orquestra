import json
import torch
import numpy as np
from tensorflow import keras
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    """
    Aux classes for decoding NumPy arrays to Python objects.
    Returns:
        A list or a JSONEnconder object.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def convert_torch_to_numpy(args):
    """
    Converts a list of Torch tensors to NumPy arrays.
    Args:
        args (list): list containing QNode arguments, including Torch tensors.
    Returns:
        list: returns list with Torch tensors converted to NumPy arrays.
    """
    res = []

    for i in args:
        if isinstance(i, torch.Tensor):
            if i.is_cuda:
                res.append(i.cpu().detach().numpy())
            else:
                res.append(i.detach().numpy())
        else:
            res.append(i)

    res = [i.tolist() if (isinstance(i, np.ndarray) and not i.shape) else i for i in res]

    return res


def serialize_torch(key, torch_tensor):
    """
    Serialize an PyTorch binary.
    Args:
        key (str): a value to be the key in the final JSON result.
        torch_tensor: a PyTorch Tensor object.
    Returns:
        A dictionary with the PyTorch serialized.
    """
    np_dict = {key: convert_torch_to_numpy(torch_tensor)}
    return json.dumps(np_dict, cls=NumpyArrayEncoder)


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


def save_json(data, filename) -> None:
    """
    Saves data as JSON.
    Args:
        data (ditc): data to save.
        filenames (str): file name to save the data in
            (should have a '.json' extension).
    """
    try:
        with open(filename,'w') as f:
            data["schema"] = "orquestra-v1-data"
            f.write(json.dumps(data, indent=2)) 

    except IOError:
        print(f'Error: Could not open {filename}')


def save_torch(data, filename) -> None:
    """
    Saves PyTorch data as .PTH.
    Args:
        data (torch tensor): data to save.
        filenames (str): file name to save the data in
            (should have a '.pth' extension).
    """
    try:
        torch.save(data, filename)

    except Exception as e:
        print(f'Error: Could not open {filename}: {e}')


def load_torch(filename):
    """
    Loads data from PTH.
    Args:
        filename (str): the file to load the data from.
    Returns:
        data (PyTorch): data that was loaded from the file.
    """
    try:
        return torch.load(buffer)
    except Exception as e:
        print(f'Error: Could not open {filename}: {e}')