
from typing import Iterable
import numpy as np



'''
Class to wrap Real, Integer, Categorical hyperparameter
'''
class Real:
    def __init__(self, low, high) -> None:
        self.__low = low
        self.__high = high
    
    def sample(self, shape):
        s = np.random.uniform(self.__low, self.__high, size=shape)
        return s
    
    def interval(self):
        return np.array([self.__low, self.__high])
    
class Int:
    def __init__(self, low, high) -> None:
        self.__low = low
        self.__high = high

    def sample(self, shape):
        s = np.random.randint(self.__low, self.__high + 1, size=shape)
        return s
    
    def interval(self):
        return np.array([self.__low, self.__high])

class Cat:
    def __init__(self, category:Iterable) -> None:
        self.category = category

    def sample(self, shape):
        s = np.random.randint(0, len(self.category), size=shape)
        return s
    
    def interval(self):
        return np.array([0, len(self.category) - 1])

class Choice:
    def __init__(self, choice:Iterable) -> None:
        self.choice = choice

    def sample(self, shape):
        s = np.random.choice(self.choice, size=shape, replace=True)
        return s
    
    def interval(self):
        return np.array([0, len(self.choice) - 1])

class RealKernels:
    '''
    Class to wrap hyperparameter for Gaussian Process Regressor kernel
    '''
    def __init__(self, bounds:np.array) -> None:
        self.__bounds = bounds

    def sample(self, shape):
        s = np.random.uniform(self.__bounds[:, 0], self.__bounds[:, 1], size=shape)
        return s

    def interval(self):
        return self.__bounds

class AcqParams:
    '''
    Class to wrap hyperparameter so that 
    it can be passed into acquisition function
    '''
    def __init__(self, params) -> None:
        self.__params = params

    def sample(self, shape):
        s = np.zeros(shape)
        for idx, p in enumerate(self.__params):
            s[:, idx] = self.__params[p].sample(shape[0])
        return s

    def interval(self):
        bounds = np.zeros((len(self.__params), 2))
        for idx, p in enumerate(self.__params):
            bounds[idx] = self.__params[p].interval()
        return bounds
    
    def Int_idxs(self):
        mask_int = []
        for idx, p in enumerate(self.__params):
            if type(self.__params[p]) == Int: 
                mask_int.append(idx)
        mask_int = np.array(mask_int)
        return mask_int
    
    def Cat_idxs(self):
        mask_cat = []
        for idx, p in enumerate(self.__params):
            if type(self.__params[p]) == Cat: 
                mask_cat.append(idx)
        mask_cat = np.array(mask_cat)
        return mask_cat




def convert_to_bound(params:dict) -> np.array:
    '''
    Convert hyperparameter dictionary into numpy array bounds

    Parameters
    ----------
    params: The dictionary stores the key how to encode the array

    Returns
    -------
    bounds: A numpy array that indicates the range of each parameters
    '''
    if "theta" in params:
        bounds = params["theta"].interval()
    elif "x" in params:
        bounds = params["x"].interval()
    else:
        bounds = np.zeros(shape=(len(params), 2))
        for idx, p in enumerate(params):
            bounds[idx] = params[p].interval()
    return bounds

def decode_acq_params(encoded, params):
    '''
    Decode the encoded acquisition data back to 
    hyperparameter dictionary
    
    Parameters
    ----------
    encoded: The encoded numpy array
    
    params: The dictionary stores the key how to decode the array

    Returns
    -------
    params_: A dictionary contains decoded parameters
    '''
    params_ = {}
    for idx, p in enumerate(params):
        if type(params[p]) == Int:
            params_[p] = encoded["x"][idx].astype(int)
        elif type(params[p]) == Cat:
            params_[p] = params[p].category[encoded["x"][idx].astype(int)]
        elif type(params[p]) == Real:
            params_[p] = encoded["x"][idx]
    return params_

def convert_to_params(encoded, params):
    '''
    Convert the numpy array (encoded hyperparameter) to
    hyperparameter dictionary

    Parameters
    ----------
    encoded: The encoded numpy array
    
    params: The dictionary stores the key how to decode the array

    Returns
    -------
    params_: A dictionary contains decoded parameters

    '''
    params_ = {}
    if "theta" in params:
        params_["theta"] = encoded
        return params_
    elif "x" in params:
        params_["x"] = encoded
        return params_
    for idx, p in enumerate(params):
        if type(params[p]) == Real:
            params_[p] = encoded[idx]
        elif type(params[p]) == Int:
            params_[p] = encoded[idx].astype(int)
        elif type(params[p]) == Cat:
            params_[p] = params[p].category[encoded[idx].astype(int)]
    return params_
            
    

if __name__ == "__main__":
    from keras.layers import Dense, Conv2D
    params = {
        "Layer1":Cat([Dense, Conv2D]),
        "Layer1_filters":Cat([8, 16, 32]),
        "Layer1_kernel_size":Cat([3, 5, 7]),
        "Layer1_units":Int(12, 64),
        "Layer1_activation":Cat(["relu", "sigmoid"]),

        "Layer2_units":Int(32, 128),
        "Layer2_activation":Cat(["relu", "sigmoid"]),
        "optimizer":Cat(["adam", "sgd", "rmsprop"]),
        "epochs":Int(5, 20),
        "batch_size":Int(64, 256),
        "learning_rate":Real(0.0001, 0.01)
    }
