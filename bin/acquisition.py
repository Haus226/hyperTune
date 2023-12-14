
from scipy.stats import norm
import numpy as np

class AcqFunc:
    '''
    Acquisition funciton object
    '''
    def __init__(self, gp) -> None:
        self.__gp = gp

    def func(self, X, kind:["ucb", "lcb", "pi", "ei", "mix"], **acq_params) -> callable:
        if kind == "ucb":
            return self.GP_UCB(X, **acq_params)
        elif kind == "lcb":
            return self.GP_LCB(X, **acq_params)
        elif kind == "ei":
            return self.GP_EI(X, **acq_params)
        elif kind == "pi":
            return self.GP_PI(X, **acq_params)
        elif kind == "mix":
            pi = self.GP_PI(X, **acq_params)
            ei = self.GP_EI(X, **acq_params)
            ucb = self.GP_UCB(X, **acq_params)
            sum = pi + ei + ucb
            return (pi * pi / sum) + (ei * ei / sum) + (ucb * ucb / sum)
        
    def GP_EI(self, X, xi=0.01, kappa=0, y_opt:float=0):
        '''
        Expected Improvement
        '''
        mean, std = self.__gp.predict(X, return_std=True)
        values = np.zeros_like(mean)
        mask = std > 0
        improve = y_opt - xi - mean[mask]
        scaled = improve / std[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = std[mask] * pdf
        values[mask] = exploit + explore
        return values

    def GP_PI(self, X, xi=0.01, kappa=0, y_opt:float=0):
        '''
        Probability of Improvement
        '''
        mean, std = self.__gp.predict(X, return_std=True)
        values = np.zeros_like(mean)
        mask = std > 0
        improve = y_opt - xi - mean[mask]
        scaled = improve / std[mask]
        values[mask] = norm.cdf(scaled)
        return values

    def GP_LCB(self, X, xi=0, kappa=1.96, y_opt=0):
        '''
        Lower Confidence Bound
        '''
        mean, std = self.__gp.predict(X, return_std=True)
        return mean - kappa * std

    def GP_UCB(self, X, xi=0, kappa=1.96, y_opt=0):
        '''
        Upper Confidence Bound
        '''
        mean, std = self.__gp.predict(X, return_std=True)
        return mean + kappa * std
