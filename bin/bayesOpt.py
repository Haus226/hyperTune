
import acquisition as acqfunc, numpy as np
from gaussRgr import GPR
from scipy.optimize import minimize
import pandas as pd
from paramsSpace import convert_to_bound, convert_to_params, AcqParams, decode_acq_params
from tqdm import tqdm
import optimizer

class BayesianOptimizer:
    def __init__(self, f:callable=None, params=None, init_X=None, init_y=None, n_iter=10, n_init=5, 
                kernel=None, verbose=0, random_state=None,
                **gp_params) -> None:
        self.f = f
        self.__n_iter = n_iter
        self.init = n_init
        self.X_train = init_X
        self.y_train = init_y
        self.acq_history = []
        self.params = params
        self.random_state = random_state
        self.__verbose = verbose
        self.best_params:dict
        self.best_score = None

        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            assert isinstance(random_state, np.random.RandomState)

        if self.X_train is None and self.y_train is None and f is not None:
            self.X_train = np.zeros((n_init, len(params)))
            for idx, p in enumerate(params):
                self.X_train[:, idx] = params[p].sample(shape=n_init)

            self.y_train = np.zeros((n_init, ))

            iteration = range(self.X_train.shape[0])
            if verbose:
                iteration = tqdm(iteration, desc=f"Initializing Bayesian Opt")
            for idx in iteration:
                self.y_train[idx] = f(**(convert_to_params(self.X_train[idx], self.params)))                
            if self.y_train.ndim == 1:
                self.y_train = self.y_train.reshape(-1, 1)

        self.pbounds = convert_to_bound(params)

        # Set the default acquisition function
        self.acqFunc = lambda x: acqfunc.AcqFunc(gp=self.gp).func(x.reshape(1, -1), "ucb", kappa=2.576)
        self.gp = GPR(kernel=kernel, random_state=self.random_state, **gp_params)

    def set_func(self, f:callable, params:dict=None):
        if params is not None:
            self.params = params
            self.pbounds = convert_to_bound(params)     
        else:
            params = self.params

        if self.X_train is None and self.y_train is None:
            self.X_train = np.zeros((self.init, len(params)))
            for idx, p in enumerate(params):
                self.X_train[:, idx] = params[p].sample(shape=self.init)

            self.y_train = np.zeros((self.init, ))
            iteration = range(self.X_train.shape[0])
            if self.__verbose:
                iteration = tqdm(iteration, desc=f"Initializing Bayesian Opt")
            for idx in iteration:
                self.y_train[idx] = f(**(convert_to_params(self.X_train[idx], self.params)))                
            if self.y_train.ndim == 1:
                self.y_train = self.y_train.reshape(-1, 1)
        self.f = f

    def set_params(self, params:dict):
        self.params = params

    def set_acqfunc(self, acqfunction:["ucb", "lcb", "pi", "ei"]="ucb", **acq_params):
        '''
        Set acquisition function and pass in the parameters of acquisition function, for
        detail, refer to AcqFunc class
        '''
        self.__acq_type = acqfunction
        self.__acq_params = acq_params

    def optimize(self, opt="de",  n_samples=25, **opt_params):
        if (self.f is None or self.params is None):
            raise ValueError("Please ensure you provide target function and the search space")
        
        params = {
            "x":AcqParams(self.params)
        }

        iteration = range(self.__n_iter)
        if self.__verbose:
            iteration = tqdm(iteration, desc="Bayesian Optimization")

        for idx in iteration:
            if self.__verbose:
                iteration.set_postfix_str(f"best_f : {round(self.y_train.max(), 5)}")
            max_acq = -np.inf
            self.gp = self.gp.fit(self.X_train, self.y_train)
            self.acqFunc = lambda x: acqfunc.AcqFunc(gp=self.gp).func(x.reshape(1, -1), self.__acq_type, y_opt=self.y_train.max(), **self.__acq_params)
                        
            if opt == "de":
                res = optimizer.DifferentialEvol(self.acqFunc, params, 
                                        popsize=n_samples,
                                        random_state=self.random_state,
                                        **opt_params).optim()
            elif opt == "hs":
                res = optimizer.HarmonySearch(self.acqFunc, params,
                                        popsize=n_samples, 
                                        random_state=self.random_state,
                                        **opt_params).optim()
            elif opt == "pso":
                res = optimizer.ParticleSwarm(self.acqFunc, params, 
                                        popsize=n_samples,
                                        random_state=self.random_state,
                                        **opt_params).optim()
            x_max = res.x
            x_max_encoded = res.x_decoded
            max_acq = res.fun

            self.acq_history.append(max_acq.squeeze() if type(max_acq) == np.array else max_acq)
            self.X_train = np.vstack((self.X_train, [x_max]))
            self.y_train = np.vstack((self.y_train, [self.f(**decode_acq_params(x_max_encoded, self.params))]))

        best_idx = np.argmax(self.y_train)
        self.best_params = convert_to_params(self.X_train[best_idx], self.params)
        self.best_score = np.max(self.y_train)

    def res(self) -> pd.DataFrame:
        acq = ["/"] * self.init
        acq.extend(self.acq_history)
        decoded = []
        for x in self.X_train:
            decoded.append(convert_to_params(x, self.params))
        df = pd.DataFrame(
            {
                "Acq":acq,
                "func":self.y_train.flatten(),
                **{k:[p[k] for p in decoded] for k in self.params},
            }
        )
        return df






