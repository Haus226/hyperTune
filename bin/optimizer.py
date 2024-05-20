
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from util import Result
from paramsSpace import Real, Int, Cat, convert_to_bound, convert_to_params
import sys

class Optimizer(ABC):
    '''
    Abstract optimizer class
    '''
    def __init__(self, func, params,
                popsize, n_iter, ttl,
                verbose, random_state
                ) -> None:
        
        self.func = func
        self.params = params
        self.popsize = popsize
        self.n_iter = n_iter
        self.verbose = verbose
        self.ttl = ttl

        self.best_score = None
        self.best_params = None
        self.pbounds = None
        self.dimensions = len(self.params) if self.params is not None else None

        # Seed the numpy first
        self.verify_random_state(random_state)
        if self.func is None and self.params is None:
            self.init_pop_fitness()


    @abstractmethod
    def optim(self):
        pass
    
    def init_dimension(self):
        '''
        Get the dimensions of the search space
        '''
        if "theta" in self.params:
            self.dimensions = self.params["theta"].interval().shape[0]
        elif "x" in self.params:
            self.dimensions = self.params["x"].interval().shape[0]
        else:
            self.dimensions = len(self.params)

    def preprocess(self):
        '''
        Record the indices of Int and Cat types in the search space
        '''
        if "theta" in self.params:
            mask_int = np.array([])
            mask_cat = np.array([])
        elif "x" in self.params:
            mask_int = self.params["x"].Int_idxs()
            mask_cat = self.params["x"].Cat_idxs()
        else:
            mask_int = []
            mask_cat = []
            for idx, p in enumerate(self.params):
                if type(self.params[p]) == Int: 
                    mask_int.append(idx)
                elif type(self.params[p]) == Cat:
                    mask_cat.append(idx)
            mask_int = np.array(mask_int)
            mask_cat = np.array(mask_cat)
        return mask_int, mask_cat

    def init_pop_fitness(self):
        '''
        Initialize the population and calculate the fitness of individual
        '''
        self.pop = np.zeros(shape=(self.popsize, self.dimensions))
        if "theta" in self.params:
            self.pop = self.params["theta"].sample((self.popsize, self.dimensions))
        elif "x" in self.params:
            self.pop = self.params["x"].sample((self.popsize, self.dimensions))
        else:
            for idx, p in enumerate(self.params):
                self.pop[:, idx] = self.params[p].sample(shape=self.popsize)
        self.fitness = np.zeros((self.popsize, ))
        iteration = range(self.popsize)

        if self.verbose:
            iteration = tqdm(iteration, desc=f"Initializing {self.__class__.__name__}")
        for idx, ind in zip(iteration, self.pop):      
            f = self.func(**convert_to_params(ind, self.params))
            self.fitness[idx] = f


    def set_func(self, func:callable):
        '''
        Set the function after the instance was initialized
        '''
        self.func = func

    def set_params(self, params:dict):
        '''
        Set the search space after the instance was initialized
        '''
        self.params = params


    def verify_func_params(self):
        '''
        Verify that the target function and parameters range are provided
        '''
        if self.func is None:
            ValueError("Please provide your function")
        elif self.params is None:
            ValueError("Please provide your parameters")

    def verify_random_state(self, random_state):
        if isinstance(random_state, int):
            return np.random.seed(random_state)
        elif random_state is None:
            return np.random.seed()
        
class DifferentialEvol(Optimizer):
    def __init__(self, func:callable=None, params:dict=None, mut_1=0.9, mut_2=0.9, 
                crossp=0.95, popsize=10, ttl=2,
                n_iter=20, verbose=0, random_state=None) -> None:
        super().__init__(func, params, popsize, n_iter, ttl, verbose, random_state)
        self.mut_1 = mut_1
        self.mut_2 = mut_2
        self.crossp = crossp

    def optim(self) -> Result:
        self.verify_func_params()
        self.init_dimension()
        self.init_pop_fitness()

        assert(self.popsize >= 3)
        
        age = np.ones((self.popsize, )) * np.inf
        if self.ttl > 0:
            age = np.ones((self.popsize, )) * self.ttl
            
        
        mask_int, mask_cat = self.preprocess()

        self.pbounds = convert_to_bound(self.params)
        min_b, max_b = self.pbounds.T

        pop_history = np.zeros((self.n_iter + 1, self.popsize, self.dimensions))
        fitness_history = np.zeros((self.n_iter + 1, self.popsize))
        pop_history[0] = self.pop.copy()
        fitness_history[0] = self.fitness.copy()

        best_idx = np.argmax(self.fitness)
        best_x = self.pop[best_idx]
        self.best_score = self.fitness[best_idx]

        for idx in range(self.n_iter):
            iteration = range(self.popsize)
            if self.verbose:
                iteration = tqdm(iteration, file=sys.stdout)

            for jdx in iteration:
                age[jdx] -= 1
                if self.verbose:
                    iteration.set_description_str(desc=f"Differential Evol {idx + 1}")
                    iteration.set_postfix_str(f"best_f : {round(self.best_score, 5)}")
                idxs = [kdx for kdx in range(self.popsize) if kdx != jdx]

                # Mutation strategy
                a, b, c = self.pop[np.random.choice(idxs, 3, replace = False)]
                mutant = a + self.mut_1 * (b - c) + self.mut_2 * (best_x - a)

                # Boundary checking
                mutant[mutant > max_b] = max_b[mutant > max_b]
                mutant[mutant < min_b] = min_b[mutant < min_b]

                # Cross over
                cross_points = np.random.rand(self.dimensions) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True

                trial = np.where(cross_points, mutant, self.pop[jdx])
                if mask_int.size:
                    trial[mask_int] = np.round(trial[mask_int])
                if mask_cat.size:
                    trial[mask_cat] = np.round(trial[mask_cat])
                
                trial_ = convert_to_params(trial, self.params)
                f = self.func(**trial_)

                if f > self.fitness[jdx] or (age[jdx] == -1):
                    self.fitness[jdx] = f
                    self.pop[jdx] = trial
                    age[jdx] = self.ttl
                    if f > self.best_score:
                        best_x = trial
                        self.best_score = f

            fitness_history[idx + 1] = self.fitness.copy()
            pop_history[idx + 1] = self.pop.copy()
        self.best_params = convert_to_params(best_x, self.params)

        return Result(self.__class__.__name__, self.func.__name__, 
                    best_x, self.best_params, self.best_score, 
                    self.params, fitness_history, pop_history)

class HarmonySearch(Optimizer):
    def __init__(self, func:callable=None, params:dict=None, HMCR=0.7, PAR=0.3, BW:np.array=None, 
                popsize=10, n_iter=20, ttl=2, verbose=0, random_state=None) -> None:
        super().__init__(func, params, popsize, n_iter, ttl, verbose, random_state)
        self.HMCR = HMCR
        self.PAR = PAR
        self.BW = BW
    
    def optim(self):
        self.verify_func_params()
        self.init_dimension()
        self.init_pop_fitness()
        
        age = np.ones((self.popsize, )) * np.inf
        if self.ttl > 0:
            age = np.ones((self.popsize, )) * self.ttl
        
        mask_int, mask_cat = self.preprocess()

        harmony_history = np.zeros((self.n_iter + 1, self.popsize, self.dimensions))
        fitness_history = np.zeros((self.n_iter + 1,self. popsize))

        self.pbounds = convert_to_bound(self.params)
        min_b, max_b = self.pbounds.T
        diff = max_b - min_b
        if self.BW is not None:
            assert(self.BW.shape[0] == self.dimensions)
            diff = self.BW

        harmony_history[0] = self.pop.copy()
        fitness_history[0] = self.fitness.copy()

        best_idx = np.argmax(self.fitness)
        worst_idx = np.argmin(self.fitness)
        best_harmony = self.pop[best_idx]
        self.best_score = self.fitness[best_idx]

        for idx in range(self.n_iter):
            r1_ = np.random.rand(self.popsize, self.dimensions)
            r2_ = np.random.rand(self.popsize, self.dimensions)
            r3_ = np.random.uniform(low=-1, high=1.001, size=(self.popsize, self.dimensions))
            iteration = range(self.popsize)
            if self.verbose:
                iteration = tqdm(iteration, file=sys.stdout)

            for jdx in iteration:
                age[jdx] -= 1
                if self.verbose:
                    iteration.set_description_str(desc=f"Harmony Search {idx + 1}")
                    iteration.set_postfix_str(f"best_f : {round(self.best_score, 5)}")
                trial = np.zeros(self.dimensions)
                for kdx in range(self.dimensions):
                    if r1_[jdx][kdx] < self.HMCR:
                        trial[kdx] = self.pop[np.random.randint(0, self.popsize)][kdx]
                    else:
                        trial[kdx] = np.random.uniform(min_b[kdx], max_b[kdx] + 0.001)
                    if r2_[jdx][kdx] < self.PAR:
                        trial[kdx] += r3_[jdx][kdx] * diff[kdx]
                trial[trial > max_b] = max_b[trial > max_b]
                trial[trial < min_b] = min_b[trial < min_b]
                if mask_int.size:
                    trial[mask_int] = np.round(trial[mask_int])
                if mask_cat.size:
                    trial[mask_cat] = np.round(trial[mask_cat])

                trial_ = convert_to_params(trial, self.params)
                f = self.func(**trial_)

                # Update worst_idx first and followed by best_idx
                # Ignore the trial harmony if worse than the worst harmony
                if (age[jdx] == -1):
                    self.pop[jdx] = trial
                    self.fitness[jdx] = f
                    age[jdx] = self.ttl
                elif f > self.fitness[worst_idx]:
                    self.pop[worst_idx] = trial
                    self.fitness[worst_idx] = f
                    age[jdx] = self.ttl
                if f > self.best_score:
                    best_harmony = trial
                    self.best_score = f
                worst_idx = np.argmin(self.fitness)
            self.best_params = convert_to_params(best_harmony, self.params)
            fitness_history[idx + 1] = self.fitness.copy()
            harmony_history[idx + 1] = self.pop.copy()

        return Result(self.__class__.__name__, self.func.__name__, 
                    best_harmony, self.best_params, self.best_score, 
                    self.params, fitness_history, harmony_history)

class ParticleSwarm(Optimizer):
    def __init__(self, func:callable=None, params:dict=None, inertia=.5, cognitive=1.5, social=1.5, 
                popsize=10, n_iter=20, ttl=2,
                verbose=0, random_state=None) -> None:
        super().__init__(func, params, popsize, n_iter, ttl, verbose, random_state)
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optim(self) -> Result:

        self.verify_func_params()
        self.init_dimension()
        self.init_pop_fitness()
        
        age = np.ones((self.popsize, )) * np.inf
        if self.ttl > 0:
            age = np.ones((self.popsize, )) * self.ttl
        
        mask_int, mask_cat = self.preprocess()

        swarm_history = np.zeros((self.n_iter + 1, self.popsize, self.dimensions))
        fitness_history = np.zeros((self.n_iter + 1, self.popsize))

        self.pbounds = convert_to_bound(self.params)
        min_b, max_b = self.pbounds.T
        diff = np.fabs(min_b - max_b)

        swarm_history[0] = self.pop.copy()
        fitness_history[0] = self.fitness.copy()
        velocity = np.random.uniform(-np.abs(diff), np.abs(diff), (self.popsize, self.dimensions))
        swarm_best = self.pop.copy()

        best_idx = np.argmax(self.fitness)
        best_position = swarm_best[best_idx]
        self.best_score = self.fitness[best_idx]

        for idx in range(self.n_iter):
            r_p = np.random.rand(self.popsize, self.dimensions)
            r_s = np.random.rand(self.popsize, self.dimensions)
            iteration = range(self.popsize)
            if self.verbose:
                iteration = tqdm(iteration, file=sys.stdout)

            for jdx in iteration:
                age[jdx] -= 1
                if self.verbose:
                    iteration.set_description_str(f"Swarm Particle {idx + 1}")
                    iteration.set_postfix_str(f"best_f : {round(self.best_score, 5)}")
                velocity[jdx, :] = self.inertia * velocity[jdx, :] + self.cognitive * r_p[jdx, :] * (swarm_best[jdx, :] - self.pop[jdx, :]) + \
                                            self.social * r_s[jdx, :] * (best_position - self.pop[jdx, :])
                self.pop[jdx, :] += velocity[jdx, :]

                self.pop[jdx, self.pop[jdx, :] > max_b] = max_b[self.pop[jdx, :] > max_b]
                self.pop[jdx, self.pop[jdx, :] < min_b] = min_b[self.pop[jdx, :] < min_b]
                if mask_int.size:
                    self.pop[jdx][mask_int] = np.round(self.pop[jdx][mask_int])
                if mask_cat.size:
                    self.pop[jdx][mask_cat] = np.round(self.pop[jdx][mask_cat])
                trial_ = convert_to_params(self.pop[jdx], self.params)
                
                f = self.func(**trial_)
                if f > self.fitness[jdx] or (age[jdx] == -1):
                    swarm_best[jdx] = self.pop[jdx]
                    self.fitness[jdx] = f
                    age[jdx] = self.ttl
                    if f > self.best_score:
                        best_position = swarm_best[jdx]
                        self.best_score = f

            fitness_history[idx + 1] = self.fitness.copy()
            swarm_history[idx + 1] = self.pop.copy()


        self.best_params = convert_to_params(best_position, self.params)

        return Result(self.__class__.__name__, self.func.__name__, 
                    best_position, self.best_params, self.best_score, 
                    self.params, fitness_history, swarm_history)


