
from cholesky import cholesky, cholesky_solve, forwardSubstituition
import numpy as np
from scipy.linalg import cholesky as cho, cho_solve, solve_triangular
import sklearn.gaussian_process.kernels as kernels
import scipy.optimize
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import optimizer
from paramsSpace import RealKernels

class GPR:
    def __init__(
        self,          
        kernel=None, 
        epsilon=1e-10,
        random_state=None,
        n_restarts_optimizer=0,
        normalize_y=False,
        optimizer="de"
        ) -> None:
        self.y_mean: np.array
        self.epsilon: np.array
        self.alpha: np.array
        self.L: np.array
        self.kernel_ = kernel
        self.epsilon = epsilon
        self.random_state = random_state
        self.X_train, self.y_train = None, None
        self.n_restarts_optimizer= n_restarts_optimizer
        self.normalize_y = normalize_y
        self.optimizer=optimizer

    def fit(self, X, y):
        if self.random_state is None or self.random_state is np.random:
            self.random_state = np.random.mtrand._rand
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)

        if self.kernel_ is None:
            self.kernel = kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * kernels.RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            # Use self.kernel so that we get the unfitted kernel not keep 
            # recloning the fitted one since self.kernel_ holds the 
            # original unfitted kernel
            self.kernel = kernels.clone(self.kernel_)

        if self.normalize_y:
            scaler = StandardScaler()
            y = scaler.fit_transform(y)
            self._y_train_mean = scaler.mean_
            self._y_train_std = scaler.scale_

        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1

        self.X_train = np.copy(X)
        self.y_train = np.copy(y)


        if self.kernel.n_dims > 0:
            def kernelFunc(theta, evaluate_grad=True if self.optimizer=="l-bfgs-b" else False):
                if evaluate_grad:
                    lml, grad = self.log_marginal_likelihood(theta, evaluate_grad)
                    return -lml, -grad
                else:
                    lml = self.log_marginal_likelihood(theta, evaluate_grad)
                    return lml
            if self.optimizer == "l-bfgs-b":
                optima = [
                    (
                        self.optimization(
                            kernelFunc, self.kernel.theta, self.kernel.bounds, self.optimizer
                        )
                    )
                ]
                if self.n_restarts_optimizer > 0:
                    bounds = self.kernel.bounds
                    for iteration in range(self.n_restarts_optimizer):
                        theta_initial = self.random_state.uniform(bounds[:, 0], bounds[:, 1])
                        optima.append(
                            self.optimization(kernelFunc, theta_initial, bounds, self.optimizer)
                        )

                # Get the log-marginal-likelihood value
                lml_values = list(map(itemgetter(1), optima))
                # Get the kernel theta with the smallest negative log-marginal-likelihood value(which means largest)
                self.kernel.theta = optima[np.argmin(lml_values)][0]
                self.kernel._check_bounds_params()
            elif self.optimizer == "de":
                res = optimizer.DifferentialEvol(kernelFunc, {"theta":RealKernels(self.kernel.bounds)}, 
                                        n_iter=self.n_restarts_optimizer * 20 if self.n_restarts_optimizer > 0 else 20,
                                        verbose=1).optim()
            elif self.optimizer == "hs":
                res = optimizer.HarmonySearch(kernelFunc, {"theta":RealKernels(self.kernel.bounds)}, 
                                        n_iter=self.n_restarts_optimizer * 20 if self.n_restarts_optimizer > 0 else 20,
                                        verbose=1).optim()
            elif self.optimizer == "pso":
                res = optimizer.ParticleSwarm(kernelFunc, {"theta":RealKernels(self.kernel.bounds)}, 
                                        n_iter=self.n_restarts_optimizer * 20 if self.n_restarts_optimizer > 0 else 20,
                                        verbose=1).optim()
            if self.optimizer in ["de", "hs", "pso"]:
                self.kernel.theta = res.x



        K = self.kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.epsilon
        self.L = cholesky(K)
        self.alpha = cholesky_solve(self.L, self.y_train)

        return self
    
    def predict(self, X, return_std:bool=False, return_cov:bool=False):
        if self.X_train is None:
            self.y_mean = np.zeros(X.shape[0])
            if return_cov and return_std:
                y_cov = self.kernel(X)
                return self.y_mean, y_cov, np.diag(y_cov)
            elif return_cov:
                y_cov = self.kernel(X)
                return self.y_mean, y_cov
            elif return_std:
                y_var = np.diag(self.kernel(X))
                return self.y_mean, np.sqrt(y_var)
            else:
                return self.y_mean
        else:
            K_ = self.kernel(X, self.X_train)
            y_mean = K_ @ self.alpha
            y_mean = self._y_train_std * y_mean + self._y_train_mean

        V = forwardSubstituition(self.L, K_.T)
        if return_cov:
            y_cov = self.kernel(X) - V.T @ V
            y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(*y_cov.shape, -1)
            if y_cov.shape[2] == 1:
                y_cov = np.squeeze(y_cov, axis=2)
            return y_mean, y_cov
        elif return_std:
            y_var = self.kernel.diag(X)
            y_var -= np.einsum("ij,ji->i", V.T, V)
            y_var = np.outer(y_var, self._y_train_std ** 2).reshape(*y_var.shape, -1)
            if y_var.shape[1] == 1:
                y_var = np.squeeze(y_var, axis=1)
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean


    def sample_y(self, X, n_samples=1, random_state=0):
        rng = np.random.RandomState(random_state)
        y_mean, y_cov = self.predict(X, return_cov=True)
        y_samples = rng.multivariate_normal(y_mean.reshape(y_mean.shape[0]), y_cov, n_samples).T
        return y_samples
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def log_marginal_likelihood(self, theta, evaluate_grad):
        kernel = self.kernel
        kernel.theta = theta
        if evaluate_grad:
            K, K_gradient = kernel(self.X_train, eval_gradient=True)
        else:
            K = kernel(self.X_train)

        K[np.diag_indices_from(K)] += self.epsilon

        L = cholesky(K)
        if self.y_train.ndim == 1:
            y_train = self.y_train[..., np.newaxis]
        else:
            y_train = self.y_train

        alpha = cholesky_solve(L, y_train)
        
        log_likelihood_dims = -0.5 * np.einsum("ik, ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)

        log_likelihood = log_likelihood_dims.sum(axis=-1)

        if evaluate_grad:
            # Compute outer product
            # Equivalent to the following lines:
            # for idx in range(alpha.shape[0]):
            #     for jdx in range(alpha.shape[0]):
            #         E[idx, jdx, :] = alpha[idx, :] * alpha[jdx, :]
            inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
            K_inv = cholesky_solve(L, np.eye(K.shape[0]))
            inner_term -= K_inv[..., np.newaxis]
            # Compute trace
            log_likelihood_gradient_dims = 0.5 * np.einsum(
                "ijl,jik->kl", inner_term, K_gradient
            )
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood


        
    def optimization(self, obj_func, initial_theta, bounds, optimizer):
        if optimizer == "l-bfgs-b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )

        return opt_res.x, opt_res.fun
