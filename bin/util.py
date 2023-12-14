
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from paramsSpace import convert_to_params

def plot1D(x, f, opt, n_samples, label):
    X = x.reshape(-1, 1)
    y = [f(x_) for x_ in x]
    if hasattr(opt, "_gp"):
        model = opt._gp
    elif hasattr(opt, "gp"):
        model = opt.gp

    if hasattr(opt, "x_history") and hasattr(opt, "y_history"):
        plt.scatter(opt.x_history, opt.y_history, c="blue", s=100, marker="+")

    mean, std = model.predict(X, return_std=True)

    y_samples = model.sample_y(X, n_samples)
    for idx, single_prior in enumerate(y_samples.T):
        # print("Single Prior : ", single_prior)
        plt.plot(
            x,
            single_prior.reshape(x.shape[0]),
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    plt.plot(x, mean, label=label)
    plt.plot(x, y, label="Actual")
    plt.fill_between(
        x,
        mean.reshape(x.shape[0], ) - std,
        mean.reshape(x.shape[0], ) + std,
        alpha=0.1,
        label=fr"{label} mean $\pm$ 1 std. dev.",
    )
    if hasattr(opt, "x_history") and hasattr(opt, "y_history"):
        plt.scatter(opt.x_history, opt.y_history, c="blue", s=100, marker="+")

    plt.legend()
    # plt.show()

def plot2D(x, y, f, opt):
    x_, y_ = np.meshgrid(x, y)
    # z = f(x_, y_)
    z = np.zeros_like(x_)
    for idx in range(x_.shape[0]):
        for jdx in range(x_.shape[1]):
            z[idx, jdx] = f(x_[idx][jdx], y_[idx][jdx])
    print(z.max())
    fig, axs = plt.subplots(ncols=3)
    
    c_actual = axs[0].contourf(x_, y_, z, levels=100, cmap="viridis")
    axs[0].set_title("Actual")
    axs[0].grid()
    axs[0].plot()


    input_mesh = np.column_stack((x_.ravel(), y_.ravel()))
    mean, std = opt.gp.predict(input_mesh, return_std=True)

    z_predict = mean.reshape(x_.shape)
    c_predict = axs[1].contourf(x_, y_, z_predict, levels=100, cmap="viridis")
    axs[1].set_title("Predict")
    axs[1].grid()
    axs[1].plot()

    if hasattr(opt, "x_history") and hasattr(opt, "y_history"):
        axs[1].scatter(opt.x_history[:opt.init, 0], opt.x_history[:opt.init, 1], c="black", s=75, marker="+")
        axs[1].scatter(opt.x_history[opt.init:, 0], opt.x_history[opt.init:, 1], c="blue", s=75, marker="+")


    diff = np.abs(z - z_predict)
    contour_diff = axs[2].contourf(x_, y_, diff, levels=100, cmap='coolwarm')
    axs[2].set_title("Actual - Predicted")
    axs[2].grid()
    cbar_diff = fig.colorbar(contour_diff, ax=axs[2], orientation='vertical')
    # cbar_diff.set_label('')

    fig.colorbar(c_actual, ax=[axs[0], axs[1]], orientation="horizontal") 
    plt.show()

def kdeplot(y_true, y_pred):
    plt.figure(figsize=(24, 8))
    ax1 = sns.kdeplot(data=y_true, color="r", label="Actual")
    sns.kdeplot(data=y_pred, color="b", label="Predicted")
    ax1.set_title("Predicted VS Actual")
    plt.legend()
    plt.show()

def regplot(y_true, y_pred):
    plt.figure(figsize=(40, 25))
    ax = sns.regplot(x=y_true, y=y_pred)
    plt.title("Best Fit Line")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    plt.show()

class Result:
    '''
    Class to store the result of optimization
    '''
    def __init__(self, algo_name, func_name, best_sol, 
                best_sol_decoded, best_fitness, params,
                fitness_history, pop_history) -> None:
        self.__algo_name = algo_name
        self.__func_namae = func_name
        self.x = best_sol
        self.x_decoded = best_sol_decoded
        self.fun = best_fitness
        self.__fitness_history = fitness_history
        self.__pop_history = pop_history
        self.__params = params

    def res(self) -> pd.DataFrame:


        n_iter = self.__pop_history.shape[0]
        pop_decoded = []
        for idx in range(n_iter):
            for p in self.__pop_history[idx]:
                pop_decoded.append(convert_to_params(p, self.__params))
        
        fitness = self.__fitness_history.flatten()

        return pd.DataFrame(
            {
                **{k:[p[k] for p in pop_decoded] for k in self.__params},
                "fitness":fitness,
                "iteration":np.repeat(np.arange(0, n_iter), self.__pop_history.shape[1])
            }
        )

    def func_name(self):
        return self.__func_namae
    
    def algo_name(self):
        return self.__algo_name

    def fitness_history(self):
        return self.__fitness_history
    
    def population_history(self):
        return self.__pop_history
    
    def plot_fitness(self):
        fitness = np.min(self.__fitness_history, axis=1)
        plt.scatter([x for x in range(self.__fitness_history.shape[0])], fitness, marker="+", c="black", s=50)
        plt.plot([x for x in range(self.__fitness_history.shape[0])], fitness)
        plt.title(f"Fitness Function : {self.__func_namae}\nAlgorithm : {self.__algo_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.grid()
        plt.show()

    def plot_fitness_mean_bar(self):
        mean = self.__fitness_history.mean(axis=1)
        plt.bar(x=[x for x in range(self.__fitness_history.shape[0])], height=mean)
        plt.title("Fitness Mean")
        plt.style.use("seaborn-darkgrid")
        plt.show()
