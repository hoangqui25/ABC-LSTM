import numpy as np


class SMA:
    def __init__(self, obj_func, lb, ub, pop_size, epochs, p_t=0.03, seed=None):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = len(lb)
        self.pop_size = pop_size
        self.epochs = epochs
        self.p_t = p_t
        self.rng = np.random.default_rng(seed)
        self.EPSILON = 1e-9

        # Initialize population
        self.pop = np.random.uniform(lb, ub, (pop_size, self.n_dims))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # Find the best soulution
        self.best_solution = self.pop[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def correct_solution(self, sol):
        return np.clip(sol, self.lb, self.ub)

    def update_population(self, new_pop, new_fit):
        for i in range(self.pop_size):
            if new_fit[i] < self.fitness[i]:
                self.pop[i] = new_pop[i]
                self.fitness[i] = new_fit[i]

    def solve(self):
        history = []
        for ep in range(1, self.epochs + 1):
            sort_idx = np.argsort(self.fitness)
            self.pop = self.pop[sort_idx]
            self.fitness = self.fitness[sort_idx]

            # Find the best and the worst solution
            f_best = self.fitness[0]
            f_worst = self.fitness[-1]
            self.best_solution = self.pop[0]
            self.best_fitness = f_best
            history.append([self.best_solution, self.best_fitness])

            # Plus eps to avoid denominator zero
            ss = f_best - f_worst + self.EPSILON
            weights = np.zeros((self.pop_size, self.n_dims))

            # Calculate the fitness weight of each slime mold
            for i in range(self.pop_size):
                # Eq.(2.5)
                if i <= self.pop_size / 2:
                    weights[i] = 1 + self.rng.uniform(0, 1, self.n_dims) * np.log10((f_best - self.fitness[i]) / ss + 1)
                else:
                    weights[i] = 1 - self.rng.uniform(0, 1, self.n_dims) * np.log10((f_best - self.fitness[i]) / ss + 1)

            aa = np.arctanh(1 - ep / self.epochs)  # Eq.(2.4)
            bb = 1 - ep / self.epochs

            # new_pop = np.zeros_like(self.pop)

            for i in range(self.pop_size):
                if self.rng.random() < self.p_t:  # Eq.(2.7)
                    pos_new = self.lb + (self.ub - self.lb) * self.rng.random(self.n_dims)
                else:
                    p = np.tanh(abs(self.fitness[i] - f_best))  # Eq.(2.2)
                    vb = self.rng.uniform(-aa, aa, self.n_dims)  # Eq.(2.3)
                    vc = self.rng.uniform(-bb, bb, self.n_dims)
                    pos_new = self.pop[i].copy()
                    for j in range(self.n_dims):
                        # Two positions randomly selected from population
                        id_a, id_b = self.rng.choice(list(set(range(self.pop_size)) - {i}), 2, replace=False)
                        if self.rng.random() < p:  # Eq.(2.1)
                            pos_new[j] = self.best_solution[j] + vb[j] * (weights[i, j] * self.pop[id_a, j] - self.pop[id_b, j])
                        else:
                            pos_new[j] = vc[j] * pos_new[j]

                pos_new = self.correct_solution(pos_new)
                self.pop[i] = pos_new
                # new_pop[i] = pos_new

            # self.pop = new_pop
            self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_solution = self.pop[best_idx].copy()
            self.best_fitness = self.fitness[best_idx]
            history.append([self.best_solution, self.best_fitness])

        return self.best_solution, self.best_fitness, history
