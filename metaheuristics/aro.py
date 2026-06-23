import numpy as np


class ARO:
    def __init__(self, obj_func, lb, ub, pop_size=100, epochs=1000):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = len(lb)
        self.pop_size = pop_size
        self.epochs = epochs

        # Initialize rabbits (solutions)
        self.pop = np.random.uniform(lb, ub, (pop_size, self.n_dims))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # Find the best rabbit (solution) and score (target)
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.pop[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def correct_solution(self, sol):
        return np.clip(sol, self.lb, self.ub)

    def solve(self):
        history = []
        for epoch in range(self.epochs):
            theta = 2 * (1 - epoch / self.epochs)

            for i in range(self.pop_size):
                L = (np.e - np.exp((epoch / self.epochs) ** 2)) * (np.sin(2 * np.pi * np.random.rand()))
                C = np.zeros(self.n_dims)
                idx = np.random.choice(np.arange(0, self.n_dims), size=int(np.ceil(np.random.rand() * self.n_dims)), replace=False)
                C[idx] = 1
                R = L * C

                A = 2 * np.log(1.0 / np.random.rand()) * theta

                if A > 1:  # Detour foraging
                    j = np.random.randint(self.pop_size)
                    new_pos = (self.pop[j]
                               + R * (self.pop[i] - self.pop[j])
                               + np.round(0.5 * (0.05 + np.random.rand())) * np.random.normal(0, 1))
                else:  # Random hiding
                    g = np.zeros(self.n_dims)
                    idx = np.random.choice(np.arange(self.n_dims), size=int(np.ceil(np.random.rand() * self.n_dims)), replace=False)
                    g[idx] = 1
                    H = np.random.normal(0, 1) * (epoch / self.epochs)
                    b = self.pop[i] + H * g * self.pop[i]
                    new_pos = self.pop[i] + R * (np.random.rand() * b - self.pop[i])

                new_pos = self.correct_solution(new_pos)
                new_fit = self.obj_func(new_pos)
                # Update the position of the individual
                if new_fit < self.fitness[i]:
                    self.pop[i] = new_pos
                    self.fitness[i] = new_fit

            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.pop[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
                history.append([self.best_solution, self.best_fitness])

        return self.best_solution, self.best_fitness, history
