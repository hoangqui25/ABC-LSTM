import numpy as np


class ABC:
    def __init__(self, obj_func, lb, ub, pop_size, epochs, limits, seed=None):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = len(lb)
        self.pop_size = pop_size
        self.epochs = epochs
        self.limits = limits
        self.rng = np.random.default_rng(seed)

        # Initialize population
        self.pop = np.random.uniform(self.lb, self.ub, (pop_size, self.n_dims))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # Initialize trials (for scout phase)
        self.trials = np.zeros(pop_size)

        # Find the best solution
        self.best_solution = self.pop[np.argmin(self.fitness)].copy()
        self.best_fitness = np.min(self.fitness)

    def correct_solution(self, sol):
        return np.clip(sol, self.lb, self.ub)

    def solve(self):
        history = []

        for ep in range(1, self.epochs + 1):
            # --- Employed bees phase ---
            for i in range(self.pop_size):
                rdx = self.rng.choice(list(set(range(self.pop_size)) - {i}))
                phi = self.rng.uniform(-1, 1, self.n_dims)
                new_pos = self.pop[i] + phi * (self.pop[rdx] - self.pop[i])
                new_pos = self.correct_solution(new_pos)
                new_fit = self.obj_func(new_pos)

                if new_fit < self.fitness[i]:
                    self.pop[i] = new_pos
                    self.fitness[i] = new_fit
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

            # --- Onlooker bees phase ---
            prob = (1 / (1 + self.fitness))  # Convert fitness to probability (higher fitness → lower prob)
            prob /= prob.sum()
            for i in range(self.pop_size):
                selected = self.rng.choice(self.pop_size, p=prob)
                rdx = self.rng.choice(list(set(range(self.pop_size)) - {selected}))
                phi = self.rng.uniform(-1, 1, self.n_dims)
                new_pos = self.pop[selected] + phi * (self.pop[rdx] - self.pop[selected])
                new_pos = self.correct_solution(new_pos)
                new_fit = self.obj_func(new_pos)

                if new_fit < self.fitness[selected]:
                    self.pop[selected] = new_pos
                    self.fitness[selected] = new_fit
                    self.trials[selected] = 0
                else:
                    self.trials[selected] += 1

            # --- Scout bees phase ---
            abandoned = np.where(self.trials >= self.limits)[0]
            for i in abandoned:
                self.pop[i] = np.random.uniform(self.lb, self.ub, self.n_dims)
                self.fitness[i] = self.obj_func(self.pop[i])
                self.trials[i] = 0

            # Update global best
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.pop[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]

            # Save history
            history.append([self.best_solution, self.best_fitness])

        return self.best_solution, self.best_fitness, history