import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class DecoderProblem(Problem):
    def __init__(self, encoder, target, lower, upper):
        super().__init__(n_var=len(lower), n_obj=1, n_constr=0, xl=lower, xu=upper,
                         elementwise_evaluation=False)
        self._encoder = encoder
        self._target = target

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.linalg.norm(self._encoder(x) - self._target, axis=1)


class OptDecoder:
    def __init__(self, handler, x_lower, x_upper, verbose=False, pop_size=100, n_gen=1000, seed=1):
        self._handler = handler
        self._x_lower = x_lower
        self._x_upper = x_upper
        self._verbose = verbose
        self._pop_size = pop_size
        self._n_gen = n_gen
        self._seed = seed

    def decode(self, target):
        algo = PSO(pop_size=self._pop_size)
        res = minimize(DecoderProblem(self._handler, target, self._x_lower, self._x_upper),
                       algo,
                       seed=self._seed,
                       verbose=self._verbose)
        if self._verbose:
            print(f'Fitness {res.F}')
        return np.ravel(res.X)
