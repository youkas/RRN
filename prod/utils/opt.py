import numpy as np
from pymoo.model.problem import Problem
from smt.sampling_methods import LHS

class Mapper:
    def __init__(self, model, nominal=None, lower=None, upper=None, size=1, resample=True):
        self.model = model
        self.nominal = np.reshape(nominal, (1, -1)) if nominal is not None else None
        self.lower = lower
        self.upper = upper
        self.size = size
        self.resample = resample
        self.master_points = None
        self.at_master_points = None
        self.at_nominal = None

    def evaluate(self, X, size = None, resample=None):
        if self.nominal is None:
            return self._evaluate_(X)

        if self.lower is None:
            return self._multi_evaluate_(X, self.nominal)

        if size is not None:
            self.size = size
        if resample is not None:
            self.resample = resample
        if self.resample or self.master_points is None:
            self.master_points = self._get_master_points_()

        obj_fun = self._multi_evaluate_(X, np.vstack([self.master_points, self.nominal]))
        self.at_master_points = obj_fun[:, :-1, :]
        self.at_nominal = obj_fun[:, -1, :].reshape((len(X), -1))
        return self.at_nominal, self.at_master_points

    def _get_master_points_(self):
        return LHS(xlimits=np.array([[l, self.upper[i]] for i, l in enumerate(self.lower)]))(self.size)

    def _evaluate_(self, x):
        y = self.model.predict(x)
        return y.reshape((len(x), -1))

    def _multi_evaluate_(self, x, master_points):
        size_x, dim_x = x.shape
        size_mp, dim_mp = master_points.shape

        values = np.hstack([np.repeat(x, size_mp, axis=0), np.tile(master_points, (size_x, 1))])

        y = self.model.predict(values)
        return y.reshape((size_x, size_mp, -1))

class Evaluator:
    def __init__(self, model, objective_function, nominal=None, lower=None, upper=None, size=1, resample=True):
        self.model = model
        self.objective_function = objective_function
        self.nominal = np.reshape(nominal, (1, -1)) if nominal is not None else None
        self.lower = lower
        self.upper = upper
        self.size = size
        self.resample = resample
        self.master_points = None
        self.at_master_points = None
        self.at_nominal = None

    def evaluate(self, X, size = None, resample=None):
        if self.nominal is None:
            return self._evaluate_(X)

        if self.lower is None:
            return self._multi_evaluate_(X, self.nominal)

        if size is not None:
            self.size = size
        if resample is not None:
            self.resample = resample
        if self.resample or self.master_points is None:
            self.master_points = self._get_master_points_()

        obj_fun = self._multi_evaluate_(X, np.vstack([self.master_points, self.nominal]))
        self.at_master_points = obj_fun[:, :-1]
        self.at_nominal = obj_fun[:, -1].reshape((-1, 1))
        return self.at_nominal, self.at_master_points

    def check(self, points, fitness, xlimits):
        for i, point in enumerate(points):
            self.check_point(point.reshape((1, -1)), fitness[i], xlimits)

    def check_point(self, point, fitness, xlimits):
        print('Point:')
        print(point)
        print('Fitness:')
        print(fitness)
        print('Decoded Point:')
        X = self.model.decode(point, xlimits[0], xlimits[1], verbose=False)
        print(X)
        t1 = self.model.encode(X)
        print('Encoded Point:')
        print(t1)
        y1 = self.evaluate(t1)
        print('Predicted at nominal fitness:')
        print(y1)
        print('Decoded at nominal fitness error %:')
        print(100 * (y1 - fitness) / y1)

    def _get_master_points_(self):
        return LHS(xlimits=np.array([[l, self.upper[i]] for i, l in enumerate(self.lower)]))(self.size)

    def _evaluate_(self, x):
        y = self.model.predict(x)
        return self.objective_function(y).reshape((-1, 1))

    def _multi_evaluate_(self, x, master_points):
        size_x, dim_x = x.shape
        size_mp, dim_mp = master_points.shape

        values = np.hstack([np.repeat(x, size_mp, axis=0), np.tile(master_points, (size_x, 1))])

        y = self.model.predict(values)
        obj_fun = self.objective_function(y)

        return obj_fun.reshape((-1, size_mp))

class DeterministProblem(Problem):
    def __init__(self, dimension, evaluator, xlimits):
        super().__init__(n_var=dimension,
                         n_obj=1,
                         n_constr=0,
                         xl=xlimits[0],
                         xu=xlimits[1],
                         elementwise_evaluation=False)

        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.evaluator.evaluate(X)

class DebRobustProblem(Problem):
    def __init__(self, dimension, evaluator, xlimits, Type=2, eta=0.1):
        super().__init__(n_var=dimension,
                         n_obj=1,
                         n_constr=1 if Type==2 else 0,
                         xl=xlimits[0],
                         xu=xlimits[1],
                         elementwise_evaluation=False)
        assert isinstance(evaluator, Evaluator)
        assert Type in [1, 2]
        self.evaluator = evaluator
        self.Type = Type
        self.eta = eta

    def _evaluate(self, X, out, *args, **kwargs):
        at_nominal, at_master_points = self.evaluator.evaluate(X)

        effective = np.mean(at_master_points, axis=1).reshape((-1, 1))

        if self.Type == 1:
            out["F"] = effective
        else:
            constraint = np.linalg.norm((at_nominal - effective)/effective, axis=1).reshape((-1, 1))
            out["F"] = at_nominal
            out["G"] = constraint - self.eta

class RobustProblem(Problem):
    def __init__(self, dimension, evaluator, xlimits):
        super().__init__(n_var=dimension,
                         n_obj=2,
                         n_constr=0,
                         xl=xlimits[0],
                         xu=xlimits[1],
                         elementwise_evaluation=False)
        assert isinstance(evaluator, Evaluator)
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        at_nominal, at_master_points = self.evaluator.evaluate(X)
        obj_fun = np.hstack([at_nominal, at_master_points])
        out["F"] = np.hstack([np.mean(obj_fun, axis=1).reshape((-1, 1)), np.std(obj_fun, axis=1).reshape((-1, 1))])
