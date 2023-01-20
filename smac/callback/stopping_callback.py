__copyright__ = "Copyright 2023, automl.org"
__license__ = "3-clause BSD"

import numpy as np

from smac.main.smbo import SMBO
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue


class StoppingCallback(Callback):
    """Callback implementing the stopping criterion by Makarova et al. (2022) [1].

    [1] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022."""

    def __init__(self, beta=0.2, upper_bound_estimation_percentage=100, wait_iterations=20, n_points_lcb=1000):
        self.beta = beta
        self.upper_bound_estimation_percentage = upper_bound_estimation_percentage
        self.wait_iterations = wait_iterations
        self.n_points_lcb = n_points_lcb
        self.incumbent = None
        self.incumbent_value = None
        self.incumbent_statistical_error = None

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool:
        """Checks if the optimization should be stopped after the given trial."""

        # todo: add the following functionality
        # - be able to query model about configs (via config selector; takes care of the encoding/decoding)

        model = smbo.intensifier.config_selector._model

        # update statistical error of incumbent if it has changed
        # todo get incumbent from intensifier instead
        for trial_info, trial_value in smbo.runhistory.items():
            if self.incumbent is None or trial_value.cost < self.incumbent_value:
                self.incumbent = trial_info
                self.incumbent_value = trial_value.cost
                incumbent_std = trial_value.additional_info['std_crossval']
                folds = trial_value.additional_info['folds']
                data_points = trial_value.additional_info['data_points']
                data_points_test = data_points/folds
                data_points_train = data_points - data_points_test
                factor_statistical_error = 1/folds + data_points_test/data_points_train
                self.incumbent_statistical_error = factor_statistical_error * incumbent_std**2

            print(trial_value.additional_info)

        # todo - select x% of the best configurations
        # compute regret
        # get model evaluations for all configurations
        encoding, costs = smbo.intensifier.config_selector._runhistory_encoder.transform()
        print("encoding", encoding)
        # todo maybe this transformation should be applied to the costs instead of the encoding
        transformed_encoding = smbo.intensifier.config_selector._runhistory_encoder.transform_response_values(encoding)
        print("transformed_encoding", transformed_encoding)

        # todo - dont rely on rf being used
        if model._rf is not None:
            # get pessimistic estimate of incumbent performance
            mean, var = model.predict_marginalized(transformed_encoding)
            # todo the predicted mean is lower than 0 - why?
            std = np.sqrt(var)
            ucbs = mean + np.sqrt(self.beta) * std
            print("mean", mean)
            print("std", std)
            print("ucbs", ucbs)
            min_ucb = np.min(ucbs)
            print("min_ucb", min_ucb)

            # get optimistic estimate of the best possible performance
            # get inspired by random search acquisition function maximizer
            min_lcb = 0

            # decide whether to stop
            regret = min_ucb - min_lcb

            # we are stopping once regret < incumbent statistical error (return false = do not continue)
            return regret >= self.incumbent_statistical_error

        else:
            print("no model built yet")

        return True