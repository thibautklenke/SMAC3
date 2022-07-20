from __future__ import annotations

from typing import Any, List, Optional

import logging
from collections import OrderedDict

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import ForbiddenValueError, deactivate_inactive_hyperparameters

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2019, AutoML"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class InitialDesign:
    """Base class for initial design strategies that evaluates multiple configurations.

    Parameters
    ----------
    configspace: ConfigurationSpace
        configuration space object
    rng: np.random.RandomState
        Random state
    n_runs: int
        Number of iterations allowed for the target algorithm
    configs: Optional[List[Configuration]]
        List of initial configurations. Disables the arguments ``n_configs_per_hyperparameter`` if given.
        Either this, or ``n_configs_per_hyperparameter`` or ``init_budget`` must be provided.
    n_configs_per_hyperparameter: int
        how many configurations will be used at most in the initial design (X*D). Either
        this, or ``init_budget`` or ``configs`` must be provided. Disables the argument
        ``n_configs_per_hyperparameter`` if given.
    max_config_ratio: float
        use at most X*budget in the initial design. Not active if a time limit is given.
    init_budget : int, optional
        Maximal initial budget (disables the arguments ``n_configs_per_hyperparameter`` and ``configs``
        if both are given). Either this, or ``n_configs_per_hyperparameter`` or ``configs`` must be
        provided.

    Attributes
    ----------
    configspace : ConfigurationSpace
    configs : List[Configuration]
        List of configurations to be evaluated
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_runs: int,
        configs: Optional[List[Configuration]] = None,
        n_configs_per_hyperparameter: Optional[int] = 10,
        max_config_ratio: float = 0.25,
        init_budget: Optional[int] = None,
        seed: int = 0,
    ):
        # TODO: Change init_budget to n_configs?
        self.configspace = configspace
        self.rng = np.random.RandomState(seed)
        self.configs = configs

        logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        n_params = len(self.configspace.get_hyperparameters())
        if init_budget is not None:
            self.init_budget = init_budget
            if n_configs_per_hyperparameter is not None:
                logger.debug(
                    "Ignoring argument `n_configs_per_hyperparameter` (value %d).",
                    n_configs_per_hyperparameter,
                )
        elif configs is not None:
            self.init_budget = len(configs)
        elif n_configs_per_hyperparameter is not None:
            self.init_budget = int(max(1, min(n_configs_per_hyperparameter * n_params, (max_config_ratio * n_runs))))
        else:
            raise ValueError(
                "Need to provide either argument `init_budget`, `configs` or "
                "`n_configs_per_hyperparameter`, but provided none of them."
            )
        if self.init_budget > n_runs:
            raise ValueError("Initial budget %d cannot be higher than the run limit %d." % (self.init_budget, n_runs))
        logger.info(f"Running initial design for {self.init_budget} configurations.")

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"init_budget": self.init_budget}

    def select_configurations(self) -> List[Configuration]:
        """Selects the initial configurations."""
        if self.init_budget == 0:
            return []
        if self.configs is None:
            self.configs = self._select_configurations()

        for config in self.configs:
            if config.origin is None:
                config.origin = "Initial design"

        # add this incumbent right away to have an entry to time point 0
        # self.traj_logger.add_entry(train_perf=2**31, incumbent_id=1, incumbent=self.configs[0])

        # removing duplicates
        # (Reference: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists)
        self.configs = list(OrderedDict.fromkeys(self.configs))
        return self.configs

    def _select_configurations(self) -> List[Configuration]:
        raise NotImplementedError

    def _transform_continuous_designs(
        self, design: np.ndarray, origin: str, configspace: ConfigurationSpace
    ) -> List[Configuration]:

        params = configspace.get_hyperparameters()
        for idx, param in enumerate(params):
            if isinstance(param, NumericalHyperparameter):
                continue
            elif isinstance(param, Constant):
                # add a vector with zeros
                design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
                design_[:, :idx] = design[:, :idx]
                design_[:, idx + 1 :] = design[:, idx:]
                design = design_
            elif isinstance(param, CategoricalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
            elif isinstance(param, OrdinalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
            else:
                raise ValueError("Hyperparameter not supported in LHD.")

        logger.debug("Initial Design")
        configs = []
        for vector in design:
            try:
                conf = deactivate_inactive_hyperparameters(
                    configuration=None, configuration_space=configspace, vector=vector
                )
            except ForbiddenValueError:
                continue
            conf.origin = origin
            configs.append(conf)
            logger.debug(conf)

        logger.debug("Size of initial design: %d" % (len(configs)))

        return configs
