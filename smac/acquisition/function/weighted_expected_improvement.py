from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

logger = get_logger(__name__)

import grpc
from automl.llm_proxy import llm_proxy_pb2
from automl.llm_proxy import llm_proxy_pb2_grpc
import ast

from automl import BUDGET

def send_to_llm(content):
    channel = grpc.insecure_channel('localhost:50054')
    stub = llm_proxy_pb2_grpc.LLMProxyStub(channel)
    messages = [llm_proxy_pb2.ChatMessage(role=r, content=str(c)) for (r, c) in content]
    request = llm_proxy_pb2.ChatRequest(
        messages=messages,
        model=""
    )
    return stub.Chat(request).content

class WEI(AbstractAcquisitionFunction):
    def __init__(self, alpha: float = 0.5, xi: float = 0, log: bool = False, use_pure_PI: bool = False) -> None:
        super().__init__()
        self._xi: float = xi
        self._log: bool = log
        if self._log:
            raise NotImplementedError
        self._eta: float | None = None
        self._alpha = alpha
        self._use_pure_PI = use_pure_PI

        self.pi_term: np.ndarray | None = None
        self.pi_pure_term: np.ndarray | None = None
        self.pi_mod_term: np.ndarray | None = None
        self.ei_term: np.ndarray | None = None

        self._config_selector = None
        self._last_size = -1

    @property
    def name(self) -> str:  # noqa: D102
        return "Weighted Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
                "alpha": self._alpha,
            }
        )

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update acsquisition function attributes

        Parameters
        ----------
        eta : float
            Function value of current incumbent.
        xi : float, optional
            Exploration-exploitation trade-off parameter
        """
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]
        alpha = kwargs.get("alpha", None)
        if alpha is not None:
            self._alpha = alpha

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute EI acquisition value

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.

        Raises
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EI is < 0 for at least one sample (normal function value space).
        ValueError
            If EI is < 0 for at least one sample (log function value space).
        """
        assert self._model is not None
        assert self._xi is not None
        if self._use_pure_PI:
            assert self._alpha == 1., f"{self._alpha} != 0.5 with use pure PI. Any other combination, especially alpha=0.5 (EI) leads to wrong WEI."

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)  # TODO: can the variance become negative?
            s = np.sqrt(v)

            if len(self._config_selector._runhistory) > self._last_size:

                self._last_size = len(self._config_selector._runhistory)

                X_run, y_run, _ = self._config_selector._collect_data()

                primer = f"""
                You are a decision-making agent (the world's best) in a Bayesian Optimization loop for hyperparameter tuning. The objective is to minimize a scalar loss.
                You have two main responsibilities
                1. Assess the current state of the optimization 
                2. Use this knowledge to make an informed decision as to act exploratively or exploitatively.
                """

                prompt_evaluate_optimization = f"""
                The past evaluations were EVALS={X_run} and the corresponding costs were COSTS={y_run}.
                Currently, the optimization is {len(X_run)/ BUDGET} % complete.
                The current incumbent cost is {self._eta}.
                Assess the state of the optimization, i.e. whether it is making good progress or it is stagnating.
                """

                response_evaluate = send_to_llm([("system", primer), ("user", prompt_evaluate_optimization)])

                prompt_generate_percentage = f"""
                Based on this assessment, compute a float value in [0, 1] which will be used the guide the optimization in its current state
                with respect to exploration and exploitation.
                A value of 1 corresponds to full exploitation and a value of 0 corresponds to full exploration.
                Return only this float value and NOTHING ELSE
                """

                response_percentage = send_to_llm([("system", primer), ("user", prompt_evaluate_optimization), ("assistant", response_evaluate), ("user", prompt_generate_percentage)])

                self._alpha = ast.literal_eval(response_percentage)
                print(self._alpha)
            
            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                if self._use_pure_PI:
                    pi_term = norm.cdf(z)
                else:
                    pi_term = (self._eta - m - self._xi) * norm.cdf(z)
                ei_term = s * norm.pdf(z)
                self.pi_term = pi_term
                self.pi_pure_term = norm.cdf(z)
                self.pi_mod_term = (self._eta - m - self._xi) * norm.cdf(z)
                self.ei_term = ei_term
                return self._alpha * pi_term + (1 - self._alpha) * ei_term

            if np.any(s == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                s_copy = np.copy(s)
                s[s_copy == 0.0] = 1.0
                f = calculate_f()
                f[s_copy == 0.0] = 0.0
            else:
                f = calculate_f()

            return f
        else:
            raise NotImplementedError