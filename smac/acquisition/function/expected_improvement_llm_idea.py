from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

import grpc
from automl.llm_proxy import llm_proxy_pb2
from automl.llm_proxy import llm_proxy_pb2_grpc
import ast

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

def send_to_llm(content):
    channel = grpc.insecure_channel('localhost:50054')
    stub = llm_proxy_pb2_grpc.LLMProxyStub(channel)
    messages = [llm_proxy_pb2.ChatMessage(role=r, content=str(c)) for (r, c) in content]
    request = llm_proxy_pb2.ChatRequest(
        messages=messages,
        model=""
    )
    return stub.Chat(request).content

class EI(AbstractAcquisitionFunction):
    r"""The Expected Improvement (EI) criterion is used to decide where to evaluate a function f(x) next. The goal is to
    balance exploration and exploitation. Expected Improvement (with or without function values in log space)
    acquisition function

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.

    Reference for EI: Jones, D.R. and Schonlau, M. and Welch, W.J. (1998). Efficient Global Optimization of Expensive
    Black-Box Functions. Journal of Global Optimization 13, 455–492

    Reference for logEI: Hutter, F. and Hoos, H. and Leyton-Brown, K. and Murphy, K. (2009). An experimental
    investigation of model-based parameter optimisation: SPO and beyond. In: Conference on Genetic and
    Evolutionary Computation

    The logEI implemententation is based on the derivation of the orginal equation by:
    Watanabe, S. (2024). Derivation of Closed Form of Expected Improvement for Gaussian Process Trained on
    Log-Transformed Objective. https://arxiv.org/abs/2411.18095

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.
    log : bool, defaults to False
        Whether the function values are in log-space.


    Attributes
    ----------
    _xi : float
        Exploration-exloitation trade-off parameter.
    _log: bool
        Function values in log-space or not.
    _eta : float
        Current incumbent function value (best value observed so far).

    """

    def __init__(
        self,
        xi: float = 0.0,
        log: bool = False,
    ) -> None:
        super(EI, self).__init__()

        self._xi: float = xi
        self._log: bool = log
        self._eta: float | None = None

        self._prompt_history = []
        self._result_history = []

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
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

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self._model.predict_marginalized(X)
        s = np.sqrt(v)

        primer = f"""
        You are a mathematical acquisition function in a Bayesian optimization loop.
        As such, you only return values and do not produce language or python code.

        Your task is to compute a single acquisition value for each candidate configuration. These values guide the optimizer in selecting the next configuration to evaluate.

        The goal is to balance **exploration** (choosing uncertain regions) and **exploitation** (favoring regions with high predicted performance). The optimizer selects the configuration with the **maximum** acquisition value.

        ### Input:
        - A list of configurations: [[x11, x12, x13], [x21, x22, x23], ...]
        - Corresponding model predictions from a random forest surrogate:
        - `MEANS`: predicted objective values
        - `VARIANCES`: estimated uncertainties

        Each configuration has a corresponding mean and variance by index (i.e., MEANS[i] and VARIANCES[i] belong to CONFIGURATIONS[i]).

        ### Output:
        Return a list of acquisition values: [[v1], [v2], ...], where vi corresponds to CONFIGURATIONS[i].

        **Important:** Remember, Return ONLY the list of comma-separated numerical values — no extra text, explanations or python code.
        """


        prompt = f"""
        CONFIGURATIONS={X},
        MEANS={m},
        VARIANCES={v}
        """

        self._prompt_history.append(prompt)

        content = [('system', primer)]

        for i in range(len(self._prompt_history)):
            if i < len(self._prompt_history):
                content.append(('user', self._prompt_history[i]))
            if i < len(self._result_history):
                content.append(('system', self._result_history[i]))

        result1 = send_to_llm(content)

        #print(result1)

        self._result_history.append(result1)

        prompt2 = f"""
        Fine, and now again ONLY return the final result of acquisition values in the format
        [[v1], [v2], ...]
        """

        self._prompt_history.append(prompt2)

        content = [('system', primer)]

        for i in range(len(self._prompt_history)):
            if i < len(self._prompt_history):
                content.append(('user', self._prompt_history[i]))
            if i < len(self._result_history):
                content.append(('system', self._result_history[i]))

        result2 = send_to_llm(content)

        print(result2)

        self._result_history.append(result2)

        f = np.array(ast.literal_eval(result2))

        return f


class EIPS(EI):
    r"""Expected Improvement per Second acquisition function

    :math:`EI(X) := \frac{\mathbb{E}\left[\max\{0,f(\mathbf{X^+})-f_{t+1}(\mathbf{X})-\xi\right]\}]}{np.log(r(x))}`,
    with :math:`f(X^+)` as the best location and :math:`r(x)` as runtime.

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the acquisition function.
    """

    def __init__(self, xi: float = 0.0) -> None:
        super(EIPS, self).__init__(xi=xi)

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement per Second"

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute EI per second acquisition value

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
            If the mean has the wrong shape, should have shape (-1, 2).
        ValueError
            If the variance has the wrong shape, should have shape (-1, 2).
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EIPS is < 0 for at least one sample.
        """
        assert self._model is not None
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self._model.predict_marginalized(X)
        if m.shape[1] != 2:
            raise ValueError(f"m has wrong shape: {m.shape} != (-1, 2)")
        if v.shape[1] != 2:
            raise ValueError(f"v has wrong shape: {v.shape} != (-1, 2)")

        m_cost = m[:, 0]
        v_cost = v[:, 0]

        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        def calculate_f() -> np.ndarray:
            z = (self._eta - m_cost - self._xi) / s
            f = (self._eta - m_cost - self._xi) * norm.cdf(z) + s * norm.pdf(z)
            f = f / m_runtime

            return f

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

        if (f < 0).any():
            raise ValueError("Expected Improvement per Second is smaller than 0 " "for at least one sample.")

        return f.reshape((-1, 1))
