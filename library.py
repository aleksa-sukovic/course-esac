import control
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import control.matlab as mtl
import pandas as pd

from typing import List, Tuple
from math import ceil

def is_stable(system: control.StateSpace) -> bool:
    if system.isdtime():
        # TODO: Handle stability for Discrete-Time System.
        return False
    else:
        all_eigenvalues = np.real(np.linalg.eigvals(system.A).flatten())
        zero_eigenvalues = all_eigenvalues[np.abs(all_eigenvalues) <= 10e-5]
        condition_multiplicity = np.unique(zero_eigenvalues).size == zero_eigenvalues.size
        condition_not_positive = all([x <= 0 for x in np.nditer(all_eigenvalues)])
        return condition_multiplicity and condition_not_positive

def is_controllable(system: control.StateSpace) -> bool:
    matrix = control.ctrb(system.A, system.B)
    matrix_rank = np.linalg.matrix_rank(matrix)
    assert matrix.shape[0] == matrix.shape[1] # We restrict to square matrices, for now.
    return matrix.shape[0] == matrix_rank

def is_observable(system: control.StateSpace) -> bool:
    matrix = control.obsv(system.A, system.C)
    matrix_rank = np.linalg.matrix_rank(matrix)
    assert matrix.shape[0] == matrix.shape[1] # We restrict to square matrices, for now.
    return matrix.shape[0] == matrix_rank

def step(system: control.StateSpace, plot: bool = True) -> control.TimeResponseData:
    result = control.step_response(system)
    (time, output) = result

    if plot:
        data_out = pd.DataFrame({"Time": time, "Output": output})
        data_in = pd.DataFrame({"Time": time, "Input": np.full(time.shape, fill_value=1.0)})

        sns.set_theme()
        sns.set_context("paper")
        sns.lineplot(x="Time", y="Output", data=data_out, color="#2f1e3d", linewidth=2.0, legend=False)
        sns.lineplot(x="Time", y="Input", data=data_in, color="#b0678e", linewidth=2.0, linestyle="dashed", alpha=0.7, legend=False)
        plt.legend(labels=["System Output", "System Input"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Step Response", loc="center", fontdict={"fontsize": 13.0}, pad=13.0)

    return result

def lsim(system: control.StateSpace, plot: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = mtl.lsim(system, **kwargs)
    y_out, time, _ = result

    if plot:
        data_out = pd.DataFrame({"Time": time, "Output": y_out})
        data_in = pd.DataFrame({"Time": time, "Input": kwargs.get("U")})

        sns.set_theme()
        sns.set_context("paper")
        sns.lineplot(x="Time", y="Output", data=data_out, color="#2f1e3d", linewidth=2.0, legend=False)
        sns.lineplot(x="Time", y="Input", data=data_in, color="#dfa1ab", linewidth=2.0, linestyle="dashed", alpha=0.7, legend=False)
        plt.legend(labels=["System Output", "System Input"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Variable Input Response", loc="center", fontdict={"fontsize": 13.0}, pad=13.0)

    return result

def simulate_inputs(targets: List[int], duration: int=5, step: float=0.05) -> Tuple[np.ndarray, np.ndarray]:
    steps = [np.full(ceil(duration / step), fill_value=t) for t in targets]
    input = np.stack(steps, axis=0).flatten()
    time = np.arange(start=0, stop=len(targets) * duration, step=step)
    return (input, time)
