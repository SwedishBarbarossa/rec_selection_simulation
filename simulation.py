from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from compiled import run_simulation


def combine_selection_predicates(predicates: list[float]) -> float:
    arr = np.array(predicates)
    arr = arr**2
    return np.sqrt(np.sum(arr))


def convert_selection_steps(
    selection_steps: list[tuple[float, int]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    selection_predicates = np.zeros(len(selection_steps), dtype=np.float64)
    selection_sizes = np.zeros(len(selection_steps), dtype=np.int64)
    for i, (predicate, size) in enumerate(selection_steps):
        selection_predicates[i] = predicate
        selection_sizes[i] = size

    # combine successive selection predicates
    if selection_predicates.size > 1:
        for i in range(1, selection_predicates.size):
            selection_predicates[i] = combine_selection_predicates(
                selection_predicates[i - 1 : i + 1].tolist()
            )

    # remove steps which don't narrow down the selection
    for i in range(selection_sizes.size - 1):
        if selection_sizes[i] == selection_sizes[i + 1]:
            selection_sizes[i] = -1

    selection_predicates = selection_predicates[selection_sizes != -1]
    selection_sizes = selection_sizes[selection_sizes != -1]

    return selection_predicates, selection_sizes


def get_simulation_result(
    *,
    applicant_data: npt.NDArray[np.float64],
    selection_steps: list[tuple[float, int]],
    n_simulations: int,
    noise_type: Literal["normal", "uniform"],
) -> pd.DataFrame:
    # convert selection steps to numpy arrays for numba
    selection_predicates, selection_sizes = convert_selection_steps(selection_steps)

    density = run_simulation(
        applicant_data=applicant_data,
        selection_predicates=selection_predicates,
        selection_sizes=selection_sizes,
        n_simulations=n_simulations,
        noise_type=noise_type,
    )

    # Create dataframe with applicant data and density
    return pd.DataFrame(
        {
            "applicant_ID": np.arange(applicant_data.size),
            "applicant_score": applicant_data,
            "density": density,
        }
    )
