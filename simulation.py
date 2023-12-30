from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from compiled import (
    extend_data,
    full_selection_process,
    get_proper_noise_level,
    get_result_of_fifty_sims,
)


def combine_selection_predicates(correlation_list: list[float]) -> float:
    # Convert the list of correlations to a NumPy array
    correlations = np.array(correlation_list, dtype=np.float64)

    # Fisher's z-transformation for each correlation
    z_scores: npt.NDArray[np.float64] = (
        np.log((1 + correlations) / (1 - correlations)) * 0.5
    )

    # Summing the z-scores
    summed_z_score: float = z_scores.sum()

    # Inverse Fisher's z-transformation for the summed z-score
    doubled_summed_z_score = summed_z_score * 2
    exp_z_score: np.float64 = np.exp(doubled_summed_z_score)
    inverse_z_sum = (exp_z_score - 1) / (exp_z_score + 1)
    if inverse_z_sum > 1:
        raise ValueError(
            "Inverse Fisher's z-transformation resulted in correlation > 1"
        )

    return inverse_z_sum


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
        for i in range(selection_predicates.size - 1, -1, -1):
            selection_predicates[i] = combine_selection_predicates(
                selection_predicates[0 : i + 1].tolist()
            )

    # raise error for non-decreasing selection sizes
    for i in range(selection_sizes.size - 1):
        if selection_sizes[i] <= selection_sizes[i + 1]:
            raise ValueError("Selection sizes must be decreasing.")

    # raise error if last selection size is not 1
    if selection_sizes[-1] != 1:
        raise ValueError("Last selection size must be 1.")

    selection_predicates = selection_predicates[selection_sizes != -1]
    selection_sizes = selection_sizes[selection_sizes != -1]

    return selection_predicates, selection_sizes


def run_simulation(
    *,
    applicant_data: npt.NDArray[np.float64],
    selection_steps: list[tuple[float, int]],
    n_simulations: int,
    noise_type: Literal["normal", "uniform"],
    sim_title: str,
) -> pd.DataFrame:
    # convert selection steps to numpy arrays for numba
    selection_predicates, selection_sizes = convert_selection_steps(selection_steps)

    density = np.zeros(applicant_data.size, dtype=np.int64)

    # Expand data. This is to avoid overfitting the multiplier to the noise.
    extended_data = extend_data(applicant_data)

    with tqdm(total=n_simulations, desc=sim_title, leave=False) as pbar:
        SIM_CONST = 50
        for _ in range(n_simulations // SIM_CONST):
            density += get_result_of_fifty_sims(
                applicant_data=applicant_data,
                extended_data=extended_data,
                selection_predicates=selection_predicates,
                selection_sizes=selection_sizes,
                noise_type=noise_type,
            )
            pbar.update(SIM_CONST)

        noise_multipliers = np.zeros(selection_predicates.size, dtype=np.float64)
        for i, predicate in enumerate(selection_predicates):
            noise_multipliers[i] = get_proper_noise_level(
                extended_data, predicate, noise_type
            )

        for _ in range(n_simulations % SIM_CONST):
            result = full_selection_process(
                applicant_data=applicant_data,
                noise_multipliers=noise_multipliers,
                selection_sizes=selection_sizes,
                noise_type=noise_type,
            )

            # Update density
            density[result] += 1
            pbar.update(1)

    # Create dataframe with applicant data and density
    return pd.DataFrame(
        {
            "applicant_ID": np.arange(applicant_data.size),
            "applicant_score": applicant_data,
            "density": density,
        }
    )
