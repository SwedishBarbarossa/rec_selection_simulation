from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def pareto_pdf(
    x: npt.NDArray[np.float64], alpha: float, xm: float
) -> npt.NDArray[np.float64]:
    return (alpha * xm**alpha) / (x ** (alpha + 1))


@njit(cache=True)
def normal_pdf(
    x: npt.NDArray[np.float64], mean: float, std_dev: float
) -> npt.NDArray[np.float64]:
    return (
        1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )


@njit(cache=True)
def generate_applicant_data(
    N: int, distribution: Literal["sigmoid", "power_law", "normal"]
) -> npt.NDArray[np.float64]:
    if distribution == "sigmoid":
        # Generating ordered data following a sigmoid distribution
        data = 1 / (1 + np.exp(-np.linspace(-5, 5, N)))
    elif distribution == "power_law":
        # Generating ordered data following a power law distribution
        xm = 5
        alpha = 10
        x_values = np.linspace(xm, 10, N)
        pdf_values = pareto_pdf(x_values, alpha, xm)
        data = pdf_values
    elif distribution == "normal":
        # Generating ordered data following a normal distribution
        mean = 5
        std_dev = 1.5
        x_values = np.linspace(0, 10, N)
        pdf_values = normal_pdf(x_values, mean, std_dev)
        data = pdf_values
    else:
        raise ValueError(
            'Invalid distribution type. Choose "sigmoid", "power_law", or "normal".'
        )

    return data


@njit(cache=True)
def simulated_measurements(
    data: npt.NDArray[np.float64],
    noised_correlation: float,
    noise_distribution: Literal["normal", "uniform"],
) -> npt.NDArray[np.float64]:
    MAX_ITER = 100_000
    tolerance = 1e-3  # Convergence tolerance
    noise_multiplier_upper = 1
    noise_multiplier_lower = 0.0
    noise_multiplier = 0.0
    current_noised_correlation = 0.0

    # Expand data. This is to avoid overfitting the noise to the data.
    repeats = 100_000 // data.size
    extended_size = data.size * repeats
    extended_data = np.zeros(extended_size, dtype=np.float64)
    for i in range(repeats):
        extended_data[i * data.size : (i + 1) * data.size] = data

    # Generate noise
    if noise_distribution == "normal":
        noise = np.random.normal(0, 1, extended_data.size)
    elif noise_distribution == "uniform":
        noise = np.random.uniform(-1, 1, extended_data.size)
    else:
        raise ValueError(
            'Invalid noise distribution type. Choose "normal" or "uniform".'
        )

    # Initialize the minimum maximum
    for _ in range(MAX_ITER):
        noised_data = extended_data + noise_multiplier_upper * noise
        current_noised_correlation = np.corrcoef(extended_data, noised_data)[0, 1]

        if current_noised_correlation < noised_correlation:
            break

        noise_multiplier_lower = noise_multiplier_upper
        noise_multiplier_upper *= 2

    # Binary search for the noise multiplier
    for _ in range(MAX_ITER):
        noise_multiplier = (noise_multiplier_lower + noise_multiplier_upper) / 2
        noised_data = extended_data + noise_multiplier * noise
        current_noised_correlation = np.corrcoef(extended_data, noised_data)[0, 1]

        if abs(current_noised_correlation - noised_correlation) <= tolerance:
            return noised_data[: data.size]

        if current_noised_correlation < noised_correlation:
            noise_multiplier_upper = noise_multiplier
        else:
            noise_multiplier_lower = noise_multiplier

    # Debugging output
    print(
        data.size,
        noise_multiplier_lower,
        noise_multiplier_upper,
        noise_multiplier,
        current_noised_correlation,
        noised_correlation,
    )
    raise ValueError(
        "Could not find the desired noise level within the specified iterations."
    )


@njit(cache=True)
def selection_event(
    applicant_data: npt.NDArray[np.float64],
    selection_predicate: float,
    noise_distribution: Literal["normal", "uniform"],
    selection_size: int,
    final_selection=False,
) -> npt.NDArray[np.float64]:
    # Get simulated measurements
    noised_applicant_data = simulated_measurements(
        applicant_data, selection_predicate, noise_distribution
    )

    if final_selection:
        applicant_idx = np.argmax(noised_applicant_data)
        result = np.full(applicant_data.size, np.nan, dtype=np.float64)
        result[applicant_idx] = applicant_data[applicant_idx]
        return result

    # Get the selection threshold
    desired_quantile = 1 - selection_size / applicant_data.size
    selection_threshold = np.quantile(noised_applicant_data, desired_quantile)

    # Get the selected applicants
    selected_applicants = noised_applicant_data >= selection_threshold

    # Set the removed applicants to nan
    result = applicant_data.copy()
    result[~selected_applicants] = np.nan

    return result


@njit(cache=True)
def full_selection_process(
    applicant_data: npt.NDArray[np.float64],
    selection_predicates: npt.NDArray[np.float64],
    selection_sizes: npt.NDArray[np.int64],
    noise_type: Literal["normal", "uniform"],
) -> int:
    """Returns the index of the selected applicant."""
    # Sanity check selection sizes.
    # They should be smaller than the number of applicants.
    # They should be in descending order.
    if selection_sizes.max() > applicant_data.size:
        raise ValueError(
            "Selection sizes cannot be larger than the number of applicants."
        )
    if not np.all(np.diff(selection_sizes) <= 0):
        raise ValueError("Selection sizes must be in descending order.")

    # Get the selected applicants
    selected_applicants = np.ones(len(applicant_data)).astype(np.bool_)

    # Iterate over the selection steps
    final_step = selection_predicates.size - 1
    for i in range(selection_predicates.size):
        selection_predicate = selection_predicates[i]
        selection_size = selection_sizes[i]
        # Get the selected applicants for the current step
        final = i == final_step
        current_selected_applicants = selection_event(
            applicant_data=applicant_data[selected_applicants],
            selection_predicate=selection_predicate,
            selection_size=selection_size,
            final_selection=final,
            noise_distribution=noise_type,
        )

        # Update the selected applicants
        selected_applicants[selected_applicants] = selected_applicants[
            selected_applicants
        ] & ~np.isnan(current_selected_applicants)

        # If there is one or zero applicants left, stop the selection process
        if selected_applicants.astype(np.int8).sum() == 1:
            break

    # Return the index of the selected applicant
    return int(selected_applicants.astype(np.int8).argmax())


@njit(cache=True)
def run_simulation(
    *,
    applicant_data: npt.NDArray[np.float64],
    selection_predicates: npt.NDArray[np.float64],
    selection_sizes: npt.NDArray[np.int64],
    n_simulations: int,
    noise_type: Literal["normal", "uniform"],
) -> npt.NDArray[np.int64]:
    density = np.zeros(applicant_data.size, dtype=np.int64)

    for _ in range(n_simulations):
        result = full_selection_process(
            applicant_data=applicant_data,
            selection_predicates=selection_predicates,
            selection_sizes=selection_sizes,
            noise_type=noise_type,
        )

        # Update density
        density[result] += 1

    return density
