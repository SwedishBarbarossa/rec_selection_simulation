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


@njit(cache=True, fastmath=True, parallel=True)
def generate_noise(
    noise_distribution: Literal["normal", "uniform"], size: int
) -> npt.NDArray[np.float64]:
    if noise_distribution == "normal":
        return np.random.normal(0, 1, size)
    elif noise_distribution == "uniform":
        return np.random.uniform(-1, 1, size)

    raise ValueError('Invalid noise distribution type. Choose "normal" or "uniform".')


@njit(cache=True, fastmath=True, parallel=True)
def simulated_measurements(
    data: npt.NDArray[np.float64],
    noise_multiplier: float,
    noise_distribution: Literal["normal", "uniform"],
) -> npt.NDArray[np.float64]:
    noise = generate_noise(noise_distribution, data.size)
    return data + noise_multiplier * noise


@njit(cache=True, fastmath=True, parallel=True)
def extend_data(
    data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    MAX_EXTEND = 1_000_000

    if data.size > MAX_EXTEND:
        raise ValueError(
            "Data size cannot be larger than the maximum extended data size."
        )

    repeats = MAX_EXTEND // data.size
    extended_size = data.size * repeats
    extended_data = np.zeros(extended_size)

    for i in range(data.size):
        extended_data[i * repeats : (i + 1) * repeats] = data[i]

    return extended_data


# super hot
@njit(cache=True, fastmath=True, parallel=True)
def get_proper_noise_level(
    data: npt.NDArray[np.float64],
    desired_correlation: float,
    noise_distribution: Literal["normal", "uniform"],
) -> float:
    if desired_correlation > 1 or desired_correlation < 0:
        raise ValueError("Desired correlation must be between 0 and 1.")

    MAX_ITER = 100_000
    TOLERANCE = 1e-2  # Convergence tolerance
    noise_multiplier_upper = 4.0
    noise_multiplier_lower = 0.0

    # Generate noise
    noise = generate_noise(noise_distribution, data.size)

    # Initialize the minimum maximum
    for _ in range(MAX_ITER):
        noised_data = data + noise_multiplier_upper * noise
        current_noised_correlation = np.corrcoef(data, noised_data)[0, 1]
        if current_noised_correlation <= desired_correlation:
            break

        noise_multiplier_lower = noise_multiplier_upper
        noise_multiplier_upper = noise_multiplier_upper * 2

    # Binary search for the noise multiplier
    for _ in range(MAX_ITER):
        noise_multiplier = (noise_multiplier_lower + noise_multiplier_upper) / 2
        noised_data = data + noise_multiplier * noise
        current_noised_correlation = np.corrcoef(data, noised_data)[0, 1]
        difference = (
            abs(current_noised_correlation - desired_correlation) / desired_correlation
        )
        if difference <= TOLERANCE:
            return noise_multiplier

        if desired_correlation > current_noised_correlation:
            noise_multiplier_upper = noise_multiplier
        else:
            noise_multiplier_lower = noise_multiplier

    raise ValueError(
        "Could not find the desired noise level within the specified iterations."
    )


@njit(cache=True, fastmath=True, parallel=True)
def find_best_scores(
    applicant_data: npt.NDArray[np.float64],
    selection_size: int,
) -> npt.NDArray[np.int64]:
    """Returns the indices of the best applicants."""
    # best_indices
    best_indices = np.zeros(selection_size, dtype=np.int64) - 1
    worst_value = -np.inf
    worst_index = -1

    for i in range(applicant_data.size):
        # if the current score is better than the worst score
        if applicant_data[i] > worst_value:
            # insert the current score at the worst index
            best_indices[worst_index] = i

            # update the worst value and index
            worst_value = np.inf
            for j in range(selection_size):
                index = best_indices[j]
                if index == -1:
                    worst_value = -np.inf
                    worst_index = j
                    break

                if applicant_data[index] < worst_value:
                    worst_value = applicant_data[index]
                    worst_index = j

    # Check that all indices were set
    if np.any(best_indices == -1):
        raise ValueError("Some indices were not set.")

    return best_indices


@njit(cache=True, fastmath=True)
def selection_event(
    applicant_data: npt.NDArray[np.float64],
    noise_multiplier: float,
    noise_distribution: Literal["normal", "uniform"],
    selection_size: int,
) -> npt.NDArray[np.int64]:
    # Get simulated measurements
    noised_applicant_data = simulated_measurements(
        applicant_data, noise_multiplier, noise_distribution
    )

    # Return the highest scoring applicants
    return find_best_scores(noised_applicant_data, selection_size)


@njit(cache=True, fastmath=True, parallel=True)
def full_selection_process(
    applicant_data: npt.NDArray[np.float64],
    noise_multipliers: npt.NDArray[np.float64],
    selection_sizes: npt.NDArray[np.int64],
    noise_type: Literal["normal", "uniform"],
) -> int:
    """Returns the index of the selected applicant."""
    # Sanity check selection sizes.
    # Multipliers and sizes should have the same size.
    # They should be smaller than the number of applicants.
    # They should be in descending order.
    # The last selection size should be 1.

    if noise_multipliers.size != selection_sizes.size:
        raise ValueError(
            "Noise multipliers and selection sizes must have the same size."
        )
    if selection_sizes.max() > applicant_data.size:
        raise ValueError(
            "Selection sizes cannot be larger than the number of applicants."
        )
    if not np.all(np.diff(selection_sizes) <= 0):
        raise ValueError("Selection sizes must be in descending order.")
    if not selection_sizes[-1] == 1:
        raise ValueError("Last selection size must be 1.")

    # Get the selected applicants
    applicant_indices = np.arange(applicant_data.size)

    # Iterate over the selection steps
    for i in range(selection_sizes.size):
        noise_multiplier = noise_multipliers[i]
        selection_size = selection_sizes[i]
        # Get the selected applicants for the current step
        """ selected_indices = selection_event(
            applicant_data=applicant_data[applicant_indices],
            noise_multiplier=noise_multiplier,
            selection_size=selection_size,
            noise_distribution=noise_type,
        ) """

        noised_applicant_data = simulated_measurements(
            applicant_data[applicant_indices], noise_multiplier, noise_type
        )
        selected_indices = find_best_scores(noised_applicant_data, selection_size)

        applicant_indices = applicant_indices[selected_indices]

        # If there is one applicant left, return the index
        if applicant_indices.size <= 1:
            if applicant_indices.size == 0:
                raise ValueError("No applicant was selected.")

            return applicant_indices[0]

    raise ValueError("No applicant was selected.")


@njit(cache=True, fastmath=True, parallel=True)
def get_result_of_fifty_sims(
    applicant_data: npt.NDArray[np.float64],
    extended_data: npt.NDArray[np.float64],
    selection_predicates: npt.NDArray[np.float64],
    selection_sizes: npt.NDArray[np.int64],
    noise_type: Literal["normal", "uniform"],
) -> npt.NDArray[np.int64]:
    ITER = 50
    results = np.zeros(ITER, dtype=np.int64)

    noise_multipliers = np.zeros(selection_predicates.size, dtype=np.float64)
    for i, predicate in enumerate(selection_predicates):
        noise_multipliers[i] = get_proper_noise_level(
            extended_data, predicate, noise_type
        )

    for i in range(ITER):
        results[i] = full_selection_process(
            applicant_data=applicant_data,
            noise_multipliers=noise_multipliers,
            selection_sizes=selection_sizes,
            noise_type=noise_type,
        )

    # convert returned indices to density
    density_update = np.zeros(applicant_data.size, dtype=np.int64)
    for i in range(ITER):
        density_update[results[i]] += 1

    return density_update
