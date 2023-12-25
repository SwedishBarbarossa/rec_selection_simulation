import pstats
from cProfile import Profile
from io import StringIO
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from tqdm import tqdm

from compiled import (
    extend_data,
    find_best_scores,
    full_selection_process,
    generate_applicant_data,
    get_proper_noise_level,
    selection_event,
    simulated_measurements,
)


def test_full_selection_process():
    # Test case 1: Normal distribution noise
    applicant_data = np.arange(100) / 50
    extended_data = np.arange(100_000) / 50_000
    selection_predicates = np.array([0.2, 0.4, 0.3], dtype=np.float64)
    noise_multipliers = np.zeros(selection_predicates.size, dtype=np.float64)
    for i in range(selection_predicates.size):
        noise_multipliers[i] = get_proper_noise_level(
            extended_data, selection_predicates[i], "normal"
        )
    selection_sizes = np.array([50, 10, 1])
    noise_type = "normal"

    result = full_selection_process(
        applicant_data=applicant_data,
        noise_multipliers=noise_multipliers,
        selection_sizes=selection_sizes,
        noise_type=noise_type,
    )

    # Assert that the result is an integer within the range of the applicant data
    assert isinstance(result, (int, np.int64))  # type: ignore
    result = int(result)
    assert result >= 0
    assert result < applicant_data.size

    # Test case 2: Uniform distribution noise
    noise_type = "uniform"

    result = full_selection_process(
        applicant_data=applicant_data,
        noise_multipliers=noise_multipliers,
        selection_sizes=selection_sizes,
        noise_type=noise_type,
    )

    # Assert that the result is an integer within the range of the applicant data
    assert isinstance(result, (int, np.int64))  # type: ignore
    result = int(result)
    assert result >= 0
    assert result < applicant_data.size

    # Test case 3: check that the result is not always the same
    results = np.zeros(100)
    for i in tqdm(range(100), desc="Stochastic results", leave=False):
        result = full_selection_process(
            applicant_data=applicant_data,
            noise_multipliers=noise_multipliers,
            selection_sizes=selection_sizes,
            noise_type=noise_type,
        )

        # Assert that the result is an integer within the range of the applicant data
        assert isinstance(result, (int, np.int64))  # type: ignore
        result = int(result)
        assert result >= 0
        assert result < applicant_data.size

        # Save the result
        results[i] = result

    # Assert that the results are not all the same
    if np.unique(results).size <= 1:
        print(np.unique(results))
    assert np.unique(results).size > 1

    if False:
        # Profile the function
        print("Profiling")
        predicates = np.random.rand(100, selection_sizes.size) ** 2
        # fit to 0.01 to 0.95
        predicates = 0.01 + 0.94 * predicates
        with Profile() as pr:
            for predicate in tqdm(predicates):
                full_selection_process(
                    applicant_data=applicant_data,
                    noise_multipliers=noise_multipliers,
                    selection_sizes=selection_sizes,
                    noise_type=noise_type,
                )

            # save the results
            pr.dump_stats("full_selection_process.prof")

            # Read the profiling data
            s = StringIO()
            ps = pstats.Stats("full_selection_process.prof", stream=s)

            # Filter and print the desired output
            ps.strip_dirs().sort_stats("cumulative").print_stats()
            output = s.getvalue().splitlines()

            # Filter out lines containing ".venv" or "<frozen" and display the top 100 lines
            filtered_output = [
                line
                for line in output
                if (".venv" not in line and "<frozen" not in line)
            ]
            top_100_filtered_output = filtered_output[:100]

            # Print the top 100 lines after filtering
            for line in top_100_filtered_output:
                print(line)

    # Add more test cases as needed


def test_selection_event():
    # Test case 1: selection_size = 40
    applicant_data = np.arange(100) / 50
    noise_multiplier = 1
    noise_distribution = "normal"
    selection_size = 40

    result = selection_event(
        applicant_data=applicant_data,
        noise_multiplier=noise_multiplier,
        noise_distribution=noise_distribution,
        selection_size=selection_size,
    )

    # Assert that the result contains the correct number of selected applicants
    assert result.size == selection_size

    # Test case 2: selection_size = 20
    selection_size = 20

    result = selection_event(
        applicant_data=applicant_data,
        noise_multiplier=noise_multiplier,
        noise_distribution=noise_distribution,
        selection_size=selection_size,
    )

    # Assert that the result contains the correct number of selected applicants
    assert result.size == selection_size

    # Test case 3: selection_size = 1
    selection_size = 1

    results = np.zeros(100, dtype=np.int64)
    for i in range(100):
        result = selection_event(
            applicant_data=applicant_data,
            noise_multiplier=noise_multiplier,
            noise_distribution=noise_distribution,
            selection_size=selection_size,
        )

        # Assert that the result contains the correct number of selected applicants
        assert result.size == selection_size

        # Save the result
        results[i] = result[0]

    # Assert that the results are not all the same
    if np.unique(results).size <= 1:
        print(np.unique(results))
    assert np.unique(results).size > 1

    # Add more test cases as needed


@njit(cache=True, fastmath=True, parallel=True)
def _get_median_correlation(
    data: npt.NDArray[np.float64],
    extended_data: npt.NDArray[np.float64],
    selection_predicate: float,
    noise_distribution: Literal["normal", "uniform"],
) -> float:
    ITER = 10_000
    noise_multiplier = get_proper_noise_level(
        extended_data, selection_predicate, noise_distribution
    )

    results = np.zeros((ITER, data.size), dtype=np.float64)
    for i in range(ITER):
        results[i] = simulated_measurements(
            data=data,
            noise_multiplier=noise_multiplier,
            noise_distribution=noise_distribution,
        )

    correlations = np.zeros(ITER, dtype=np.float64)
    for i in range(ITER):
        correlations[i] = np.corrcoef(data, results[i])[0, 1]

    return float(np.median(correlations))


def test_simulated_measurements():
    def _test_case(
        data: npt.NDArray[np.float64],
        extended_data: npt.NDArray[np.float64],
        noise_distribution: Literal["normal", "uniform"],
        selection_predicate: float,
    ):
        median_corr = _get_median_correlation(
            data=data,
            extended_data=extended_data,
            selection_predicate=selection_predicate,
            noise_distribution=noise_distribution,
        )

        # Assert that the result has a reasonable correlation with the data
        assert median_corr != 1
        assert median_corr != 0
        median_error = abs(median_corr - selection_predicate) / selection_predicate
        assert median_error < 0.025

    noise_distribution = "normal"
    with tqdm(total=12, leave=False) as pbar:
        for data_size in [200, 10_000]:
            for data_distribution in ["power_law", "normal"]:
                applicant_data = generate_applicant_data(
                    N=data_size, distribution=data_distribution  # type: ignore
                )
                extended_data = extend_data(applicant_data)
                for selection_predicate in [0.1, 0.3, 0.5]:
                    pbar.set_description(
                        f"data size: {data_size}, "
                        + f"data distribution: {data_distribution}, "
                        + f"selection predicate: {selection_predicate}"
                    )

                    _test_case(
                        data=applicant_data,
                        extended_data=extended_data,
                        noise_distribution=noise_distribution,
                        selection_predicate=selection_predicate,
                    )
                    pbar.update(1)


def test_find_best_scores():
    for _ in range(100):
        # Generate some unsorted data
        data = np.random.rand(1000)
        sorted = np.argsort(data)

        # Test case 1: k = 10
        k = 10

        result = find_best_scores(data, k)
        result.sort()

        # Assert that the result corresponds with what numpy would return
        numpy_res = sorted[-k:]
        numpy_res.sort()
        assert numpy_res.size == result.size
        assert np.all(numpy_res == result)
