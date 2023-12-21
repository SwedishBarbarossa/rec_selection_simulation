import numpy as np

from compiled import full_selection_process, selection_event, simulated_measurements


def test_full_selection_process():
    # Test case 1: Normal distribution noise
    applicant_data = np.arange(100) / 50
    selection_predicates = np.array([0.2, 0.4, 0.3], dtype=np.float64)
    selection_sizes = np.array([50, 10, 5])
    noise_type = "normal"

    result = full_selection_process(
        applicant_data=applicant_data,
        selection_predicates=selection_predicates,
        selection_sizes=selection_sizes,
        noise_type=noise_type,
    )

    # Assert that the result contains only the selected applicants
    assert (~np.isnan(result)).sum() == 1

    # Test case 2: Uniform distribution noise
    noise_type = "uniform"

    result = full_selection_process(
        applicant_data=applicant_data,
        selection_predicates=selection_predicates,
        selection_sizes=selection_sizes,
        noise_type=noise_type,
    )

    # Assert that the result contains only the selected applicants
    assert (~np.isnan(result)).sum() == 1

    # Add more test cases as needed


def test_selection_event():
    # Test case 1: final selection = False
    applicant_data = np.arange(100) / 50
    selection_predicate = 0.05
    noise_distribution = "normal"
    selection_size = 40
    final_selection = False

    result = selection_event(
        applicant_data=applicant_data,
        selection_predicate=selection_predicate,
        noise_distribution=noise_distribution,
        selection_size=selection_size,
        final_selection=final_selection,
    )

    # Assert that the result contains the correct number of selected applicants
    assert (~np.isnan(result)).sum() == selection_size

    # Test case 2: final selection = False, different selection size
    selection_size = 20

    result = selection_event(
        applicant_data=applicant_data,
        selection_predicate=selection_predicate,
        noise_distribution=noise_distribution,
        selection_size=selection_size,
        final_selection=final_selection,
    )

    # Assert that the result contains the correct number of selected applicants
    assert (~np.isnan(result)).sum() == selection_size

    # Test case 3: final selection = True
    final_selection = True

    result = selection_event(
        applicant_data=applicant_data,
        selection_predicate=selection_predicate,
        noise_distribution=noise_distribution,
        selection_size=selection_size,
        final_selection=final_selection,
    )

    # Assert that the result contains only the selected applicant
    assert (~np.isnan(result)).sum() == 1

    # Add more test cases as needed


def test_simulated_measurements():
    # Test case 1: normal distribution
    data = np.random.normal(size=100_000)
    selection_predicate = 0.1
    noise_distribution = "normal"

    result = simulated_measurements(
        data=data,
        noised_correlation=selection_predicate,
        noise_distribution=noise_distribution,
    )

    # Assert that the result has the same shape as the data
    assert result.shape == data.shape

    # Assert that the result has a reasonable correlation with the data
    assert np.corrcoef(data, result)[0, 1] != 1
    assert np.corrcoef(data, result)[0, 1] >= selection_predicate * 0.99
    assert np.corrcoef(data, result)[0, 1] <= selection_predicate * 1.01

    # Test case 2: uniform distribution
    noise_distribution = "uniform"

    result = simulated_measurements(
        data=data,
        noised_correlation=selection_predicate,
        noise_distribution=noise_distribution,
    )

    # Assert that the result has the same shape as the data
    assert result.shape == data.shape

    # Assert that the result has a reasonable correlation with the data
    assert np.corrcoef(data, result)[0, 1] != 1
    assert np.corrcoef(data, result)[0, 1] >= selection_predicate * 0.99
    assert np.corrcoef(data, result)[0, 1] <= selection_predicate * 1.01

    # Test case 3: sorted data
    data = np.arange(100_000) / 50_000
    noise_distribution = "normal"

    result = simulated_measurements(
        data=data,
        noised_correlation=selection_predicate,
        noise_distribution=noise_distribution,
    )

    # Assert that the result has the same shape as the data
    assert result.shape == data.shape

    # Assert that the result has a reasonable correlation with the data
    assert np.corrcoef(data, result)[0, 1] != 1
    assert np.corrcoef(data, result)[0, 1] >= selection_predicate * 0.99
    assert np.corrcoef(data, result)[0, 1] <= selection_predicate * 1.01

    # Test case 4: small data
    data = np.random.normal(size=20_000)

    result = simulated_measurements(
        data=data,
        noised_correlation=selection_predicate,
        noise_distribution=noise_distribution,
    )

    # Assert that the result has the same shape as the data
    assert result.shape == data.shape

    # Assert that the result has a reasonable correlation with the data
    assert np.corrcoef(data, result)[0, 1] != 1
    assert np.corrcoef(data, result)[0, 1] >= selection_predicate * 0.8
    assert np.corrcoef(data, result)[0, 1] <= selection_predicate * 1.2

    # Add more test cases as needed
