import os
from enum import Enum
from multiprocessing import Pool
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from compiled import generate_applicant_data, simulated_measurements
from simulation import combine_selection_predicates, get_simulation_result


def plot_results(dfs: list[tuple[pd.DataFrame, str]], plot_title: str) -> None:
    # Set up dataframes for comparison
    processed_dfs = []
    for df, selection_process in dfs:
        df_copy = df.copy()
        df_copy["selection_process"] = selection_process
        processed_dfs.append(
            df_copy.loc[np.repeat(df_copy.index.values, df_copy["density"])]
        )

    res_df = pd.concat(processed_dfs)

    # Create seaborn plot
    g = sns.boxplot(
        data=res_df,
        x="selection_process",
        y="applicant_score",
        hue="selection_process",
        legend=False,
        palette="flare",
    )
    g.set_ylabel("Applicant score")
    g.set_xlabel("")
    # rotate x-axis labels
    for item in g.get_xticklabels():
        item.set_rotation(7)
    g.set_title(plot_title)

    # set size
    plt.gcf().set_size_inches(12, 8)

    # save plot
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = plot_title.replace(" ", "_").lower()
    plt.savefig(f"{current_dir}/plots/{file_name}.png")

    # clear plot
    plt.clf()


def plot_score_distribution(
    applicant_data: np.ndarray,
    noise_type: Literal["normal", "uniform"],
    name_suffix: str = "",
) -> None:
    ### Plot the base score distribution ###
    # Create seaborn plot
    g = sns.scatterplot(data=applicant_data, color="blue")
    g.set_ylabel("Real applicant score")
    g.set_xlabel("Applicant ID")
    g.set_title("Applicant score distribution")

    # set size
    plt.gcf().set_size_inches(12, 8)

    # save plot
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = "applicant_score_distribution"
    plt.savefig(f"{current_dir}/plots/{file_name}{name_suffix}.png")

    # clear plot
    plt.clf()

    ### Plot the noised score distribution ###
    # Get simulated measurements
    noised_applicant_data = simulated_measurements(applicant_data, 0.5, noise_type)

    # Create seaborn plot with regression line
    g = sns.regplot(
        x=np.arange(applicant_data.size),
        y=noised_applicant_data,
        color="blue",
        ci=None,
        order=2,
        line_kws=dict(color="r"),
        marker="x",
        truncate=True,
        scatter=True,
    )
    g.set_ylabel("Simulated applicant measurement")
    g.set_xlabel("Applicant ID")
    g.set_title(f"Noised applicant score distribution ({noise_type})")

    # set size
    plt.gcf().set_size_inches(12, 8)

    # save plot
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = f"noised_applicant_score_distribution_{noise_type}"
    plt.savefig(f"{current_dir}/plots/{file_name}{name_suffix}.png")

    # clear plot
    plt.clf()


class ApplicantScoreDistributionTypes(Enum):
    sigmoid = "sigmoid"
    power_law = "power_law"
    normal = "normal"


class NoiseDistributionTypes(Enum):
    normal = "normal"
    uniform = "uniform"


class SelectionProcedure:
    """Selection procedures and their correlations with job performance."""

    # unspecified sources are from 10.1037/apl0000994

    # Tests
    Big_5_contextualized = 0.48  # personality test
    Job_knowledge_tests = 0.4
    Work_sample_tests = 0.33
    Cognitive_ability_tests = 0.31
    Integrity_tests = 0.31
    Personality_based_EI = 0.3  # Emotional intelligence
    Big_5_overall = 0.27  # personality test
    SJT_knowledge = 0.26  # situational judgment test
    SJT_behavioral_tendency = 0.26  # situational judgment test
    Ability_based_EI = 0.22  # Emotional intelligence

    # Interviews
    Employment_interviews_structured = 0.42
    Employment_interviews_unstructured = 0.19

    # Biodata
    Empirically_keyed_biodata = 0.38  # questionnaires
    Interests = 0.24
    Rationally_keyed_biodata = 0.22  # questionnaires
    Job_experience_years = 0.09

    # Other
    Assessment_centers = 0.29
    Commute_distance = 0.103  # from 10.1080/10803548.2021.2010970


if __name__ == "__main__":
    ### Configuration ###
    N = 200  # Number of applicants at the start of the selection process
    SIMULATIONS = 10_000  # Number of simulations to run
    selected_distribution = ApplicantScoreDistributionTypes.power_law.value
    noise_type = NoiseDistributionTypes.normal.value
    load_data = True  # Set to False to regenerate data
    #####################

    ### Generate applicant data ###
    # Generate default applicant data based on the selected distribution
    applicant_data = generate_applicant_data(N, selected_distribution)
    applicant_data_large = generate_applicant_data(10_000, selected_distribution)

    # Create plot folder if it doesn't exist
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(f"{current_dir}/plots"):
        os.makedirs(f"{current_dir}/plots")

    # Plot the score distributions
    plot_score_distribution(applicant_data, noise_type)
    plot_score_distribution(applicant_data_large, noise_type, "_large")

    #### Screenings ####
    CV_screening = SelectionProcedure.Job_experience_years
    test_screening = combine_selection_predicates(
        [
            SelectionProcedure.Big_5_overall,
            SelectionProcedure.Cognitive_ability_tests,
        ]
    )
    serious_CV_screening = combine_selection_predicates(
        [
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Interests,
        ]
    )
    ideal_screening = combine_selection_predicates(
        [
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Cognitive_ability_tests,
            SelectionProcedure.Big_5_overall,
            SelectionProcedure.Commute_distance,
        ]
    )

    ### Selection steps ###
    # format: (selection_predicate, remaining_applicants)
    StepsType = list[tuple[float, int]]
    standard_steps: StepsType = [
        (CV_screening, 20),
        (SelectionProcedure.Employment_interviews_unstructured, 1),
    ]
    rv_standard_steps: StepsType = [
        (SelectionProcedure.Employment_interviews_unstructured, 20),
        (CV_screening, 1),
    ]
    advanced_steps: StepsType = [
        (CV_screening, 50),
        (test_screening, 20),
        (SelectionProcedure.Employment_interviews_unstructured, 1),
    ]
    ro_advanced_steps: StepsType = [
        (test_screening, 20),
        (SelectionProcedure.Employment_interviews_unstructured, 10),
        (CV_screening, 1),
    ]
    serious_steps: StepsType = [
        (serious_CV_screening, 50),
        (test_screening, 20),
        (SelectionProcedure.Employment_interviews_structured, 1),
    ]
    ro_serious_steps: StepsType = [
        (test_screening, 20),
        (SelectionProcedure.Employment_interviews_structured, 20),
        (serious_CV_screening, 1),
    ]

    our_steps: StepsType = [(ideal_screening, 1)]
    our_steps_sel_q: StepsType = [
        (ideal_screening, 10),
        (SelectionProcedure.Employment_interviews_structured, 1),
    ]

    ### Simulation steps ###
    SimStepType = TypedDict(
        "SimStepType",
        {
            "title": str,
            "steps": StepsType,
            "df": pd.DataFrame | None,
        },
    )
    sim_steps: dict[str, SimStepType] = {
        "standard": {
            "title": "Standard selection process",
            "steps": standard_steps,
            "df": None,
        },
        "rv_standard": {
            "title": "Reversed standard process",
            "steps": rv_standard_steps,
            "df": None,
        },
        "advanced": {
            "title": "Advanced standard process",
            "steps": advanced_steps,
            "df": None,
        },
        "ro_advanced": {
            "title": "Reordered advanced standard process",
            "steps": ro_advanced_steps,
            "df": None,
        },
        "serious": {
            "title": "Serious standard process",
            "steps": serious_steps,
            "df": None,
        },
        "ro_serious": {
            "title": "Reordered serious standard process",
            "steps": ro_serious_steps,
            "df": None,
        },
        "our": {
            "title": "Our selection process",
            "steps": our_steps,
            "df": None,
        },
        "our_large": {
            "title": "Our selection process (10 000)",
            "steps": our_steps,
            "df": None,
            "applicant_data": applicant_data_large,  # type: ignore
        },
        "our_sel_q": {
            "title": "Our process (selection questions)",
            "steps": our_steps_sel_q,
            "df": None,
        },
        "our_sel_q_large": {
            "title": "Our process (selection questions, 10 000)",
            "steps": our_steps_sel_q,
            "df": None,
            "applicant_data": applicant_data_large,  # type: ignore
        },
    }

    if load_data:
        for alias, sim_step in sim_steps.items():
            print(f"Loading data for: {sim_step['title']}")
            sim_step["df"] = pd.read_csv(f"results/{alias}.csv")

    else:
        ### Run simulations ###
        processor_count = os.cpu_count()
        if processor_count is None:
            processor_count = 1
        elif processor_count > 4:
            processor_count -= 2
        elif processor_count > 2:
            processor_count -= 1

        def run_simulation_wrapper(sim_step: SimStepType) -> pd.DataFrame:
            print(f"Running simulation for: {sim_step['title']}")
            return get_simulation_result(
                applicant_data=sim_step.get("applicant_data", applicant_data),
                selection_steps=sim_step["steps"],
                n_simulations=SIMULATIONS,
                noise_type=noise_type,
            )

        if processor_count == 1:
            # Run simulations sequentially
            sim_results = []
            for sim_step in tqdm(sim_steps.values()):
                sim_results.append(run_simulation_wrapper(sim_step))
        else:
            # Run simulations in parallel
            with Pool(processor_count) as p:
                sim_results = list(
                    tqdm(
                        p.imap(run_simulation_wrapper, sim_steps.values()),
                        total=len(sim_steps),
                        desc="Simulations Progress",
                    )
                )

        # Save results
        for (alias, sim_step), sim_res in zip(sim_steps.items(), sim_results):
            sim_res.to_csv(f"results/{alias}.csv", index=False)
            sim_step["df"] = sim_res

    # Check if all data is available
    if any(not isinstance(x["df"], pd.DataFrame) for x in sim_steps.values()):
        raise RuntimeError("Missing data for some plots")

    # Plot the effects of reordering the selection process
    plots = [
        "standard",
        "rv_standard",
        "advanced",
        "ro_advanced",
        "serious",
        "ro_serious",
    ]
    dfs: list[tuple[pd.DataFrame, str]] = [  # type: ignore
        (sim_steps[x]["df"], sim_steps[x]["title"]) for x in plots if x in sim_steps
    ]
    plot_results(dfs, "Reversing the selection process")

    # Plot how we compare to other selection processes
    plots = [
        "standard",
        "advanced",
        "serious",
        "our",
        "our_sel_q",
        "our_large",
        "our_sel_q_large",
    ]
    dfs: list[tuple[pd.DataFrame, str]] = [  # type: ignore
        (sim_steps[x]["df"], sim_steps[x]["title"]) for x in plots if x in sim_steps
    ]
    plot_results(dfs, "Comparison of selection processes")
