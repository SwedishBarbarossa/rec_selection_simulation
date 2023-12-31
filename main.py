import os
from enum import Enum
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from compiled import (
    generate_applicant_data,
    get_proper_noise_level,
    simulated_measurements,
)
from simulation import combine_selection_predicates, run_simulation


def plot_results(dfs: list[tuple[pd.DataFrame, str]], plot_title: str) -> None:
    # Set up dataframes for comparison
    processed_dfs = []
    median_colors = {}  # Dictionary to store median values and corresponding colors
    for df, selection_process in dfs:
        df_copy = df.copy()
        df_copy["applicant_score"] /= np.max(  # fit applicant_score to 0-1 scale
            df_copy["applicant_score"]
        )
        df_copy["selection_process"] = selection_process
        repeated_df = df_copy.loc[np.repeat(df_copy.index.values, df_copy["density"])]
        median = repeated_df["applicant_score"].median()  # Calculate median
        median_colors[selection_process] = 1 - median  # Store median value
        processed_dfs.append(repeated_df)

    res_df = pd.concat(processed_dfs)

    # Set the style for dark background with white text and white gridlines
    plt.style.use("dark_background")

    # Create a custom color palette based on median values
    color_palette = sns.color_palette("coolwarm", as_cmap=True)
    color_mapping = {
        process: color_palette(median_colors[process] / max(median_colors.values()))
        for process in median_colors
    }

    # Create seaborn plot
    g = sns.boxplot(
        data=res_df,
        x="selection_process",
        y="applicant_score",
        hue="selection_process",
        legend=False,
        palette=color_mapping,
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
    # Fit data to 0-1 scale
    applicant_data_cpy: npt.NDArray[np.float64] = applicant_data.copy()
    applicant_data_cpy /= max(
        np.max(applicant_data_cpy), abs(np.min(applicant_data_cpy))
    )

    ### Plot the base score distribution ###
    # Set the style for dark background with white text and white gridlines
    plt.style.use("dark_background")

    # Create seaborn plot
    g = sns.lineplot(data=applicant_data_cpy, color="white")
    g.set_ylabel("Real applicant score")
    g.set_xlabel("Applicant ID")
    g.set_title("Applicant score distribution")

    # Set y-axis limits
    g.set_ylim(-1.1, 1.1)

    # set size
    plt.gcf().set_size_inches(12, 8)

    # save plot
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = "applicant_score_distribution"
    plt.savefig(f"{current_dir}/plots/{file_name}{name_suffix}.png")

    # clear plot
    plt.clf()

    ### Plot the noised score distribution ###
    NUM_SIMULATIONS = 7  # Number of simulated selections per correlation
    for correlate in [0.1, 0.3, 0.5, 0.7]:
        # Get simulated measurements
        noise_level = get_proper_noise_level(applicant_data_cpy, correlate, noise_type)
        noised_applicant_data = simulated_measurements(
            applicant_data_cpy, noise_level, noise_type
        )

        # Plot the noised score distribution
        g = sns.scatterplot(data=noised_applicant_data, color="white", marker="x")
        g.set_ylabel("Simulated applicant score")
        g.set_xlabel("Applicant ID")
        g.set_title(f"Simulated applicant measurement ({noise_type}, {correlate})")

        # Add line based on the real applicant score
        g = sns.lineplot(data=applicant_data_cpy, color="red")  # Line plot

        # Set y-axis limits
        limits = (
            max(noised_applicant_data.max(), abs(noised_applicant_data.min())) * 1.1
        )
        g.set_ylim(-limits, limits)

        # set size
        plt.gcf().set_size_inches(12, 8)

        # save plot
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = f"simulated_applicant_measurement_{noise_type}_{correlate}".replace(
            ".", "_"
        )
        plt.savefig(f"{current_dir}/plots/{file_name}{name_suffix}.png")

        # clear plot
        plt.clf()

        # Run a couple of simulations and pick out the top 20 applicants
        dataframes = []
        for i in range(NUM_SIMULATIONS):
            # Get simulated measurements
            noised_applicant_data = simulated_measurements(
                applicant_data_cpy, noise_level, noise_type
            )

            # get indices of the top 20 applicants
            top_20 = np.argsort(noised_applicant_data)[-20:]
            top_20_scores = noised_applicant_data[top_20]
            dataframes.append(
                pd.DataFrame(
                    {
                        "applicant_score": top_20_scores,
                        "applicant_ID": top_20,
                        "simulation": f"Simulation {i+1}",
                    }
                )
            )

        df = pd.concat(dataframes)
        # Plot the noised score distribution
        g = sns.scatterplot(
            x=df["applicant_ID"],
            y=df["applicant_score"],
            hue=df["simulation"],
            palette="tab10",
            s=100,
        )
        g.set_ylabel("Simulated applicant score")
        g.set_xlabel("Applicant ID")
        g.set_title(f"Simulated top 20 applicant selection ({noise_type}, {correlate})")

        # Add line based on the real applicant score
        g = sns.lineplot(data=applicant_data_cpy, color="red")  # Line plot

        # Set x-axis limits to leave space for the legend
        g.set_xlim(-applicant_data_cpy.size * 0.05, applicant_data_cpy.size * 1.25)

        # set size
        plt.gcf().set_size_inches(12, 8)

        # save plot
        current_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = f"simulated_applicant_selection_{noise_type}_{correlate}".replace(
            ".", "_"
        )
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
    SIMULATIONS = 100_000  # Number of simulations to run
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
    ro_serious_final_step = combine_selection_predicates(
        [
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Interests,
            SelectionProcedure.Employment_interviews_structured,
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
    ideal_screening_bio = combine_selection_predicates(
        [
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Cognitive_ability_tests,
            SelectionProcedure.Big_5_overall,
            SelectionProcedure.Commute_distance,
            SelectionProcedure.Empirically_keyed_biodata,
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
        (SelectionProcedure.Job_experience_years, 50),
        (test_screening, 20),
        (SelectionProcedure.Employment_interviews_structured, 1),
    ]
    ro_serious_steps: StepsType = [
        (test_screening, 20),
        (ro_serious_final_step, 1),
    ]

    our_steps: StepsType = [(ideal_screening, 1)]
    our_steps_sel_q: StepsType = [
        (ideal_screening, 10),
        (SelectionProcedure.Employment_interviews_structured, 1),
    ]
    our_steps_two: StepsType = [(ideal_screening_bio, 1)]
    our_steps_two_sel_q: StepsType = [
        (ideal_screening_bio, 10),
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
            "title": "Standard process",
            "steps": standard_steps,
            "df": None,
        },
        "rv_standard": {
            "title": "Reversed standard process",
            "steps": rv_standard_steps,
            "df": None,
        },
        "advanced": {
            "title": "Advanced process",
            "steps": advanced_steps,
            "df": None,
        },
        "ro_advanced": {
            "title": "Reordered advanced process",
            "steps": ro_advanced_steps,
            "df": None,
        },
        "serious": {
            "title": "Serious process",
            "steps": serious_steps,
            "df": None,
        },
        "ro_serious": {
            "title": "Reordered serious process",
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
        "our_bio": {
            "title": "Our selection process (biodata)",
            "steps": our_steps_two,
            "df": None,
        },
        "our_bio_large": {
            "title": "Our selection process (biodata, 10 000)",
            "steps": our_steps_two,
            "df": None,
            "applicant_data": applicant_data_large,  # type: ignore
        },
        "our_bio_sel_q": {
            "title": "Our process (biodata, selection questions)",
            "steps": our_steps_two_sel_q,
            "df": None,
        },
        "our_bio_sel_q_large": {
            "title": "Our process (biodata, selection questions, 10 000)",
            "steps": our_steps_two_sel_q,
            "df": None,
            "applicant_data": applicant_data_large,  # type: ignore
        },
    }

    if load_data:
        for alias, sim_step in sim_steps.items():
            print(f"Loading data for: {sim_step['title']}")
            sim_step["df"] = pd.read_csv(f"results/{alias}.csv")

    else:
        sim_results = []
        with tqdm(total=len(sim_steps), desc="Simulations") as pbar:
            for sim_step in sim_steps.values():
                sim_results.append(
                    run_simulation(
                        applicant_data=sim_step.get("applicant_data", applicant_data),
                        selection_steps=sim_step["steps"],
                        n_simulations=SIMULATIONS,
                        noise_type=noise_type,
                        sim_title=sim_step["title"],
                    )
                )
                pbar.update(1)

        # Save results
        for (alias, sim_step), sim_res in zip(sim_steps.items(), sim_results):
            sim_res.to_csv(f"results/{alias}.csv", index=False)
            sim_step["df"] = sim_res

    # Check if all data is available
    if any(not isinstance(x["df"], pd.DataFrame) for x in sim_steps.values()):
        raise RuntimeError("Missing data for some plots")

    # Plot the standard selection processes
    plots = [
        "standard",
        "advanced",
        "serious",
    ]
    dfs: list[tuple[pd.DataFrame, str]] = [  # type: ignore
        (sim_steps[x]["df"], sim_steps[x]["title"]) for x in plots if x in sim_steps
    ]
    plot_results(dfs, "Standard selection processes")

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

    # Plot our selection processes
    plots = [
        "our",
        "our_sel_q",
        "our_bio",
        "our_bio_sel_q",
        "our_large",
        "our_sel_q_large",
        "our_bio_large",
        "our_bio_sel_q_large",
    ]
    dfs: list[tuple[pd.DataFrame, str]] = [  # type: ignore
        (sim_steps[x]["df"], sim_steps[x]["title"]) for x in plots if x in sim_steps
    ]
    plot_results(dfs, "Our selection process at different stages")

    # Plot our current and future selection processes compared to the standard process
    plots = [
        "standard",
        "advanced",
        "our",
        "our_large",
        "our_bio_sel_q_large",
    ]
    dfs: list[tuple[pd.DataFrame, str]] = [  # type: ignore
        (sim_steps[x]["df"], sim_steps[x]["title"]) for x in plots if x in sim_steps
    ]
    plot_results(dfs, "Our process compared to current options")
