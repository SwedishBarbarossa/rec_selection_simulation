import os
from enum import Enum
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from numba import njit
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


@njit(cache=True)
def calculate_expected_costs(
    chances: npt.NDArray[np.float64], costs: npt.NDArray[np.int64], failed_cost: int
):
    """Calculate the expected average cost for each combination of chance and cost."""
    expected_costs = np.zeros((chances.size, costs.size))
    for i in range(chances.size):
        for j in range(costs.size):
            expected_costs[i][j] = costs[j] * chances[i] + (
                failed_cost * (1 - chances[i])
            )

    return expected_costs


def plot_cost_benefit(
    standard_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
    serious_df: pd.DataFrame,
    our_df: pd.DataFrame,
    our_large_df: pd.DataFrame,
) -> None:
    MAX_COST = 200_000  # Maximum cost of recruitment
    MIN_COST = 0  # Minimum cost of recruitment
    cost_step = 25  # Step size for cost of recruitment
    cost_span = MAX_COST - MIN_COST
    if cost_span % cost_step != 0:
        raise RuntimeError("Cost span is not divisible by cost step")
    # Number of steps for cost of recruitment
    COST_STEPS = (MAX_COST - MIN_COST) // cost_step
    cost_tick_step = 10_000  # Step size for cost of recruitment
    # Number of steps for cost of recruitment
    COST_TICK_STEPS = (MAX_COST - MIN_COST) // cost_tick_step + 1

    ALTERNATIVE_COST = 700_000  # Cost of failed recruitment

    MIN_CHANCE = 0.3  # Minimum percentage of successful recruitment
    MAX_CHANCE = 1.0  # Maximum percentage of successful recruitment
    chance_steps_per_p = 10  # Number of steps per percentage point
    chance_span = (MAX_CHANCE - MIN_CHANCE) * 100
    if chance_span % chance_steps_per_p != 0:
        raise RuntimeError("Chance span is not divisible by chance step")
    # Number of steps for chance of success
    CHANCE_STEPS = int((MAX_CHANCE - MIN_CHANCE) * 100 * chance_steps_per_p)
    chance_tick_step = 0.1  # Step size for chance of success
    CHANCE_TICK_STEPS = int(((MAX_CHANCE - MIN_CHANCE) // chance_tick_step)) + 2

    stats_mapping = {
        "Standard": {
            "cost": 30_000,  # guess
            "chance": 0,
            "df": standard_df.copy(),
        },
        "Advanced": {
            "cost": 100_000,  # guess
            "chance": 0,
            "df": advanced_df.copy(),
        },
        "Serious": {
            "cost": 150_000,  # guess
            "chance": 0,
            "df": serious_df.copy(),
        },
        "Our": {
            "cost": 30_000,
            "chance": 0,
            "df": our_df.copy(),
        },
        "Our (10 000)": {
            "cost": 30_000,
            "chance": 0,
            "df": our_large_df.copy(),
        },
    }

    for alias in stats_mapping:
        ### Calculate the chance of success ###
        # The DF has 3 columns:
        # - applicant_ID: The ID of the applicant
        # - applicant_score: The score of the applicant
        # - density: The number of times the applicant got selected in the process
        # Assume the relationship between applicant_score and chance of success is linear
        # Assume max applicant_score is 100% chance of success
        # Assume zero applicant_score is 0% chance of success

        # Get the DF
        df = stats_mapping[alias]["df"]

        # Density sum
        density_sum = df["density"].sum()

        # Normalize the score column to 0-1
        df["applicant_score"] /= np.max(df["applicant_score"])

        # Multiply the score column by density to get the total score
        df["applicant_score"] *= df["density"]

        # Normalize the score column by the number of selections
        df["applicant_score"] /= density_sum

        # Set the chance of success
        stats_mapping[alias]["chance"] = df["applicant_score"].sum()

    stats_df = pd.DataFrame(
        {
            "title": list(stats_mapping.keys()),
            "cost": [x["cost"] for x in stats_mapping.values()],
            "chance": [x["chance"] for x in stats_mapping.values()],
        }
    )

    # Set the style for dark background with white text and white gridlines
    plt.style.use("dark_background")

    ### Plot the heatmap ###
    costs = np.linspace(MAX_COST, MIN_COST, COST_STEPS + 1, endpoint=True).astype(
        np.int64
    )
    chances = np.linspace(
        MAX_CHANCE,
        MIN_CHANCE,
        CHANCE_STEPS + 1,
        endpoint=True,
    )

    # Calculating the expected average cost for each combination of chance and cost
    expected_costs = calculate_expected_costs(
        chances,
        costs,
        ALTERNATIVE_COST,
    )

    # Create a custom color map from blue to red
    cmap = sns.diverging_palette(
        240, 10, s=500, l=20, as_cmap=True, center="dark", sep=5
    )

    # Plotting the background heatmap
    sns.heatmap(
        expected_costs,
        cmap=cmap,
    )

    # Get the current Axes object to access the colorbar
    ax = plt.gca()
    # Access the colorbar object
    cbar = ax.collections[0].colorbar
    if cbar is None:
        raise RuntimeError("Could not access colorbar")

    # Set title for the color scale bar
    cbar.set_label(
        "Expected Total Cost of Recruitment (SEK)", rotation=270, labelpad=-55
    )

    cbar.ax.ticklabel_format(style="plain", useOffset=False)
    cbar_ticks = cbar.get_ticks()
    cbar.set_ticklabels([f"{x:,}".replace(",", " ").split(".")[0] for x in cbar_ticks])

    ### Plot the points ###
    # Scale stats_df to the same scale as the heatmap
    scaled_df = stats_df.copy()
    scaled_df["cost"] = (scaled_df["cost"] - MIN_COST) / (MAX_COST - MIN_COST)
    scaled_df["cost"] *= costs.size
    scaled_df["cost"] = costs.size - scaled_df["cost"]

    scaled_df["chance"] = (scaled_df["chance"] - MIN_CHANCE) / (MAX_CHANCE - MIN_CHANCE)
    scaled_df["chance"] *= chances.size
    scaled_df["chance"] = chances.size - scaled_df["chance"]

    # Marking the points
    sns.scatterplot(
        data=scaled_df,
        x="cost",
        y="chance",
        hue="title",
        palette="bright",
        s=250,
    )

    ### Finishing touches ###### Finishing touches ###
    # Add x-labels divisible by COST_TICK_STEPS from costs
    cost_tick_indices = np.linspace(
        0, COST_STEPS, COST_TICK_STEPS, endpoint=True, dtype=np.int64
    )
    plt.xticks(
        cost_tick_indices,
        [f"{cost:,}".replace(",", " ") for cost in costs[cost_tick_indices]],
    )

    # Add y-labels from 30% to 100% in 10% increments
    chance_tick_indices = np.linspace(
        0, CHANCE_STEPS, CHANCE_TICK_STEPS, endpoint=True, dtype=np.int64
    )
    print(chance_tick_indices)
    plt.yticks(
        chance_tick_indices,
        [f"{chance:.0%}" for chance in chances[chance_tick_indices]],
    )

    # Adding labels and title
    plt.xlabel("Cost of Recruitment (SEK)")
    plt.ylabel("Odds of Successful Recruitment")
    plt.title("Expected Average Cost of Recruitment based on Cost")

    # Set size
    plt.gcf().set_size_inches(12, 8)

    # Save the plot
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = "cost_benefit"
    plt.savefig(f"{current_dir}/plots/{file_name}.png")

    # Clear the plot
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

    ### Tests ###
    # Big 5
    Openness_to_experience = 0.05
    Conscientiousness = 0.19
    Extraversion = 0.10
    Agreeableness = 0.10
    Emotional_stability = 0.09

    # Big 5 contextualized
    Openness_to_experience_contextualized = 0.12
    Conscientiousness_contextualized = 0.25
    Extraversion_contextualized = 0.21
    Agreeableness_contextualized = 0.19
    Emotional_stability_contextualized = 0.23

    # rest of tests
    Job_knowledge_tests = 0.4
    Work_sample_tests = 0.33
    Cognitive_ability_tests = 0.31
    Integrity_tests = 0.31
    Personality_based_EI = 0.3  # Emotional intelligence
    SJT_knowledge = 0.26  # situational judgment test
    SJT_behavioral_tendency = 0.26  # situational judgment test
    Ability_based_EI = 0.22  # Emotional intelligence

    ### Interviews ###
    Employment_interviews_structured = 0.42
    Employment_interviews_unstructured = 0.19

    ### Biodata ###
    Empirically_keyed_biodata = 0.38  # questionnaires
    Interests = 0.24
    Rationally_keyed_biodata = 0.22  # questionnaires
    Job_experience_years = 0.09

    ### Other ###
    Assessment_centers = 0.29
    Commute_distance = 0.103  # from 10.1080/10803548.2021.2010970

    @classmethod
    def Big_5_contextualized(cls) -> list[float]:  # personality test
        return [
            cls.Openness_to_experience_contextualized,
            cls.Conscientiousness_contextualized,
            cls.Extraversion_contextualized,
            cls.Agreeableness_contextualized,
            cls.Emotional_stability_contextualized,
        ]

    @classmethod
    def Big_5_overall(cls) -> list[float]:  # personality test
        return [
            cls.Openness_to_experience,
            cls.Conscientiousness,
            cls.Extraversion,
            cls.Agreeableness,
            cls.Emotional_stability,
        ]


if __name__ == "__main__":
    ### Configuration ###
    N = 200  # Number of applicants at the start of the selection process
    SIMULATIONS = 100_000  # Number of simulations to run
    SELECTED_DISTRIBUTION = ApplicantScoreDistributionTypes.power_law.value
    NOISE_TYPE = NoiseDistributionTypes.normal.value
    LOAD_DATA = True  # Set to False to regenerate data
    GENERATE_EXPLANATORY_PLOTS = False  # Set to True to generate explanatory plots
    GENERATE_COST_BENEFIT_PLOT = False  # Set to True to generate cost-benefit plot
    #####################

    ### Generate applicant data ###
    # Generate default applicant data based on the selected distribution
    applicant_data = generate_applicant_data(N, SELECTED_DISTRIBUTION)
    applicant_data_large = generate_applicant_data(10_000, SELECTED_DISTRIBUTION)

    # Create plot folder if it doesn't exist
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(f"{current_dir}/plots"):
        os.makedirs(f"{current_dir}/plots")

    if GENERATE_EXPLANATORY_PLOTS:
        # Plot the score distributions
        plot_score_distribution(applicant_data, NOISE_TYPE)
        plot_score_distribution(applicant_data_large, NOISE_TYPE, "_large")

    #### Screenings ####
    CV_screening = SelectionProcedure.Job_experience_years
    test_screening = combine_selection_predicates(
        [
            *SelectionProcedure.Big_5_overall(),
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
            *SelectionProcedure.Big_5_overall(),
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Cognitive_ability_tests,
            SelectionProcedure.Commute_distance,
        ]
    )
    ideal_screening_bio = combine_selection_predicates(
        [
            *SelectionProcedure.Big_5_overall(),
            SelectionProcedure.Job_experience_years,
            SelectionProcedure.Cognitive_ability_tests,
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

    if LOAD_DATA:
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
                        noise_type=NOISE_TYPE,
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

    if GENERATE_COST_BENEFIT_PLOT:
        # Plot the cost-benefit plot
        plot_cost_benefit(
            standard_df=sim_steps["standard"]["df"],  # type: ignore
            advanced_df=sim_steps["advanced"]["df"],  # type: ignore
            serious_df=sim_steps["serious"]["df"],  # type: ignore
            our_df=sim_steps["our"]["df"],  # type: ignore
            our_large_df=sim_steps["our_large"]["df"],  # type: ignore
        )

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
