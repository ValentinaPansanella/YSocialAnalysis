# YSocialAnalysis

This repository contains Python scripts for analyzing the output of social network simulations, with a focus on the performance and effects of different recommendation systems. The analyses investigate the relationship between user connectivity (degree) and content visibility, as well as the overall distribution of recommendations across users and posts.

## Core Analyses

This project performs two primary analyses on simulation data from Barabási-Albert (BA) and Erdős-Rényi (ER) networks using 'Follow' (F) and 'Random-Content' (RC) recommendation systems.

1.  **Recommendations vs. Author Degree (`rec_vs_degree.py`)**
    This analysis explores the correlation between an author's degree in the social network and the visibility their posts receive. It aggregates data across multiple simulation runs to compute and plot:
    *   **Total Recommendations:** The average number of times an author's content is recommended.
    *   **Unique Reach:** The average number of unique users who see an author's content.
    *   **ΔR (Delta Recs):** The net attention flow for a user, calculated as (`unique_reach` - `authors_viewed`). This metric indicates whether a user is a net content broadcaster (positive ΔR) or consumer (negative ΔR).
    *   **Normalized Recommendations:** Total recommendations normalized by the number of posts an author has created.

2.  **Recommendation Distribution (`recs_distribution.py`)**
    This script analyzes the concentration of recommendations. It generates log-log frequency distribution plots to answer:
    *   How many posts receive *k* recommendations?
    *   How many authors receive *k* recommendations for their content?
    This helps visualize the inequality in content visibility driven by the recommendation algorithms.

## Repository Structure

```
.
├── figs/                     # Output directory for generated figures
├── res/                      # Output directory for log files
├── notebook/
│   ├── support.ipynb         # Exploratory notebook for developing analysis logic.
│   └── unzip_files.ipynb     # Utility notebook to decompress simulation data archives.
├── rec_vs_degree.py          # Main script for the recommendations vs. degree analysis.
├── recs_distribution.py      # Main script for the recommendation distribution analysis.
├── requirements.txt          # Python package dependencies.
├── utils.py                  # Helper functions for data loading, degree calculation, and metrics.
└── utils_figures.py          # Plotting functions and style configuration for visualizations.
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   Simulation data from the `experiments_recsys` project.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ValentinaPansanella/YSocialAnalysis.git
    cd YSocialAnalysis
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Data Setup

The analysis scripts expect the simulation data to be located in a directory named `experiments_recsys` in the project root. The data for each simulation run is typically provided as a ZIP archive (e.g., `BA_RC_0.zip`).

1.  Place all your simulation `.zip` files inside the `experiments_recsys` directory.
2.  Unzip each archive into its own folder. The `notebook/unzip_files.ipynb` notebook provides a simple script to perform this operation. The final structure should look like this:
    ```
    experiments_recsys/
    ├── BA_RC_0/
    │   ├── database_server.db
    │   └── ...
    ├── BA_RC_1/
    │   └── ...
    └── ...
    ```

## Usage

The analysis can be run directly from the command line. The scripts are configured via constants in `utils.py` (e.g., `NETWORKS`, `RECSYSS`, `RUNS`).

1.  **Run the Recommendations vs. Degree analysis:**
    ```sh
    python rec_vs_degree.py
    ```
    This will generate log files in `res/` and save output plots in `figs/recommendations_vs_degree/`.

2.  **Run the Recommendation Distribution analysis:**
    ```sh
    python recs_distribution.py
    ```
    This will generate log files in `res/` and save output plots in `figs/recommendations/`.

The scripts will automatically iterate through the configured networks, recommendation systems, and run numbers, process the data, and save the resulting figures and log files.
