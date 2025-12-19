# MOBE Battery Toolkit

MOBE Battery Toolkit

Overview

The MOBE Battery Toolkit is a Python-based simulation and analysis engine designed to model Lithium-Ion battery behavior. It integrates the KiBaM (Kinetic Battery Model) to simulate charge recovery effects with an Equivalent Circuit Model (ECM) to capture transient voltage responses.

This toolkit provides a modular framework for ingesting raw battery datasets (specifically NASA formatting), extracting Open Circuit Voltage (OCV) curves, optimizing model parameters using Simulated Annealing, and visualizing aging characteristics over a battery's lifecycle.

Project Structure

The project directory is organized as follows:

```
MOBE_PROJECT/
├── batData/                  # Directory for raw input files (e.g., NASA .mat files)
├── Batteries/                # Processed output directory
│   └── [Battery_ID]/         # Specific folder per battery entity
│       ├── battery_details.json   # Configuration metadata
│       └── BatteryData/           # CSVs for cycle data, aging summaries, and impedance
├── MOBI.venv/                # Python Virtual Environment
├── MOBE Battery Toolkit.py   # Main application entry point
└── README.md                 # Project documentation
```


Installation and Setup

Prerequisites

Python 3.8 or higher

Recommended IDE: VS Code or PyCharm

Environment Configuration

It is recommended to run this toolkit within a virtual environment to manage dependencies and avoid conflicts.

Windows (PowerShell/CMD):

# Create the environment
python -m venv MOBI.venv

# Activate the environment
.\MOBI.venv\Scripts\activate


Mac/Linux:

# Create the environment
python3 -m venv MOBI.venv

# Activate the environment
source MOBI.venv/bin/activate


Dependencies

Install the required Python libraries using pip:

pip install numpy pandas scipy matplotlib numba


numpy & pandas: Utilized for high-performance data manipulation and time-series analysis.

scipy: Required for parsing legacy MATLAB (.mat) files.

matplotlib: Used for generating analysis dashboards and simulation plots.

numba: A JIT compiler that significantly accelerates the physics solver (KiBaM equations) by compiling Python code to machine instructions.

System Architecture

The application is modularized into distinct logical blocks:

1. Configuration (BatteryConfig, SimulationDefaults)

Defines the simulation parameters and physical constraints. SimulationDefaults contains the initialization values and boundary conditions for the optimizer:

Q: Total Capacity (Ah)

k: Diffusion rate between bound and available charge tanks

c: Ratio of available charge tank to total capacity

R_s, R1, C1: Equivalent circuit components (Series resistance and RC pair)

2. Physics Engine (KiBaMPhysics, BatteryModel)

This module implements the mathematical core of the simulation:

KiBaM Solver: Solves the two-tank hydraulic model differential equations to calculate charge flow between "available" (immediate current) and "bound" (recovery effect) charge states.

Circuit Model: Superimposes voltage drops across the internal resistance and RC pairs onto the Open Circuit Voltage (OCV) to determine terminal voltage.

Optimization: Utilizes numba to compile the iterative solver loop, improving execution speed by approximately 50x.

3. Optimization Engine (ModelOptimizer)

Fits the physics model to experimental data using Simulated Annealing. The algorithm iteratively perturbs battery parameters ($k, c, R, Q$) to minimize the Root Mean Square Error (RMSE) between the modeled voltage and measured voltage. It employs a temperature-based acceptance probability to avoid local minima during the optimization process.

4. Data Ingestion (NasaIngestor, DataStandardizer)

Parses complex nested structures within NASA MATLAB (.mat) files. This module extracts charge, discharge, and impedance cycles, standardizing column names and units for internal processing.

5. Visualization (Visualizer)

Provides graphical insights into the battery analysis:

Dashboard: Displays Capacity Fade and Impedance Rise over the battery's lifecycle.

Parameter Evolution: Visualizes the drift of internal parameters ($k, c, R_s$) as the battery ages.

Simulation Result: Overlays model predictions against real-world data for validation.

Usage Guide

Run the application via the terminal:

python "MOBE Battery Toolkit.py"


Standard Workflow

1. Data Import
Select option 2. Create Battery / Import NASA Data.

Enter a unique Battery ID (e.g., B0005).

Define nominal specifications (Capacity, Min/Max Voltage).

Provide the path to the raw .mat file for ingestion.

2. OCV Extraction
Select option 3. Extract OCV Curve.

The system requires a reference Open Circuit Voltage curve to function. It attempts to extract this from low-current discharge cycles or the initial discharge cycle in the dataset.

3. Analysis Modes
Once the data is prepared, several analysis modes are available:

Lifecycle Analysis (Option 4): Iterates through every discharge cycle in the dataset, running the optimizer to fit parameters for each cycle. This generates a parameter_evolution.csv file, tracking how internal resistance increases or capacity fades over time.

Single Cycle Fit (Option 5): Targets a specific cycle ID for high-resolution fitting. This is useful for debugging or detailed analysis of specific aging points.

Simulation Playground (Option 7): Allows the user to define synthetic load profiles (Constant Current or Pulse Discharge). The tool predicts voltage response based on the learned parameters.

4. Visualization
Select option 1. Dashboard to view summary statistics, including capacity fade curves and impedance evolution graphs.

Dependencies: 
numpy
pandas
scipy
matplotlib
numba
