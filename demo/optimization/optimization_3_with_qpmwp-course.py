############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 3 - USING CLASSES FROM QPMWP-COURSE
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     26.01.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------



# Install scipy
# uv pip install scipy



# IPython/Jupyter magic commands used for automatically reloading Python modules 
# during development (so that changes in the code are reflected without needing to restart the kernel):

# %reload_ext autoreload
# %autoreload 2




# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = r'C:\Users\marce\OneDrive - Universität Zürich UZH\YEAR 202526\QUANTITATIVE PORTFOLIO MANAGEMENT WITH PYTHON\qpmwp-course'
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.constraints import Constraints
from optimization.quadratic_program import QuadraticProgram
from optimization.optimization_data import OptimizationData
from optimization.optimization import MeanVariance







# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 10
data = load_data_msci(path = r'C:\Users\marce\OneDrive - Universität Zürich UZH\YEAR 202526\QUANTITATIVE PORTFOLIO MANAGEMENT WITH PYTHON\qpmwp-course\data\\', n = N)
data





# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix
# --------------------------------------------------------------------------

return_series = data['return_series']
scalefactor = 1  # could be set to 252 (trading days) for annualized returns

# --------------------------------------------------------------------------
# 1. Estimation of Expected Returns (mu)
# --------------------------------------------------------------------------

# We use the 'geometric' method which accounts for the 'volatility drag' (risk) 
# and provides the compound growth rate of the assets.
# Formula from mean_geometric(): mu = exp( mean( ln(1 + R) ) * scalefactor ) - 1
expected_return = ExpectedReturn(
    method='geometric',
    scalefactor=scalefactor # 1 for daily; 252 for annual
)

# Implementation Details:
# The code converts simple returns R to log-returns ln(1+R), calculates the mean,
# applies the scaling, and then converts back to simple returns via exp(x)-1.
# This is more numerically stable than the product: ((1+R1)(1+R2)...(1+RT))^(1/T)
expected_return.estimate(X=return_series, inplace=True)
mu = expected_return.estimate(X=return_series, inplace=False)


# --------------------------------------------------------------------------
# 2. Estimation of Covariance Matrix (Sigma)
# --------------------------------------------------------------------------

# 'pearson' calculates the standard sample covariance matrix.
# Formula from cov_pearson(): Σ_ij = E[ (R_i - E[R_i])(R_j - E[R_j]) ]
# Check eigenvalues: If the minimum is > 0, the matrix is Positive Definite

# PRE-CHECK: Inspect raw data eigenvalues to see if the matrix is naturally "healthy"
# If min eigenvalue < 0, the optimizer will crash without the 'make_pos_def' logic.
raw_cov = return_series.cov()
print(f"Pre-repair Min Eigenvalue: {np.linalg.eigvals(raw_cov).min():.10e}")

covariance = Covariance(
    method='pearson',
    check_positive_definite=True # Critical for solver stability
)

# Logic for positive definiteness (is_pos_def & make_pos_def):
# 1. Verification: Uses Cholesky Decomposition (np.linalg.cholesky). If it fails,
#    the matrix is not Positive Definite (PD) and cannot be used in a QP solver.
# 2. Correction: Uses Singular Value Decomposition (SVD) to project the matrix 
#    onto the nearest PD space (Higham 1988 method). This ensures the 
#    quadratic form w'Σw is always > 0 (strictly positive risk).
# --- 3. THE "NEAREST PD" LOGIC (SVD REPAIR) ---
# If Sigma is not Positive Definite (PD), the QP solver crashes because it 
# would theoretically find 'negative' risk, leading to infinite weights.
# The make_pos_def() function uses SVD (Singular Value Decomposition) 
# to project Sigma onto the nearest valid PD space by removing 
# negative/zero eigenvalues while preserving the original correlation structure.
covariance.estimate(X=return_series, inplace=True)
Sigma = covariance.estimate(X=return_series, inplace=False)


# --------------------------------------------------------------------------
# 3. Visual Analysis of Estimates
# --------------------------------------------------------------------------

# Create a figure with two subplots: one for Returns, one for Risk/Correlation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot A: Expected Returns (mu) ---
# We use a bar chart to compare the 'Geometric Mean' across countries
mu.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title(f"Expected Returns (Geometric, Scale={scalefactor})")
ax1.set_ylabel("Expected Return")
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- Plot B: Correlation Heatmap ---
# We convert the Covariance (Sigma) to Correlation for easier interpretation
# Correlation scales everything between -1 and 1
vola = np.sqrt(np.diag(Sigma))
corr_matrix = Sigma / np.outer(vola, vola)

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax2, 
            xticklabels=mu.index, yticklabels=mu.index, center=0)
ax2.set_title("Asset Correlation Matrix (from Sigma)")

plt.tight_layout()
plt.show()



# --------------------------------------------------------------------------
# Constraints: Setting the "Rules of the Game" for the Portfolio
# --------------------------------------------------------------------------

# 1. Instantiate the class
# We pass the list of country codes (AT, AU, BE...) so the class knows 
# the 'dimension' (N=10) of the problem it needs to solve.
constraints = Constraints(ids = return_series.columns.tolist())

# 2. Add budget constraint (Sum of weights = 1.0)
# This ensures we are "fully invested" and not using leverage.
# Math: w1 + w2 + ... + w10 = 1
constraints.add_budget(rhs=1, sense='=')

# 3. Add box constraints (Asset-level limits)
# We set a floor of 0 (No Short-Selling) and a ceiling of 0.2 (20%).
# This prevents the optimizer from putting 100% of your money into one country,
# forcing diversification across at least 5 different assets.
# Math: 0 <= wi <= 0.2
constraints.add_box(lower=0, upper=0.2)

# 4. Add linear constraints (Group-level limits)
# Here we are creating "Country Clusters" (e.g., perhaps by region).
G = pd.DataFrame(np.zeros((2, N)), columns=constraints.ids)

# Rule 1: The first 5 assets (e.g., AT through CA) cannot exceed 50% total.
G.iloc[0, 0:5] = 1
# Rule 2: The last 5 assets cannot exceed 50% total.
G.iloc[1, 5:10] = 1

h = pd.Series([0.5, 0.5]) 

# Math: G * w <= h 
# (e.g., w1 + w2 + w3 + w4 + w5 <= 0.5)
constraints.add_linear(G=G, sense='<=', rhs=h)

# 5. Inspect the "Rulesets"
# These return the internal dictionaries we defined in the Class logic.
constraints.budget   # Check Amat (ones), sense (=), and rhs (1)
constraints.box      # Check the Series of 0s and 0.2s
constraints.linear   # Check the G matrix and rhs (0.5)



# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class QuadraticProgram
# --------------------------------------------------------------------------

# 1. PREPARE CONSTRAINTS
# Transform the high-level constraint object into the standard matrices 
# required for QP: G (inequality), h (inequality vector), A (equality), and b (equality vector).
GhAb = constraints.to_GhAb()
GhAb

# 2. DEFINE OBJECTIVE FUNCTION PARAMETERS
# Setting the risk-aversion parameter (lambda). 
# Higher values prioritize risk reduction over return.
risk_aversion = 3

# 3. INITIALIZE THE QP WRAPPER
# P: The quadratic part (Covariance Matrix * Risk Aversion). 
#    Note: QuadraticProgram will automatically check for Positive Semi-Definiteness (PSD).
# q: The linear part (Negative Expected Returns) because we are MINIMIZING.
# lb/ub: Box constraints (asset weights limits) passed to the 'problem_data' dict.
qp = QuadraticProgram(
    P = covariance.matrix.to_numpy() * risk_aversion,
    q = expected_return.vector.to_numpy() * -1,
    G = GhAb['G'],
    h = GhAb['h'],
    A = GhAb['A'],
    b = GhAb['b'],
    lb = constraints.box['lower'].to_numpy(),
    ub = constraints.box['upper'].to_numpy(),
    solver = 'cvxopt',
)

# Inspect the structured problem data stored within the instance
qp.problem_data

# 4. PRE-FLIGHT FEASIBILITY CHECK
# Uses the class method to solve a zero-objective problem. 
# If this returns False, the solver will fail regardless of the returns or risk.
qp.is_feasible()

# 5. EXECUTE OPTIMIZATION
# This triggers: PSD checks -> Sparse conversion (if applicable) -> Solver execution.
qp.solve()

# Retrieve the minimized cost: 0.5 * x'Px + q'x + constant
qp.objective_value()

# 6. EXTRACT & EVALUATE RESULTS
# Access the raw qpsolvers solution object stored in self._results
solution = qp.results.get('solution')
solution

# 7. CONVERGENCE DIAGNOSTICS
# Verify if the solver actually converged (solution.found) 
# and check the numerical quality via residuals and duality gap.
solution.found
solution.primal_residual()   # Measures constraint violation
solution.dual_residual()     # Measures optimality error
solution.duality_gap()[0]    # Difference between primal and dual objective

# 9. Result Visualization - Mapping weights to asset names

# 1. Map weights to asset names
# We use the index/columns from our input data to ensure the labels match
asset_names = expected_return.vector.index
weights = pd.Series(solution.x, index=asset_names, name="Optimal Weights")

# 2. Filter out tiny values (noise) for a cleaner plot
weights_filtered = weights[weights.abs() > 1e-4]

# 3. Plotting
plt.figure(figsize=(10, 6))
weights_filtered.sort_values().plot(
    kind='barh', 
    color='skyblue', 
    edgecolor='navy'
)

plt.title(f'Mean-Variance Optimal Portfolio Weights (Risk Aversion: {risk_aversion})')
plt.xlabel('Weight')
plt.ylabel('Assets')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8) # Reference line at 0
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Optional: Print the top 5 holdings
print("Top 5 Holdings:")
print(weights.sort_values(ascending=False).head(5))




# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios - using class MeanVariance
# --------------------------------------------------------------------------

# 1. STRATEGY INSTANTIATION
# Initialize the Mean-Variance strategy. This links the estimators 
# (covariance/returns) and the constraints with the solver settings.
# Note: 'risk_aversion' (lambda) acts as the trade-off coefficient between risk and return.
mv = MeanVariance(
    covariance=covariance,
    expected_return=expected_return,
    constraints=constraints,
    risk_aversion=1,
    solver_name='cvxopt',
)

# Inspect the solver parameters and hyperparameters
mv.params

# 2. DATA PREPARATION
# Encapsulate the historical data window. Using the last 256 observations 
# (roughly 1 trading year) to estimate the forward-looking covariance and returns.
optimization_data = OptimizationData(return_series=return_series.tail(256))

# 3. AUTOMATED OBJECTIVE DERIVATION
# This calls the internal estimators. It transforms the return series into 
# the matrices P and q using the formula: P = 2 * lambda * Cov, q = -Returns.
mv.set_objective(optimization_data=optimization_data)

# Verify the calculated coefficients before solving
mv.objective.coefficients

# 4. EXECUTION
# The solve() method handles:
#   a) Extracting GhAb from constraints
#   b) Validating Matrix P is Positive Definite
#   c) Passing all data to the QuadraticProgram wrapper
mv.solve()

# Inspect execution status (e.g., 'status': True) and the raw weights dictionary
mv.results

# 5. POST-PROCESSING
# Map the numerical solution back to a pandas Series with asset tickers 
# for downstream analysis and reporting.
weights_mv = pd.Series(mv.results['weights'], index=return_series.columns)
weights_mv

# --------------------------------------------------------------------------
# Result Visualization
# --------------------------------------------------------------------------

# 1. Clean up the data
# We convert the weights to a Series and remove assets with near-zero weights 
# to keep the plot legible.
plot_weights = weights_mv[weights_mv.abs() > 1e-4].sort_values()

# 2. Generate the plot
# Using a horizontal bar chart (barh) to accommodate long asset names.
plt.figure(figsize=(10, 8))
colors = ['#1f77b4' if w > 0 else '#d62728' for w in plot_weights] # Blue for Long, Red for Short

plot_weights.plot(
    kind='barh', 
    color=colors, 
    edgecolor='black', 
    alpha=0.8
)

# 3. Annotate the chart
# Reference lines and labels help interpret the 'risk_aversion' impact.
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.title(f'Mean-Variance Optimal Weights\n(Risk Aversion $\lambda$ = {mv.params["risk_aversion"]})', fontsize=14)
plt.xlabel('Portfolio Weight', fontsize=12)
plt.ylabel('Assets', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add a text box with the number of active positions
plt.text(0.95, 0.05, f'Active Positions: {len(plot_weights)}', 
         transform=plt.gca().transAxes, ha='right', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()




# --------------------------------------------------------------------------
# Solve for a tracking-error minimizing portfolio by least-squares
# Using class LeastSquares
# --------------------------------------------------------------------------

from optimization.optimization import LeastSquares

# 1. STRATEGY INSTANTIATION
# LeastSquares is used here for "Index Tracking." 
# It minimizes the squared difference between the portfolio and the benchmark.
ls = LeastSquares(
    constraints=constraints,
    solver_name='cvxopt',
)

# 2. DATA PREPARATION & ALIGNMENT
# We need both the candidate assets (X) and the target benchmark (y).
# 'align=True' ensures that the dates for both series match perfectly before 
# the regression matrices are calculated.
y = data['bm_series']
optimization_data = OptimizationData(
    return_series=return_series.tail(256),
    bm_series=y,
    align=True
)

# 3. CONSTRUCT THE OBJECTIVE
# This builds the P and q matrices: 
# P = 2 * X'X, q = -2 * X'y
ls.set_objective(optimization_data=optimization_data)

# 4. EXECUTE THE SOLVER
# Solves the constrained regression problem. Unlike standard OLS, 
# this respects your investment constraints (e.g., full investment, no-shorting).
ls.solve()

# 5. RESULT EXTRACTION
# Map weights to asset names for analysis.
weights_ls = pd.Series(ls.results['weights'], index=return_series.columns)

# --------------------------------------------------------------------------
# Visualization: Tracking Portfolio Composition
# --------------------------------------------------------------------------

# Filter out negligible weights and sort for a clean visual
plot_weights_ls = weights_ls[weights_ls.abs() > 1e-4].sort_values()

plt.figure(figsize=(10, 8))
plot_weights_ls.plot(
    kind='barh', 
    color='teal', 
    edgecolor='black', 
    alpha=0.7
)

plt.title('Least-Squares Tracking Portfolio Weights\n(Minimized Tracking Error)', fontsize=14)
plt.xlabel('Weight', fontsize=12)
plt.ylabel('Assets', fontsize=12)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Quick summary of the tracking quality
print(f"Total Portfolio Weight: {weights_ls.sum():.4f}")
print(f"Number of Active Replicators: {len(plot_weights_ls)}")

# --------------------------------------------------------------------------
# Alternative: Ridge Tracking (L2 Penalty)
# --------------------------------------------------------------------------

# 1. SET UP RIDGE OPTIMIZATION
# We use a penalty (alpha). A typical starting point is 0.1 or 1.0.
ls_ridge = LeastSquares(
    constraints=constraints,
    solver_name='cvxopt',
    l2_penalty=0.5  # This activates the Ridge logic in your class
)

# 2. CALCULATE OBJECTIVE & SOLVE
ls_ridge.set_objective(optimization_data=optimization_data)
ls_ridge.solve()

# 3. EXTRACT RIDGE WEIGHTS
weights_ridge = pd.Series(ls_ridge.results['weights'], index=return_series.columns)

# --------------------------------------------------------------------------
# COMPARISON PLOT: Pure LS vs. Ridge LS
# --------------------------------------------------------------------------

comparison_df = pd.DataFrame({
    'Pure LS': weights_ls,
    'Ridge LS': weights_ridge
})

# Filter for meaningful weights
comparison_df = comparison_df[(comparison_df.abs() > 0.01).any(axis=1)]

comparison_df.sort_values('Pure LS').plot(
    kind='barh', 
    figsize=(12, 8),
    color=['#1f77b4', '#ff7f0e'], # Blue vs Orange
    alpha=0.8
)

plt.title('Weight Distribution: Pure Tracking vs. Ridge Regularized', fontsize=14)
plt.xlabel('Portfolio Weight')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




# --------------------------------------------------------------------------
# Simulations & Comparative Analysis
# --------------------------------------------------------------------------

# 1. CONSOLIDATE STRATEGY WEIGHTS
# Combine the weight vectors from our different optimizations into a single DataFrame.
# This makes it easy to perform matrix multiplication across the return series.
weights_mat = pd.concat({
    'Mean-Variance': weights_mv,
    'Least-Squares': weights_ls
}, axis=1)

# 2. CALCULATE PORTFOLIO RETURNS
# Using the dot product (@) to compute the daily returns for each strategy:
# R_portfolio = Returns_Matrix * Weights_Vector
sim = return_series @ weights_mat

# 3. APPEND BENCHMARK DATA
# Add the target benchmark returns to the simulation for a 'baseline' comparison.
sim['Benchmark'] = data['bm_series']

# Clean the data by removing periods where all strategies have missing values
sim.dropna(how='all', inplace=True)

# 4. VISUALIZE LOG-CUMULATIVE RETURNS
# We use log-returns for the plot as they are additive and better 
# represent the relative growth rates (geometric compounding) over time.
plt.figure(figsize=(12, 7))
np.log((1 + sim).cumprod()).plot(ax=plt.gca(), linewidth=2)

# 5. ANNOTATE THE FINAL COMPARISON
plt.title('Strategy Comparison: Log-Cumulative Growth', fontsize=14)
plt.ylabel('Log-Wealth Index', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(title='Strategy', loc='upper left')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()