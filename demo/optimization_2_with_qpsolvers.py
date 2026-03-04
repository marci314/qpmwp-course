############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 2 - USING LIBRARY QPSOLVERS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     26.01.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Install qpsolvers
# uv pip install qpsolvers[open_source_solvers]     # If this fails, try: uv pip install qpsolvers, then install solvers separately (e.g. uv pip install cvxopt)




# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qpsolvers






# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

# Load msci country index return series

path_to_data = 'C:/Users/marce/OneDrive - Universität Zürich UZH/YEAR 202526/QUANTITATIVE PORTFOLIO MANAGEMENT WITH PYTHON/qpmwp-course/data/'
# N = 24
N = 10
df = pd.read_csv(os.path.join(path_to_data, 'msci_country_indices.csv'),
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')
series_id = df.columns[0:N]
return_series = df[series_id]

# Create 'level' series from return series
level_series = (1 + return_series).cumprod()

# # Alternatively, compute returns from level series
# returns = level_series.pct_change(1).dropna()

# Visualization

# --- Figure 1: Daily Returns ---
# We create the figure and specific axes (ax1) to avoid blank windows.
fig1, ax1 = plt.subplots(figsize=(10, 4)) 
return_series.plot(ax=ax1) 
ax1.set_title("MSCI Country Index Daily Returns")
ax1.grid(True)
plt.show() # This "finalizes" and displays the first window

# --- Figure 2: Log-Wealth Levels ---
# We create a new figure and second set of axes (ax2).
fig2, ax2 = plt.subplots(figsize=(10, 4))
# We plot the log-transformed levels directly onto ax2.
np.log(level_series).plot(ax=ax2, alpha=1, legend=True)
ax2.set_title("Log-Cumulative Returns (Wealth Levels)")
ax2.grid(True)
plt.show() # This displays the second window





# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix (using sample mean and covariance)
# --------------------------------------------------------------------------

scalefactor = 1  # could be set to 252 (trading days) for annualized returns


# Expected returns

##  This would be wrong:
##  mu = X.mean()

## This is correct:
mu = np.exp(np.log(1 + return_series).mean(axis=0) * scalefactor) - 1

# Covariance matrix
covmat = return_series.cov() * scalefactor


mu, covmat



# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


# We represent the portfolio domain with the form
# P = {x | Gx <= h, Ax = b, lb <= x <= ub}


# Lower and upper bounds
lb = np.zeros(covmat.shape[0])
# ub = np.repeat(0.2, N)
ub = np.repeat(1, N)

lb, ub


# Budget constraint
A = np.ones((1, N))
b = np.array(1.0)

A, b


# LInear inequality constraints
G = np.zeros((2, N))
G[0, 0:5] = 1
G[1, 5:10] = 1
# h = np.array([0.5, 0.5])
h = np.array([1, 1])

G, h





# --------------------------------------------------------------------------
# Solve for the mean-variance optimal portfolio with fixed risk aversion
# --------------------------------------------------------------------------

# Reference: https://qpsolvers.github.io/qpsolvers/quadratic-programming.html

# 1. Scale the covariance matrix by the risk aversion parameter
# In QP form, the objective is (1/2) * x' * P * x + q' * x
risk_aversion = 1
P_matrix = covmat.to_numpy() * risk_aversion

# 2. Define problem and solve
# We pass the matrices directly. Note: q is -mu because qpsolvers MINIMIZES.
# Maximizing returns is the same as minimizing negative returns.
problem = qpsolvers.Problem(
    P = P_matrix,
    q = mu.to_numpy() * -1, 
    G = G,   # Inequality matrix (Gx <= h)
    h = h,   # Inequality vector
    A = A,   # Equality matrix (Ax = b)
    b = b,   # Equality vector
    lb = lb, # Lower bounds
    ub = ub  # Upper bounds
)

# 3. Call the solver
# 'cvxopt' is a reliable open-source interior-point solver for QP problems.
solution = qpsolvers.solve_problem(
    problem = problem,
    solver = 'cvxopt',
    initvals = None,
    verbose = False,
)

# 4. Inspect the solution object
# solution.x contains the optimal weights
# solution.obj contains the final value of the objective function
print(f"Solver found a solution: {solution.found}")
print(f"Primal Residual: {solution.primal_residual()}") # Should be very small (near 0)
print(f"Duality Gap: {solution.duality_gap()}")         # Measure of optimality

# 5. Extract weights and Convert to Series
# We map the raw numpy array back to the country index names for clarity.
weights_mv = pd.Series(solution.x, index=return_series.columns)

# --------------------------------------------------------------------------
# Corrected Plotting
# --------------------------------------------------------------------------
# We use subplots and ax to ensure the plot renders correctly in the terminal.
fig3, ax3 = plt.subplots(figsize=(10, 4))
weights_mv.plot(kind='bar', ax=ax3, color='skyblue', edgecolor='black')

ax3.set_title("Optimal Mean-Variance Weights (QPSolvers)")
ax3.set_ylabel("Portfolio Weight")
ax3.set_xlabel("Country Index")
ax3.axhline(0, color='black', linewidth=0.8) # Reference line at 0
ax3.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()





# --------------------------------------------------------------------------
# Solve for the Global Minimum-Variance (GMV) portfolio
# --------------------------------------------------------------------------

# 1. Define problem and solve
# To find the minimum variance, we set the linear objective 'q' to zero.
# This tells the solver to ignore the 'mu' (returns) entirely.
# The goal is simply to minimize: (1/2) * x' * P * x
problem = qpsolvers.Problem(
    P = covmat.to_numpy(),
    q = (mu * 0).to_numpy(), # q = 0: Reward is ignored, only risk matters
    G = G,                   # Group constraints (e.g., max 50% in first 5 assets)
    h = h,
    A = A,                   # Budget constraint (sum of weights = 1.0)
    b = b,
    lb = lb,                 # Lower bounds (No short-selling)
    ub = ub                  # Upper bounds
)

# 2. Call the solver
solution = qpsolvers.solve_problem(
    problem = problem,
    solver = 'cvxopt',
    initvals = None,
    verbose = False,
)

# 3. Extract and label the weights
# We map the numerical array solution back to the index names (countries).
weights_minv = pd.Series(solution.x, index=return_series.columns)

# --------------------------------------------------------------------------
# Corrected Plotting
# --------------------------------------------------------------------------
# Using the subplots/ax method to ensure the plot renders in the terminal.
fig4, ax4 = plt.subplots(figsize=(10, 4))
weights_minv.plot(kind='bar', ax=ax4, color='teal', edgecolor='black')

ax4.set_title("Minimum-Variance Optimal Weights (Risk Only)")
ax4.set_ylabel("Portfolio Weight")
ax4.set_xlabel("Country Index")
ax4.axhline(0, color='black', linewidth=0.8)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()






# --------------------------------------------------------------------------
# Efficient Frontier
# Solve a sequence of mean-variance optimal portfolios
# with varying risk aversion parameters
# --------------------------------------------------------------------------

# 1. Define a grid of risk aversion parameters (lambda)
# We move from 0 (total risk indifference) to 20 (extremely risk-averse).
risk_aversion_grid = np.linspace(0, 20, 100)

# Prepare an empty dict to store weights for each scenario
weights_dict = {}

# 2. Loop over the grid
# For each value in the grid, we solve a new optimization problem.
for risk_aversion in risk_aversion_grid:

    # Define the problem: minimize (-Expected Return + 0.5 * Risk_Aversion * Variance)
    problem = qpsolvers.Problem(
        P = (covmat * risk_aversion).to_numpy(),
        q = mu.to_numpy() * -1, 
        G = G,
        h = h,
        A = A,
        b = b,
        lb = lb,
        ub = ub
    )

    # Solve using the CVXOPT solver
    solution = qpsolvers.solve_problem(
        problem = problem,
        solver = 'cvxopt',
        initvals = None,
        verbose = False,
    )

    # Map raw weights to country labels and store in the dictionary
    weights = pd.Series(solution.x, index=return_series.columns)
    weights_dict[risk_aversion] = weights

# 3. Convert results to a DataFrame
# Rows = Risk Aversion level, Columns = Country Weights
weights_df = pd.DataFrame(weights_dict).T
weights_df.index.name = 'risk_aversion'

# --------------------------------------------------------------------------
# Calculate Performance and Plot Efficient Frontier
# --------------------------------------------------------------------------

# Calculate Portfolio Volatility: sigma_p = sqrt(w' * Cov * w)
# We use np.diag to extract the variance of each specific portfolio along the grid
portf_vola = np.sqrt(np.diag(weights_df @ covmat @ weights_df.T))

# Calculate Portfolio Expected Return: mu_p = w' * mu
portf_return = weights_df @ mu

# Calculate Sharpe Ratio (Return / Risk) for the color mapping
sharpe_ratio = portf_return / portf_vola

# Corrected Plotting
fig5, ax5 = plt.subplots(figsize=(10, 6))
scatter = ax5.scatter(portf_vola, portf_return, c=sharpe_ratio, cmap='viridis', marker='o')

# Add labels and styling
ax5.set_title("The Efficient Frontier")
ax5.set_xlabel("Portfolio Volatility (Risk)")
ax5.set_ylabel("Portfolio Expected Return")
ax5.grid(True, linestyle=':', alpha=0.6)

# Add a colorbar to show the Sharpe Ratio
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Sharpe Ratio (Return/Risk)')

plt.show()

# --------------------------------------------------------------------------
# Find the optimal risk aversion and backtest performance
# --------------------------------------------------------------------------

# 1. Identify the Risk Aversion parameter that maximizes the Sharpe Ratio (SR)
sr = portf_return / portf_vola
idx_max = sr.idxmax()

fig6, ax6 = plt.subplots(figsize=(10, 4))
sr.plot(ax=ax6, title="Sharpe Ratio across Risk Aversion Levels")
ax6.axvline(idx_max, color='red', linestyle='--', label=f'Max SR at lambda = {idx_max:.2f}')
ax6.set_xlabel("Risk Aversion (lambda)")
ax6.set_ylabel("Sharpe Ratio")
ax6.legend()
plt.show()

# 2. Simulate historical cumulative returns for all portfolios on the frontier
# We multiply the daily returns by the weights of every portfolio found in the loop
sim = return_series @ weights_df.T

# Create a new plot for cumulative performance
fig7, ax7 = plt.subplots(figsize=(10, 6))

# --------------------------------------------------------------------------
# Cleaned Performance Comparison Plot
# --------------------------------------------------------------------------

# 1. Setup Figure
fig7, ax7 = plt.subplots(figsize=(12, 6))

# 2. Plot the 100 frontier portfolios ("Spaghetti")
# We use legend=False here to ensure those 100 decimal labels stay hidden.
np.log((1 + sim).cumprod()).plot(ax=ax7, legend=False, alpha=0.15, cmap='viridis')

# 3. Add the 3 Named Portfolios
# We explicitly do NOT pass legend=False here so the 'label' is registered for the final legend.
np.log((1 + return_series @ weights_mv).cumprod()).plot(
    ax=ax7, label='Target Mean-Variance (Black)', linewidth=2.5, color='black'
)
np.log((1 + return_series.mean(axis=1)).cumprod()).plot(
    ax=ax7, label='Equally Weighted 1/N (Red)', linewidth=2, linestyle='--', color='red'
)
np.log((1 + return_series @ weights_minv).cumprod()).plot(
    ax=ax7, label='Minimum-Variance (Blue)', linewidth=2.5, color='blue'
)

# 4. Final Styling
ax7.set_title("Historical Performance Comparison (Log-Scale)")
ax7.set_ylabel("Log-Cumulative Return")
ax7.grid(True, alpha=0.3)

# 5. The Clean Legend (Only shows the 3 labeled portfolios)
ax7.legend(loc='upper left', frameon=True)

# 6. The Risk Aversion Colorbar
# This explains the gradient of the 'spaghetti' lines without using text labels.
sm = plt.cm.ScalarMappable(cmap='viridis', 
                           norm=plt.Normalize(vmin=risk_aversion_grid.min(), 
                                              vmax=risk_aversion_grid.max()))
sm._A = [] 
cbar = fig7.colorbar(sm, ax=ax7, pad=0.02)
cbar.set_label('Risk Aversion (lambda) - Higher is Safer')

plt.tight_layout()
plt.show()





# --------------------------------------------------------------------------
# Solve for the minimum tracking error portfolio, setup as a Least Squares problem
# --------------------------------------------------------------------------

# EXPLANATION:
# Tracking Error (TE) is the volatility of the difference between portfolio 
# returns (Rw) and benchmark returns (y). To minimize TE, we solve a 
# Least Squares problem: min ||Rw - y||^2. 
#
# Matching realized volatility alone isn't enough; we need synchronization. 
# If a portfolio has the same volatility as the index but moves in the opposite 
# direction, the Tracking Error would be massive. By minimizing the squared 
# return differences, we force the portfolio to "shadow" the benchmark's 
# timing and direction, effectively driving the active risk toward zero.


# See: https://qpsolvers.github.io/qpsolvers/least-squares.html


# # Load msci world index return series
y = pd.read_csv(f'{path_to_data}NDDLWI.csv',
                 index_col=0,
                 header=0,
                 parse_dates=True,
                 date_format='%d-%m-%Y')


# 1. Define the target 'y' (The Benchmark)
# We want our portfolio returns (Rw) to stay as close to y as possible.
# Here, we use the equally weighted average of all assets as the index to track.
#y = return_series.mean(axis=1)

# 2. Derive QP Coefficients
# We are minimizing the squared error: f(w) = ||Rw - y||^2
# Algebraically expanded: f(w) = w'(R'R)w - 2(R'y)'w + y'y
#
# To translate this into the QP Standard Form: 1/2 * w'Pw + q'w
# - We set P = 2 * (R'R) to cancel the 1/2 in the standard form.
# - We set q = -2 * (R'y) to match the expanded linear term.
# - The constant term (y'y) is ignored as it doesn't change the optimal weights.

# P matrix captures the variance/covariance structure of the assets.
P = 2 * (return_series.T @ return_series)

# q vector captures the correlation/co-movement between assets and the benchmark.
q = -2 * return_series.T @ y

# 3. Define and Solve
# We pass our derived P and q into the solver alongside existing constraints.
problem = qpsolvers.Problem(
    P = P.to_numpy(),
    q = q.to_numpy(),
    G = G, h = h, A = A, b = b, lb = lb, ub = ub
)

solution = qpsolvers.solve_problem(
    problem = problem,
    solver = 'cvxopt',
    verbose = False
)

# 4. Extract weights
# weights_ls represents the portfolio that most closely clones the benchmark.
weights_ls = pd.Series(solution.x, index=return_series.columns)

# Visualization
fig8, ax8 = plt.subplots(figsize=(10, 4))
weights_ls.plot(kind='bar', ax=ax8, color='orange', edgecolor='black')
ax8.set_title("Minimum Tracking Error Portfolio Weights")
ax8.set_ylabel("Weight")
ax8.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# --------------------------------------------------------------------------
# Backtest and Compare Strategy Performance
# --------------------------------------------------------------------------

# 1. Calculate historical daily returns for each strategy
# The '@' operator performs matrix multiplication (returns * weights)
sim_mv = (return_series @ weights_mv).rename('Mean-Variance Portfolio')
sim_ls = (return_series @ weights_ls).rename('Min Tracking Error (Least Squares)')

# 2. Combine with the benchmark into a single DataFrame for comparison
sim = pd.concat({
    'Benchmark': y,
    'Mean-Variance': sim_mv,
    'Least-Squares': sim_ls,
}, axis=1).dropna()

# --------------------------------------------------------------------------
# Cleaned Performance Comparison Plot
# --------------------------------------------------------------------------

# 1. Setup Figure
fig7, ax7 = plt.subplots(figsize=(12, 6))

# 2. Plot the 100 frontier portfolios ("Spaghetti")
# 'legend=False' is CRITICAL here to prevent the "ObjectNotFound" / Layout error
np.log((1 + sim).cumprod()).plot(ax=ax7, legend=False, alpha=0.15, cmap='viridis')

# 3. Add the 3 Main Named Portfolios
# We give these specific labels so they are the ONLY ones in the legend
np.log((1 + return_series @ weights_mv).cumprod()).plot(
    ax=ax7, label='Target Mean-Variance (Black)', linewidth=2.5, color='black'
)
np.log((1 + return_series.mean(axis=1)).cumprod()).plot(
    ax=ax7, label='Equally Weighted 1/N (Red)', linewidth=2, linestyle='--', color='red'
)
np.log((1 + return_series @ weights_minv).cumprod()).plot(
    ax=ax7, label='Minimum-Variance (Blue)', linewidth=2.5, color='blue'
)

# 4. Final Styling
ax7.set_title("Historical Performance Comparison (Log-Scale)")
ax7.set_ylabel("Log-Cumulative Return")
ax7.grid(True, alpha=0.3)

# 5. The Clean Legend
# This will now only show the 3 lines we labeled above
ax7.legend(loc='upper left', frameon=True)

# 6. The Risk Aversion Colorbar
# This provides the scale for the background 'spaghetti' without the text mess
sm = plt.cm.ScalarMappable(cmap='viridis', 
                           norm=plt.Normalize(vmin=risk_aversion_grid.min(), 
                                              vmax=risk_aversion_grid.max()))
sm._A = [] 
cbar = fig7.colorbar(sm, ax=ax7, pad=0.02)
cbar.set_label('Risk Aversion (lambda) - Brighter is Safer')

# Use plt.show() without tight_layout if it keeps throwing warnings
plt.show()