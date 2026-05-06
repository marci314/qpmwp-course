############################################################################
### QPMwP CODING EXAMPLES - OPTIMIZATION 1 - USING LIBRARY CVXPY
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     26.01.2026
# First version:    18.01.2025
# --------------------------------------------------------------------------




# Under Terminal, click on New Terminal.
# In the Terminal window, select Command Prompt.
# Create and activate a virtual environment, and install required packages. For that, type the following commands:

# .venv\Scripts\activate

# uv pip install ipykernel
# uv pip install pandas
# uv pip install matplotlib
# uv pip install cvxpy




# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp






# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

# Load msci country index return series

path_to_data = 'data'
# N = 24
N = 10
df = pd.read_csv(os.path.join(path_to_data, 'msci_country_indices.csv'),
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')
series_id = df.columns[0:N]
return_series = df[series_id]

# Create 'level' series from return series - how much we would have if we invested 1 unit at the beginning and let it grow with the returns
level_series = (1 + return_series).cumprod()

# # Alternatively, compute returns from level series
# returns = level_series.pct_change(1).dropna()

# Visualization

# --- Figure 1 ---
# 1. Create the figure and the 'ax' (axes) first
fig1, ax1 = plt.subplots(figsize=(10, 4)) 
return_series.plot(ax=ax1) 
ax1.set_title("Return Series")
ax1.grid(True)
plt.show()

# --- Figure 2 ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
# Use np.log on the series and plot on ax2
np.log(level_series).plot(ax=ax2, alpha=1, legend=True)
ax2.set_title("Log-Level Series")
ax2.grid(True)
plt.show()






# --------------------------------------------------------------------------
# Estimates of the expected returns and covariance matrix (using sample mean and covariance)
# --------------------------------------------------------------------------

scalefactor = 1  # could be set to 252 (trading days) for annualized returns


# Expected returns

##  This would be wrong, because we ignore the compounding effect of returns:
## If an asset goes up 50% and then down 50%, the arithmetic mean is $0\%$, but you actually lost 25% of your money ($1.5 \times 0.5 = 0.75$)
##  mu = X.mean()

## This is correct:
# Mathematical Derivation:
# 1. Total Wealth: V_T = V_0 * Product(1 + R_t)
# 2. Average daily growth (g): (1 + g)^T = Product(1 + R_t)
# 3. Linearize with log: T * ln(1 + g) = Sum( ln(1 + R_t) )
# 4. Find sample mean: ln(1 + g) = (1/T) * Sum( ln(1 + R_t) )
# 5. Scale to horizon (f): ln(1 + mu) = f * [ (1/T) * Sum( ln(1 + R_t) ) ]
# 6. Revert to simple return: mu = exp( f * mean(ln(1 + R)) ) - 1
mu = np.exp(np.log(1 + return_series).mean(axis=0) * scalefactor) - 1

# Covariance matrix
# Mathematical Derivation:
# 1. Sample Covariance: sigma_ij = [1 / (T-1)] * Sum( (Ri - mean_i) * (Rj - mean_j) )
# 2. Scaling: Variance and Covariance scale linearly with time (T). 
#    While volatility scales by sqrt(f), variance scales by (sqrt(f))^2 = f.
# 3. Formula: Cov_scaled = Cov_sample * f
covmat = return_series.cov() * scalefactor

mu, covmat

# REMARK ON ASSUMPTIONS:
# Linear scaling of mu and covmat requires the following assumptions:
# 1. I.I.D. Returns: Returns are Independent and Identically Distributed.
# 2. Random Walk: Prices follow a random walk (no autocorrelation/trends).
# 3. Stationarity: Mean and variance remain constant over the scaling period.
# 4. No Fat Tails: Assumes risk scales without extreme "Black Swan" events.



# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------

# This section defines the "rules" of the portfolio domain P.
# Standard Form: P = {x | Gx <= h, Ax = b, lb <= x <= ub}

# 1. Lower and Upper Bounds (lb, ub)
# Defines limits for individual asset weights x_i.
lb = np.zeros(covmat.shape[0])  # lb: No Short-Selling (weights >= 0)
# ub = np.repeat(0.2, N)        # Diversification limit (max 20% per asset)
ub = np.repeat(0.2, N)            # ub: Max exposure (max 100% per asset)

# 2. Budget Constraint (A, b)
# The "Fully Invested" constraint: weights must sum to 100%.
# Equation: Ax = b  =>  1*x1 + 1*x2 + ... + 1*xN = 1.0
A = np.ones((1, N))
b = np.array(1.0)

# 3. Linear Inequality Constraints (G, h)
# Group/Sector constraints to limit regional exposure.
# Equation: Gx <= h
G = np.zeros((2, N))
G[0, 0:5] = 1   # Group 1: First 5 countries
G[1, 5:10] = 1  # Group 2: Next 5 countries
h = np.array([0.5, 0.5])  # Limit: Max 50% exposure for each group

# REMARK:
# - lb/ub: Controls individual asset concentration.
# - A/b:   Ensures the portfolio is 100% invested (capital allocation).
# - G/h:   Enforces structural/sectoral diversification.

lb, ub, A, b, G, h






# --------------------------------------------------------------------------
# Solve mean-variance optimal portfolios with cvxpy
# --------------------------------------------------------------------------

# 1. Objective function parameters
# The standard Markowitz objective is: Minimize (0.5 * x.T * P * x + q.T * x)
# where P is the risk (covariance) and q is the negative expected return.
risk_aversion = 1.0
q = mu.to_numpy() * -1             # We minimize -mu, which is the same as maximizing mu
P = covmat.to_numpy() * risk_aversion  # Covariance matrix scaled by risk aversion factor

# 2. Decision vector
# This is what the solver is looking for: an N-dimensional vector of weights.
x = cp.Variable(N, name='weights')

# 3. Constraints
# We build a list of conditions the portfolio must satisfy.
cons_list = [x >= lb, x <= ub]     # Individual asset bounds (no shorting, max exposure)
if G is not None:
    cons_list.append(G @ x <= h)   # Group/Sector constraints
if A is not None:
    cons_list.append(A @ x == b)   # Budget constraint (sum of weights = 1.0)

# 4. Objective function
# Quadratic form handles the variance (x.T * P * x), and linear handles returns (q.T * x).
obj = q @ x + 0.5 * cp.quad_form(x, P)

# 5. Finalize problem
# We define the goal: minimize the objective function subject to our constraints.
model = cp.Problem(cp.Minimize(obj), cons_list)

# 6. Solve the problem
# We use the CVXOPT solver. Verbose=False keeps the terminal output clean.
model.solve(solver=cp.CVXOPT, verbose=False)

# 7. Extract solution and objective
# Check if the solver actually found a valid point (feasibility).
if model.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError(f"Optimization failed. Status: {model.status}")

# Retrieve the numerical results
x_opt = model.variables()[0].value      # The raw optimal weights
x_opt = pd.Series(x_opt, index=mu.index) # Convert back to a labeled Series
obj_val = model.value                   # The final value of the objective function
status = model.status                   # Should be 'optimal'

# Output and Visualization
print("Optimal weights:", x_opt)
print("Optimal objective value:", obj_val)
print("Solver status:", status)

x_opt.plot(kind='bar', title="Optimal Portfolio Weights")
plt.ylabel("Weights")
plt.show()

# 8. Dual variables (Lagrange multipliers)
# These represent the "Shadow Prices" or how much the objective would change 
# if a constraint were slightly relaxed.
# index 0: x >= lb
# index 1: x <= ub
# index 2: Gx <= h (if applicable)
model.constraints[0].dual_value  # Sensitivity to the lower bound
model.constraints[1].dual_value  # Sensitivity to the upper bound
model.constraints[2].dual_value  # Sensitivity to the group/budget constraints







