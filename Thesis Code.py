#!/usr/bin/env python
# coding: utf-8

# In[545]:


""" Load Relevant Packages """ 

import sys
sys.path.insert(0,'/Users/christoffer/Desktop/CBS/Cand.merc Finance & Investments/3. Semester/Python For The Financial Economist')

"""
Magic commands
"""



"""
Python packages
"""

import numpy as np
import seaborn as sns
from scipy import stats, optimize
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy
from typing import Union


# In[617]:


# Load Data

ret = pd.read_excel('/Users/christoffer/Desktop/CBS/Cand.merc Finance & Investments/Thesis/Kodning/data.xlsx')
returns = ret.set_index('date')


# In[618]:


# colors for factors

cmap = plt.get_cmap('jet')
ind_colors = cmap(np.linspace(0, 1, 153))


# # Define inputs for calculations

# In[619]:


"""Define inputs"""

# number of periods
num_periods = len(ret)

# number of factors
num_assets = 153

# window size 
window_size =  12 * 5 # 5 years of monthly observations 

# effective number of periods
eff_num_periods = num_periods - window_size

# half-life
half_life = 60

# time points
time_points = np.arange(1, window_size + 1)

# exponential probabilities 
def calculate_exponential_decay_probabilities(target_time_point: Union[int, float], time_points: np.ndarray,
                                              half_life: Union[float, int]) -> np.ndarray:
    """
    Calculates exponential decay probabilities for an array of time points based on a target time point and a half life.
    Parameters
    ----------
    target_time_point:
        The target time point.
    time points:
        The array of time points to calculate probabilities for.
    half_life:
        The half life of the exponential decay.
    Returns
    -------
    Exponential decay probabilities.
    """
    
    numerator = np.exp(-np.log(2) / half_life * np.clip(target_time_point - time_points, 0, np.inf))
    denominator = np.sum(numerator)

    p_t = numerator / denominator

    return p_t

exp_probs = calculate_exponential_decay_probabilities(window_size, time_points, half_life)

# equally weighted portfolio
w_eq = np.repeat(1.0 / num_assets, num_assets)
weights_eq = np.array(w_eq).astype(float)


# # Calculate Risk Contributions for eq_weigths

# In[620]:


def calculate_cov_mat(x: np.ndarray, probs: np.ndarray, axis: int = 0) -> np.ndarray:

    """
    Estimates a covariance matrix based on a historical dataset and a set of probabilities.
    Parameters
    ----------
    x:
        The dataset to estimate covariance for.
    probs:
        The probabilities to weight the observations of the dataset by.
    axis:
        The axis to estimate over.
    Returns
    -------
    np.ndarray
        The estimated covariance matrix.
    """

    x = x.T if axis == 1 else x

    expected_x_squared = np.sum(probs[:, None, None] * np.einsum('ji, jk -> jik', x, x), axis=0)
    mu = probs @ x
    mu_squared = np.einsum('j, i -> ji', mu, mu)
    cov_mat = expected_x_squared - mu_squared

    return cov_mat


# In[621]:


rel_risk_contribs = np.zeros((eff_num_periods, num_assets))

"""
Perform calculations
"""

for t in range(eff_num_periods):
    
    # covariance matrix
    cov_mat = calculate_cov_mat(returns.iloc[t: window_size + t, :].values, probs=exp_probs)
    
    # calculate relative risk constribution 
    rel_risk_contribs[t, :] = rb.calculate_risk_contributions_std(w_eq, cov_mat, scale=True)


# In[631]:


"""
Plot Risk contributions
"""

fig, ax = plt.subplots(figsize=(28, 16))

ax.stackplot(returns.index[window_size:],
              rel_risk_contribs.T,
              edgecolor="black",
              labels=returns.columns,
              colors=ind_colors);

ax.set_ylabel('Relative Risk Contribution')
ax.set_title("Relative Risk Contributions for All 153 factors");
ax.set_ylim(0,1)
ax.legend(ncol=10, bbox_to_anchor=(1.01, -0.05));


# # Calculate Diversification Ratio

# In[623]:


"""
Define function for portfolio std and var
"""

def calculate_portfolio_variance(weights: np.ndarray, cov_mat: np.ndarray) -> float:

    return weights @ cov_mat @ weights


def calculate_portfolio_std(weights: np.ndarray, cov_mat: np.ndarray) -> float:

    return np.sqrt(calculate_portfolio_variance(weights, cov_mat))


# In[624]:


"""
Define function to calculate the diversification ratio of Yves Choueifaty and Yves Coignard (2008)
"""

def calculate_cc_ratio(weights: np.ndarray, cov_mat: np.ndarray):

    port_std = calculate_portfolio_std(weights=weights, cov_mat=cov_mat)

    vol_vec = np.sqrt(np.diag(cov_mat))
    avg_std = np.inner(weights, vol_vec)

    return avg_std / port_std


# In[625]:


cc_ratios = np.zeros(eff_num_periods)
avg_corr = np.zeros(eff_num_periods)

upper_tri_idx = np.triu_indices(num_assets)

"""
Perform calculations
"""

for t in range(eff_num_periods):
    
    # covariance matrix
    cov_mat = calculate_cov_mat(returns.iloc[t: window_size + t, :].values, probs=exp_probs)
    
    # average correlation
    avg_corr[t] = cov_to_corr_matrix(cov_mat)[upper_tri_idx].flatten().mean()
    
    # calculate relative risk constribution 
    cc_ratios[t] = calculate_cc_ratio(w_eq, cov_mat)


# In[627]:


"""
Plot Diversification Ratio and Average Correlation
"""

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(returns.index[window_size:], cc_ratios, label="Diversification Ratio")
ax.set_ylabel("Diversification ratio")
ax.set_title("Diversification ratio: All 153 Factor Portfolios (eq. weighted)");
ax.legend(loc='upper left')
ax_new = ax.twinx()
ax.set_ylim(1.5,7)
ax_new.set_ylim(0,0.3)

ax_new.plot(returns.index[window_size:], avg_corr, label="avg. correlation", color="gray")
ax_new.legend(loc='upper right')
ax_new.grid(None)


# # Most Diversified Portfolio

# In[628]:


"""
Define function for long only portfolio
"""

def calculate_most_diversified_portfolio(cov_mat: np.ndarray, init_weights=None) -> np.ndarray:
    
    # define intial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)
    
    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # perform optimization
    res = optimize.minimize(lambda x: -calculate_cc_ratio(x, cov_mat), init_weights,
                            constraints=[eq_constraint,], bounds=[(0, 1)]*n)
    
    return res.x


# In[629]:


"""
Perform calculations for MDP
"""

most_div_port_weights = np.zeros((eff_num_periods, num_assets))
most_div_port_cc = np.zeros(eff_num_periods)


for t in range(eff_num_periods):
    
    # covariance matrix
    cov_mat = calculate_cov_mat(returns.iloc[t: window_size + t, :].values, probs=exp_probs)
    
    # most diversified
    most_div_port_weights[t, :] = calculate_most_diversified_portfolio(cov_mat)
    most_div_port_cc[t] = calculate_cc_ratio(most_div_port_weights[t, :], cov_mat)

# store in data-frame
df_most_div_port_weights = pd.DataFrame(data=most_div_port_weights,
                                        index=returns[window_size:].index,
                                        columns=returns.columns)


# In[643]:


""" 
Plot MDP Weights
"""

fig, ax = plt.subplots(figsize=(30, 12))

ax.stackplot(returns.index[window_size:],
              most_div_port_weights.T,
              edgecolor="black",
              labels=returns.columns,
              colors=ind_colors);

ax.set_ylabel('Sector weights')
ax.set_title("Most diversfied portfolios including all 153 factors");
ax.legend(ncol=7, bbox_to_anchor=(1.01, -0.05));
ax.set_ylim(0,1)


# # Minimum Variance Portfolio

# In[598]:


"""
Define function for minimum variance portfolio
"""

def calculate_min_var_portfolio(cov_mat: np.ndarray, init_weights=None) -> np.ndarray:
    
    # define intial values
    n = cov_mat.shape[0]
    if init_weights is None:
        init_weights = np.repeat(1.0 / n, n)
    
    # define sum to one constraint
    eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # perform optimization
    res = optimize.minimize(lambda x: calculate_portfolio_variance(x, cov_mat)*100*100,
                            init_weights,
                            constraints=[eq_constraint,],
                            bounds=[(0, 1)]*n)
    
    return res.x


# In[599]:


"""
Perform calculations for min_var
"""

min_var_port_weights = np.zeros((eff_num_periods, num_assets))
min_var_port_cc = np.zeros(eff_num_periods)


for t in range(eff_num_periods):
    
    # covariance matrix
    cov_mat = calculate_cov_mat(returns.iloc[t: window_size + t, :].values, probs=exp_probs)
    
    # mnimum variance
    min_var_port_weights[t, :] = calculate_min_var_portfolio(cov_mat)
    min_var_port_cc[t] = calculate_cc_ratio(min_var_port_weights[t, :], cov_mat)
    
# store in data-frame
df_min_var_port_weights = pd.DataFrame(data=min_var_port_weights,
                                        index=returns[window_size:].index,
                                        columns=returns.columns)


# In[600]:


"""
Plot Minimum Variance Weights
"""

fig, ax = plt.subplots(figsize=(30, 12))

ax.stackplot(returns.index[window_size:],
              min_var_port_weights.T,
              edgecolor="black",
              labels=returns.columns,
              colors=ind_colors);

ax.set_ylabel('Sector weights')
ax.set_title("Minimum variance portfolios including all 153 factors");
ax.legend(ncol=7, bbox_to_anchor=(1.01, -0.05));


# # Diversifaction Ratio For All Portfolios

# In[ ]:


ax.plot(returns.index[window_size:], min_var_port_cc, label="DR, minimum variance")


# In[635]:


"""
Plotting Diversification Ratios
"""

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(returns.index[window_size:], cc_ratios, label="DR, eq. weighted")
ax.plot(returns.index[window_size:], most_div_port_cc, label="DR, most diversified")
ax.set_ylabel("Diversification ratio")
ax.set_title("Diversification ratios");
ax.set_ylim(2,300)
ax.legend();


# In[636]:


"""
Plot Diversification Ratio and Average Correlation
"""

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(returns.index[window_size:], most_div_port_cc, label="Diversification Ratio")
ax.set_ylabel("Diversification ratio")
ax.set_title("Diversification ratio: All 153 Factor Portfolios (MDP)");
ax.legend(loc='upper left')
ax_new = ax.twinx()
ax.set_ylim(1.5,30)
ax_new.set_ylim(0.05,0.3)

ax_new.plot(returns.index[window_size:], avg_corr, label="avg. correlation", color="gray")
ax_new.legend(loc='upper right')
ax_new.grid(None)


# # Evaluating Strategies

# In[637]:


df_port_ret = pd.DataFrame(index=returns.index[window_size:])

# monthly return
df_port_ret['min-var'] = (returns[window_size:] * df_min_var_port_weights).sum(axis=1)
df_port_ret['most-div'] = (returns[window_size:] * df_most_div_port_weights).sum(axis=1)
df_port_ret['eq-weight'] = (returns[window_size:] @ w_eq)

# total return index
df_port_tri = (1 + df_port_ret).cumprod(axis=0)


# In[638]:


"""
Plot Cummulative Returns 
"""

df_port_tri.plot(xlabel="", figsize=(12, 6));


# In[639]:


mean_return = df_port_ret.mean(axis=0)
mean_return


# In[640]:


std_return = df_port_ret.std(axis=0)
std_return


# In[641]:


ir_ratio = mean_return / std_return
ir_ratio


# In[ ]:




