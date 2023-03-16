import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load time series data into a DataFrame
data = pd.read_excel('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/Data.xlsx', index_col='date', parse_dates=True)
data = data.dropna()
print(data)

############### Dickey Fuller test ###############

# Create an empty dataframe to store the test results
results_dickeyfuller = pd.DataFrame(columns=['Stock', 'ADF Statistic', 'p-value', 'Stationary', 'AIC'])

# Loop through each stock's returns and run the Augmented Dickey-Fuller test
for col in data.columns:
    result = adfuller(data[col])
    p_val = round(result[1], 5)
    stationary = 'Yes' if p_val < 0.05 else 'No'
    model = ARIMA(data[col], order=(1,1,0)).fit()
    aic = model.aic
    stock_results = pd.DataFrame({'Stock': [col],
                                   'ADF Statistic': [result[0]],
                                   'p-value': [p_val],
                                   'Stationary': [stationary],
                                   'AIC': aic})
    results_dickeyfuller = pd.concat([results_dickeyfuller, stock_results], ignore_index=True)

# Print the results dataframe
print(results_dickeyfuller)

results_dickeyfuller.to_excel('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/Dickey_fuller_test.xlsx', index=False)

#as the the ADF statistic is large in absolute value and negative it
#indicate that the variables does not have a unit root. The P-value
#of the variables are all zero, which mean we can reject the nul
#hypothesis "that the data has a unit root" and we can reject this 
#at 0,05 og 0,01 % confidence

############### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) ###############
 
# Create an empty DataFrame to store the results
kpss_table = pd.DataFrame(columns=['Variable', 'Test Statistic', 'p-value', 'Lags', 'Stationary'])

# Loop over each variable in the data and perform the KPSS test
for col in data.columns:
    result = kpss(data[col])
    is_stationary = result[0] < result[3]['5%']
    temp_df = pd.DataFrame({
        'Variable': col,
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags': result[2],
        'Stationary': is_stationary
    }, index=[0])
    kpss_table = pd.concat([kpss_table, temp_df], ignore_index=True)

## OBS ##
#result = kpss(data[col], nlags=20) --> By this line we can adjust the number of lags to the number we want.

# Display the table
print(kpss_table)

kpss_table.to_excel('/Users/jacobhenrichsen/iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/Data/Manipulated data/KPSS.xlsx', index=False)

#If we want the values to be displayed as picture tables or anything else
kpss_table = kpss_table.sort_values(by='Test Statistic', ascending=False)

# Create a bar chart of the test statistic for each variable
plt.bar(kpss_table['Variable'], kpss_table['Test Statistic'])
plt.xticks(rotation=90)
plt.ylabel('Test Statistic')
plt.title('KPSS Test Results')

############### Augmentet Dickey-Fuller test ###############

ADF_table = pd.DataFrame(columns=['Variable', 'ADF Statistic', 'p-value', '5% CV', '1% CV'])

# Loop over each variable in the data and perform the ADF test
for col in data.columns:
    result = adfuller(data[col])
    nobs = result[3]
    k_vars = len(result[0]) - 1
    cv_1 = adfuller(data[col], regression='ct', maxlag=k_vars, autolag=None)[4]['1%']
    cv_5 = adfuller(data[col], regression='ct', maxlag=k_vars, autolag=None)[4]['5%']
    temp_df = pd.DataFrame({
        'Variable': [col],
        'ADF Test Statistic': [result[0]],
        'p-value': [result[1]],
        '5% CV': [cv_5],
        '1% CV': [cv_1]
    })
    ADF_table = pd.concat([ADF_table, temp_df], ignore_index=True)

 
# Print the results DataFrame
print(ADF_table)

############ Augmented Engle-Granger cointegration test ############

# create an empty dataframe to store the test results
results_eg = pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Test Statistic', 'p-value', 'Stationary'])

# loop through each pair of variables and run the cointegration test
for i, var1 in enumerate(data.columns[:-1]):
    for var2 in data.columns[i+1:]:
        result = coint(data[var1], data[var2])
        is_stationary = result[1] < 0.05
        temp_df = pd.DataFrame({
            'Variable 1': var1,
            'Variable 2': var2,
            'Test Statistic': result[0],
            'p-value': result[1],
            'Stationary': is_stationary
        }, index=[0])
        results_eg = pd.concat([results_eg, temp_df], ignore_index=True)

# print the results dataframe
print(results_eg)

#This will print the cointegration of each pair of stock and they
#can be stored as a heat map
#create empty dataframe for storing t-statistics
t_stats_EG_df = pd.DataFrame(index=data.columns, columns=data.columns)

# calculate t-statistics for each pair of stocks
for i in data.columns:
    for j in data.columns:
        if i != j:
            result = coint(data[i], data[j])
            t_stats_EG_df.loc[i,j] = result[0]

plt.figure(figsize=(10,8))
sns.heatmap(t_stats_EG_df.astype(float), annot=True, cmap='coolwarm')
plt.title('Engle-Granger Cointegration T-Statistics Heatmap')
plt.show()

######    The Gauss-Markov assumptions     #######

# Load the data
data1 = sm.datasets.get_rdataset("Guerry", "HistData").data

# Fit the linear regression model
model = sm.formula.ols("Lottery ~ Literacy + np.log(Pop1831)", data=data).fit()

# Check for linearity
sm.stats.diagnostic.linear_rainbow(model)

# Check for independence of errors
sm.stats.stattools.durbin_watson(model.resid)

# Check for homoscedasticity
sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)

# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
print(vif)

# Check for zero conditional mean
sm.stats.diagnostic.acorr_ljungbox(model.resid)