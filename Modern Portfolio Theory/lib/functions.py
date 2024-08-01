import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro
import matplotlib.pyplot as plt

def eda_stats(df):
    # Average annualized return assuming 252 traiding days in a year
    print("Average annualized return:", ((1+np.mean(df))**252)-1)
    # Annualized volatility
    print("Annualized volatility (std):", np.std(df) * np.sqrt(252))
    # Skewness of the distribution
    print("Skewness:", skew(df.dropna()))
    # Calculate excess kurtosis (for normal kurtosis add 3 to result). Excess kurtosis greater than 0 means non-normality of dist. Higher kurtois means higher risk.
    print("Excess kurtosis:", kurtosis(df.dropna()))
    print("")
    print("Shapiro-Wilk test for normality")
    # testing for normality
    # if kurtosis is greater than 3 and skewness is non zero, distribution is probably not normal.
    # To estimate probability of normal dist use Shapiro-Wilk test
    p_value = shapiro(df.dropna())[1]
    if p_value <= 0.05:
        print("Null hypothesis of normality is rejected.")
    else:
        print("Null hypothesis of normality is accepted.")
        
def daily_returns_dist_and_ts(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('{name} daily returns distribution and time series'.format(name=df.name.capitalize()))
    ax1.hist(df, bins=75, density=False)
    ax2.plot(df.index, df)