import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro
import matplotlib.pyplot as plt
import seaborn as sns

def eda_stats(df):
    for column in df.columns:
        print("-------------------------------------------")
        print("-- Statistics for {}".format(df[column].name))
        print("-------------------------------------------")
        # Average annualized return assuming 252 traiding days in a year
        print("Average annualized return:", ((1+np.mean(df[column]))**252)-1)
        # Annualized volatility
        print("Annualized volatility (std):", np.std(df[column]) * np.sqrt(252))
        # Skewness of the distribution
        print("Skewness:", skew(df[column].dropna()))
        # Calculate excess kurtosis (for normal kurtosis add 3 to result). Excess kurtosis greater than 0 means non-normality of dist. Higher kurtois means higher risk.
        print("Excess kurtosis:", kurtosis(df[column].dropna()))
        print("Shapiro-Wilk test for normality")
        # testing for normality
        # if kurtosis is greater than 3 and skewness is non zero, distribution is probably not normal.
        # To estimate probability of normal dist use Shapiro-Wilk test
        p_value = shapiro(df[column].dropna())[1]
        if p_value <= 0.05:
            print("Null hypothesis of normality is rejected.")
        else:
            print("Null hypothesis of normality is accepted.")
        print("")

def corr_matrix_heatmap(correlation_matrix):
    # Create a heatmap
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap="YlGnBu", 
                linewidths=0.3,
                annot_kws={"size": 8})

    # Plot aesthetics
    plt.xticks(rotation=90)
    plt.yticks(rotation=0) 
    plt.show()

def daily_returns_dist_and_ts(df):
    # Iterating through all columns in the DataFrame
    for column in df.columns:
        # Creating a new figure with two axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        
        # Histogram for the current column
        ax1.hist(df[column], bins=75, density=False)
        ax1.set_title(f'Daily returns distribution of {column}')
        
        # Line plot for the current column
        ax2.plot(df.index, df[column])
        ax2.set_title(f'Daily returns time series of {column}')
        plt.xticks(rotation=90)
        
        # Setting a common title and displaying the plots
        fig.suptitle(f'Column: {column}')
        plt.tight_layout()
        plt.show()

def plot_volatility_and_returns(returns):
    # Calculating volatility and expected annual returns
    volatility = returns.std() * np.sqrt(252)
    expected_returns = returns.mean() * 252
    
    # Creating a subplot with two axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Horizontal bar chart for volatility
    volatility.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Volatility (Standard Deviation)')
    ax1.set_xlabel('Volatility')
    
    # Horizontal bar chart for expected annual returns
    expected_returns.plot(kind='barh', ax=ax2, color='lightgreen')
    ax2.set_title('Expected Annual Returns')
    ax2.set_xlabel('Expected Returns')
    
    # Setting a common title and layout
    fig.suptitle('Volatility and Expected Annual Returns of Assets', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_cumulative_returns(returns, weights, columns_to_plot, ax):
    """
    Plots the cumulative returns of selected assets and an equal-weighted portfolio.

    Parameters:
    - returns: DataFrame containing the returns of individual assets.
    - weights: Array or list of weights for the portfolio.
    - columns_to_plot: List of columns from the returns DataFrame to include in the plot.
    """
    
    # Ensure the weights are a numpy array
    weights = np.array(weights)
    
    # Make a copy of the returns DataFrame to avoid modifying the original data
    returns_copy = returns.copy()
    
    # Calculate the equal-weighted portfolio returns
    returns_copy['portfolio'] = returns_copy.dot(weights)
    
    # Selecting columns to plot, including the portfolio
    selected_columns = ['portfolio'] + columns_to_plot
    
    # Calculating cumulative returns
    cum_returns = ((1 + returns_copy[selected_columns]).cumprod() - 1)
    
    # Plotting cumulative returns
    # cum_returns.plot(figsize=(10, 5))
    cum_returns.plot(ax=ax)
    ax.set_title('Cumulative Returns of Selected Assets and Portfolio')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Returns')
    ax.legend(loc='best')

def plot_drawdowns(prices, weights, window, ax):
    """
    Plots the daily drawdown and maximum daily drawdown in a given time window for a portfolio.

    Parameters:
    - prices: DataFrame containing the price data of individual assets.
    - weights: Array or list of weights for the portfolio.
    - window: The time window for calculating the rolling maximum and drawdown (default is 250 days).
    """
    
    # Ensure the weights are a numpy array
    weights = np.array(weights)
    
    # Make a copy of the prices DataFrame to avoid modifying the original data
    prices_copy = prices.copy()
    
    # Calculate the equal-weighted portfolio price series
    prices_copy['portfolio'] = prices_copy.dot(weights)
    pf_price = prices_copy['portfolio']
    
    # Calculate the rolling maximum price
    roll_max = pf_price.rolling(min_periods=1, window=window).max()
    
    # Calculate the daily drawdown
    daily_drawdown = pf_price / roll_max - 1.0
    
    # Calculate the maximum daily drawdown in the given time window
    max_daily_drawdown = daily_drawdown.rolling(min_periods=1, window=window).min()
    
    # Plot the drawdowns
    # ax.figure(figsize=(10, 5))
    ax.plot(daily_drawdown.index, daily_drawdown, label='Daily drawdown')
    ax.plot(max_daily_drawdown.index, max_daily_drawdown, label='Maximum daily drawdown in time-window')
    ax.set_title('Daily Drawdown and Maximum Daily Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend()

def plot_portfolio_allocation(portfolio_weights, ax):
    """
    Plots a pie chart of the investment portfolio allocation with specific color mapping.
    
    Parameters:
    - portfolio_weights: pandas Series where index represents asset names and values represent weights.
    
    Output:
    - Displays a pie chart with the given portfolio allocation.
    """
    
    # Define color mapping: green shades for bonds, orange shades for stocks, and yellow for gold
    colors = ['lightgreen', 'green', 'yellow', 'orange', 'darkorange']
    
    # Create a pie chart
    # ax.figure(figsize=(8, 4))
    ax.pie(portfolio_weights, labels=portfolio_weights.index, autopct='%1.1f%%', colors=colors, startangle=140)
    
    # Set the title
    ax.set_title('Portfolio Allocation')

def display_summary(ax, returns, weights):
    # Ensure the weights are a numpy array
    weights = np.array(weights)
    # Calculate the covariance matrix
    cov_mat = returns.cov()
    # Annualize the co-variance matrix
    cov_mat_annual = cov_mat*252
    # Calculate the portfolio standard deviation (volatility)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat_annual, weights)))
    
    # Ensure weights is a pandas Series for proper indexing
    if isinstance(weights, list):
        weights = pd.Series(weights, index=returns.columns)
    # Calculate daily portfolio return as the dot product of daily returns and weights
    daily_portfolio_return = returns.dot(weights)
    # Calculate cumulative returns of the portfolio
    cumulative_returns = (1 + daily_portfolio_return).cumprod()
    # Get the final cumulative return
    final_value = cumulative_returns.iloc[-1]
    # Calculate the number of years (assuming daily returns and 252 trading days per year)
    num_days = len(daily_portfolio_return)
    num_years = num_days / 252
    # Calculate CAGR
    cagr = (final_value ** (1 / num_years)) - 1
    
    # t = relativedelta.relativedelta(returns.index.max(), returns.index.min()).years
    # cagr = round(cum_returns.portfolio[-1] ** (1 / t) - 1, 4)
    
    # Calculate average annual return
    avg_annual_return = returns.dot(weights).mean() * 252
    
    # Display the text on the empty axis
    ax.text(0.5, 0.5, 
            f'CAGR: {cagr:.2%}\nAvg Annual Return: {avg_annual_return:.2%}\nVolatility: {portfolio_volatility:.2%}',
            fontsize=12, 
            ha='center', 
            va='center',
            transform=ax.transAxes)
    
    # Remove axis lines and ticks
    ax.axis('off')

def combine_plots(portfolio_weights, returns, weights, columns_to_plot, prices, window=252):
    # Create a 2x2 grid layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Plot portfolio allocation on the left
    plot_portfolio_allocation(portfolio_weights, ax=axs[0, 0])
    
    # Plot cumulative returns on the top right
    plot_cumulative_returns(returns, weights, columns_to_plot, ax=axs[0, 1])
    
    # Plot drawdowns on the bottom right
    plot_drawdowns(prices, weights, window, ax=axs[1, 1])

    # Display summary on the bottom left subplot
    display_summary(ax=axs[1, 0], returns=returns, weights=weights)
    
    # Set a title for the entire figure
    fig.suptitle('Portfolio Analysis', fontsize=22)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def create_portfolio_weights(returns, weights):
    """
    Creates a pandas Series representing portfolio weights.
    
    Parameters:
    - returns (pd.DataFrame): DataFrame containing asset returns with column names as asset names.
    - weights (list or array): List or array of weights corresponding to the columns in `returns`.
    
    Returns:
    - pd.Series: Series representing the portfolio weights with asset names as the index.
    """
    # Ensure the number of weights matches the number of columns in returns
    if len(weights) != len(returns.columns):
        raise ValueError("The number of weights must match the number of assets (columns) in the returns DataFrame.")
    
    # Create the pandas Series
    portfolio_weights = pd.Series(weights, index=returns.columns)
    
    return portfolio_weights

class EfficientFrontier:
    def __init__(self, risk_free, num_portfolios, returns):
        """
        Initializes the EfficientFrontier class with the risk-free rate, number of portfolios, and returns data.
        
        Parameters:
        - risk_free (float): Risk-free rate (annualized) used for Sharpe ratio calculation.
        - num_portfolios (int): Number of portfolios to simulate.
        - returns (pd.DataFrame): DataFrame with asset returns.
        """
        self.risk_free = risk_free
        self.num_portfolios = num_portfolios
        self.returns = returns
        self.portfolios = None
        self._simulate_portfolios()
    
    def _simulate_portfolios(self):
        """
        Simulates portfolios and calculates their returns, volatility, and Sharpe ratio.
        """
        p_ret = []
        p_weights = []
        p_vol = []

        # Set number of assets and portfolios to simulate
        df = self.returns
        num_assets = len(df.columns)
        e_r = ((1 + df.mean()) ** 252) - 1
        cov_matrix = df.cov()

        # For each random portfolio, find the return and volatility
        for _ in range(self.num_portfolios):
            # Create random weights
            weights = np.random.random(num_assets)
            weights = weights / np.sum(weights)
            p_weights.append(weights)
            
            # Calculate portfolio return
            ret = np.dot(weights, e_r)
            p_ret.append(ret)
            
            # Calculate portfolio volatility
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
            ann_sd = np.sqrt(var * 252)  # Using 252 trading days for annualization
            p_vol.append(ann_sd)
        
        # Create DataFrame for portfolios
        data = {'returns': p_ret, 'volatility': p_vol}
        for counter, symbol in enumerate(df.columns.tolist()):
            data[symbol + '_weight'] = [w[counter] for w in p_weights]
        
        self.portfolios = pd.DataFrame(data)
        self.portfolios['sharpe'] = (self.portfolios['returns'] - self.risk_free) / self.portfolios['volatility']
        self.portfolios = self.portfolios[list(df.columns + '_weight') + ['returns', 'volatility', 'sharpe']]
    
    def ef_df(self):
        """
        Returns the DataFrame of simulated portfolios.
        
        Returns:
        - pd.DataFrame: DataFrame with portfolio data.
        """
        return self.portfolios

    def ef_plot(self):
        """
        Displays a scatter plot of the simulated portfolios.
        """
        # Find portfolios with lowest volatility and highest Sharpe ratio
        min_var_port = self.portfolios.loc[self.portfolios['volatility'].idxmin()]
        max_shp_port = self.portfolios.loc[self.portfolios['sharpe'].idxmax()]
        
        # Plotting the efficient frontier
        plt.figure(figsize=(10, 6))
        plt.scatter(self.portfolios['volatility'], self.portfolios['returns'], marker='o', s=10, alpha=0.3, c='blue', label='Portfolios')
        plt.scatter(min_var_port['volatility'], min_var_port['returns'], color='red', marker='*', s=500, label='GMV')
        plt.scatter(max_shp_port['volatility'], max_shp_port['returns'], color='green', marker='*', s=500, label='MSR')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

    def min_vol(self, target_volatility, sort_by='sharpe'):
        """
        Finds the portfolio with the maximum returns for a given target volatility.
        
        Parameters:
        - target_volatility (float): The target volatility value.
        
        Returns:
        - pd.Series: Portfolio weights with maximum returns for the given volatility.
        """
        # Find the portfolio with the closest volatility to the target
        # closest_volatility_idx = (self.portfolios['volatility'] - target_volatility).abs().idxmin()
        # closest_portfolio = self.portfolios.iloc[closest_volatility_idx]
        
        # Sort the portfolios by Sharpe ratio of Return
        sorted_portfolios = self.portfolios[self.portfolios['volatility'] <= target_volatility].sort_values(by=[sort_by], ascending=False)
        
        # Return the weights of this portfolio
        portfolio_weights = sorted_portfolios.iloc[0, :].drop(['returns', 'volatility', 'sharpe'])
        return portfolio_weights
    
    def gmv(self):
        """
        Finds the Global Minimum Variance portfolio weights.

        Returns:
        - pd.Series: Global Minimum Variance portfolio weights.
        """
        # Find the GMV portfolio weigths
        min_var_port = self.portfolios.loc[self.portfolios['volatility'].idxmin()]
        
        # Return the weights of this portfolio
        portfolio_weights = min_var_port.drop(['returns', 'volatility', 'sharpe'])
        return portfolio_weights
    
    def msr(self):
        """
        Finds the Maximum Sharpe Ratio portfolio weights.

        Returns:
        - pd.Series: Maximum Sharpe Ratio portfolio weights.
        """
        # Find the MSR portfolio weigths
        max_shp_port = self.portfolios.loc[self.portfolios['sharpe'].idxmax()]
        
        # Return the weights of this portfolio
        portfolio_weights = max_shp_port.drop(['returns', 'volatility', 'sharpe'])
        return portfolio_weights