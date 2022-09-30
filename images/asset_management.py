# Source 
# Portfolio optimization in finance is the technique of creating a portfolio of assets, for which your investment has the maximum return and minimum risk.
# https://pythoninvest.com/long-read/practical-portfolio-optimisation
# https://github.com/realmistic/PythonInvest-basic-fin-analysis


##############################################################################################################

# ░█████╗░░██████╗░██████╗███████╗████████╗
# ██╔══██╗██╔════╝██╔════╝██╔════╝╚══██╔══╝
# ███████║╚█████╗░╚█████╗░█████╗░░░░░██║░░░
# ██╔══██║░╚═══██╗░╚═══██╗██╔══╝░░░░░██║░░░
# ██║░░██║██████╔╝██████╔╝███████╗░░░██║░░░
# ╚═╝░░╚═╝╚═════╝░╚═════╝░╚══════╝░░░╚═╝░░░

# ███╗░░░███╗░█████╗░███╗░░██╗░█████╗░░██████╗░███████╗███╗░░░███╗███████╗███╗░░██╗████████╗
# ████╗░████║██╔══██╗████╗░██║██╔══██╗██╔════╝░██╔════╝████╗░████║██╔════╝████╗░██║╚══██╔══╝
# ██╔████╔██║███████║██╔██╗██║███████║██║░░██╗░█████╗░░██╔████╔██║█████╗░░██╔██╗██║░░░██║░░░
# ██║╚██╔╝██║██╔══██║██║╚████║██╔══██║██║░░╚██╗██╔══╝░░██║╚██╔╝██║██╔══╝░░██║╚████║░░░██║░░░
# ██║░╚═╝░██║██║░░██║██║░╚███║██║░░██║╚██████╔╝███████╗██║░╚═╝░██║███████╗██║░╚███║░░░██║░░░
# ╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝░╚═════╝░╚══════╝╚═╝░░░░░╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░
##############################################################################################################

##############################################################################################################
# Portfolio Optimization with Python using Efficient Frontier with Practical Examples
##############################################################################################################

# Portfolio optimization is the process of creating a portfolio of assets, for which your investment has the maximum return and minimum risk.
# Modern Portfolio Theory (MPT), or also known as mean-variance analysis is a mathematical process which allows the user to maximize returns for a given risk level.
# It was formulated by H. Markowitz and while it is not the only optimization technique known, it is the most widely used.

# Efficient frontier is a graph with ‘returns’ on the Y-axis and ‘volatility’ on the X-axis. 
# It shows the set of optimal portfolios that offer the highest expected return for a given risk level or the lowest risk for a given level of expected return.



##############################################################################################################
# Practical Portfolio Optimisation
##############################################################################################################
#     What? Identify an optimal split for a known set of stocks and a given investment size.
#     Why? Smart portfolio management will add a lot to the risk management of your trades: it can reduce the volatility of a portfolio, increase returns per unit of risk, and reduce the bad cases losses
#     How? Use the library PyPortfolioOpt
#         User guide: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html
#         Detailed Colab example (Mean-Variance-Optimisation): https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb
#     Plan
#         1. Prep work : imports, getting financial data, and pivot table of daily prices
#         2. Correlation matrix
#         3. PyPortfolioOpt : min volatility, max Sharpe, and min cVAR portfolios
#         4. PyPortfolioOpt : Efficient Frontier
#         5. PyPortfolioOpt : Discrete Allocation




##############################################################################################################
# 0. Prep work : imports, getting financial data, and pivot table of daily prices
##############################################################################################################

# pip install yfinance

import pandas as pd
import yfinance as yf
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


INVESTMENT = 2000 

# PTR = PetroChina Company Limited ADR
# BUD = Anheuser Busch Inbev SA (AB InBev)
# XOM = Exxon Mobil Corporation
# BA = Boeing Co
# CHTR = Charter Communications Inc
# SHOP = Shopify Inc
# NVDA = NVIDIA Corporation
# NKE = Nike Inc

TICKERS =['PTR','BUD', 'XOM', 'BA','CHTR', 'SHOP', 'NVDA', 'NKE']

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

##############################################################################################################

stocks_prices = pd.DataFrame({'A' : []})
stocks_info = pd.DataFrame({'A' : []})

for i,ticker in enumerate(TICKERS):
  print(i,ticker)
  yticker = yf.Ticker(ticker)
  
  # Get max history of prices
  historyPrices = yticker.history(period='max')
  # generate features for historical prices, and what we want to predict
  historyPrices['Ticker'] = ticker
  historyPrices['Year']= historyPrices.index.year
  historyPrices['Month'] = historyPrices.index.month
  historyPrices['Weekday'] = historyPrices.index.weekday
  historyPrices['Date'] = historyPrices.index.date
  
  # historical returns
  for i in [1,3,7,30,90,365]:
    historyPrices['growth_'+str(i)+'d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)

  # future growth 3 days  
  historyPrices['future_growth_3d'] = historyPrices['Close'].shift(-3) / historyPrices['Close']

  # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
  historyPrices['volatility'] =   historyPrices['Close'].rolling(30).std() * np.sqrt(252)

  if stocks_prices.empty:
    stocks_prices = historyPrices
  else: 
    stocks_prices = pd.concat([stocks_prices,historyPrices], ignore_index=True)


##############################################################################################################
# Check one day
filter_last_date = stocks_prices.Date==stocks_prices.Date.max()
print(stocks_prices.Date.max())

stocks_prices[filter_last_date]
print(stocks_prices[filter_last_date])


##############################################################################################################
# https://medium.com/analytics-vidhya/how-to-create-a-stock-correlation-matrix-in-python-4f32f8cb5b50
df_pivot = stocks_prices.pivot('Date','Ticker','Close').reset_index()
df_pivot.tail(5)

print(df_pivot.tail(5))



##############################################################################################################
# 1. Correlation matrix
##############################################################################################################

# https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/
df_pivot.corr()

print(df_pivot.corr())


##############################################################################################################
# Print correlation matrix
# https://seaborn.pydata.org/generated/seaborn.heatmap.html

corr = df_pivot.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True, cmap='RdYlGn')

# show the plot window
plt.show()




##############################################################################################################
# 2. PyPortfolioOpt : min volatility, max Sharpe, and min cVAR portfolios
##############################################################################################################

# User guide: https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html
# https://github.com/robertmartin8/PyPortfolioOpt

# pip install PyPortfolioOpt

import pypfopt
print(f'\n Library version: {pypfopt.__version__}')


##############################################################################################################
# https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb

from pypfopt import risk_models
from pypfopt import plotting

from pypfopt import expected_returns
from pypfopt import EfficientFrontier


##############################################################################################################
# json: for pretty print of a dictionary: https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python/44689627
import json

mu = expected_returns.capm_return(df_pivot.set_index('Date'))
# Other options for the returns values: expected_returns.ema_historical_return(df_pivot.set_index('Date'))
# Other options for the returns values: expected_returns.mean_historical_return(df_pivot.set_index('Date'))
print(f'Expected returns for each stock: {mu} \n')

S = risk_models.CovarianceShrinkage(df_pivot.set_index('Date')).ledoit_wolf()

# Weights between 0 and 1 - we don't allow shorting
ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
ef.min_volatility()
weights_min_volatility = ef.clean_weights()

print(f'Portfolio weights for min volatility optimisation (lowest level of risk): {json.dumps(weights_min_volatility, indent=4, sort_keys=True)} \n')
print(f'Portfolio performance: {ef.portfolio_performance(verbose=True, risk_free_rate=0.01305)} \n')
# Risk-free rate : 10Y TBonds rate on 21-Jul-2021 https://www.cnbc.com/quotes/US10Y





###########################
# IMPORTANT: RISK-FREE RATE
###########################
# Risk-free rate : the input should be checked and modified accordingly https://www.cnbc.com/quotes/US10Y
##############################################################################################################






pd.Series(weights_min_volatility).plot.barh(title = 'Optimal Portfolio Weights (min volatility) by PyPortfolioOpt');


# Bug found: The objective function was changed after the initial optimization. Please create a new instance instead.
# Bug solved: I have to rename the 'ef' with 'ef_2' because it was causing a bug.
ef_2 = EfficientFrontier(mu, S, weight_bounds=(0, 1))
ef_2.max_sharpe()
weights_max_sharpe = ef_2.clean_weights()


print(f'Portfolio weights for max Sharpe optimisation (highest return-per-risk): {json.dumps(weights_max_sharpe, indent=4, sort_keys=True)} \n')
print(f'Portfolio performance: {ef_2.portfolio_performance(verbose=True, risk_free_rate=0.01305)} \n')




##############################################################################################################
ef = EfficientFrontier(mu, S)
ef.max_sharpe()
weight_arr = ef.weights
ef.portfolio_performance(verbose=True);

# to show the chart you need to add the following line
plt.show()


##############################################################################################################
returns = expected_returns.returns_from_prices(df_pivot.set_index('Date')).dropna()
returns.head()

print(returns.head())


##############################################################################################################
# https://www.investopedia.com/terms/c/conditional_value_at_risk.asp
# Compute CVaR
portfolio_rets = (returns * weight_arr).sum(axis=1)
portfolio_rets.hist(bins=50);

# to show the chart you need to add the following line
plt.show()


##############################################################################################################
# VaR
var = portfolio_rets.quantile(0.05)
cvar = portfolio_rets[portfolio_rets <= var].mean()
print("VaR: {:.2f}%".format(100*var))
print("CVaR: {:.2f}%".format(100*cvar))


##############################################################################################################
from pypfopt import EfficientCVaR

ec = EfficientCVaR(mu, returns)
ec.min_cvar()
ec.portfolio_performance(verbose=True);




##############################################################################################################
# 3. PyPortfolioOpt : Efficient Frontier
##############################################################################################################

from pypfopt import CLA, plotting

cla = CLA(mu, S)
cla.max_sharpe()
cla.portfolio_performance(verbose=True, risk_free_rate=0.01305);



ax = plotting.plot_efficient_frontier(cla, showfig=False)

# to show the chart you need to add the following line
plt.show()




n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt((w.T * (S @ w.T)).sum(axis=0))
sharpes = rets / stds

print("Sample portfolio returns:", rets)
print("Sample portfolio volatilities:", stds)



##############################################################################################################
# Plot efficient frontier with Monte Carlo sim
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
# the following plotting is causing an unexpected issue. It is the plot of only the boundary of efficient frontier on subplots
# plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find and plot the tangency portfolio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Plot random portfolios
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Format
ax.set_title("PyPortfolioOpt: Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.show()

print("The portfolio performance is: ")
ef.portfolio_performance(verbose=True);



##############################################################################################################
# 4. PyPortfolioOpt : Discrete Allocation
##############################################################################################################

# Oftentimes, you don't have enough money to replicate the exact 'optimal' weights. So you need to find the closest possible portfolio with undivided amount of stocks

# last day prices
df_pivot.set_index('Date').iloc[-1]

print("The last day prices are: ")
print(df_pivot.set_index('Date').iloc[-1])


##############################################################################################################
print(f'Portfolio weights for max Sharpe optimisation (highest return-per-risk): {json.dumps(weights_max_sharpe, indent=4, sort_keys=True)} \n')


##############################################################################################################
df_last_day = stocks_prices[filter_last_date]
df_last_day['max_sharpe_weight']= df_last_day['Ticker'].apply(lambda x:weights_max_sharpe[x])


##############################################################################################################
df_last_day['stock_investment_amount'] = INVESTMENT*df_last_day['max_sharpe_weight'] /df_last_day['Close']


##############################################################################################################
df_last_day[['Close','Ticker','Date','max_sharpe_weight','stock_investment_amount']]
print(df_last_day[['Close','Ticker','Date','max_sharpe_weight','stock_investment_amount']])



##############################################################################################################
from pypfopt import DiscreteAllocation

latest_prices = df_pivot.set_index('Date').iloc[-1]  # prices as of the day you are allocating
da = DiscreteAllocation(weights_max_sharpe, latest_prices, total_portfolio_value = INVESTMENT, short_ratio=0.0)
alloc, leftover = da.lp_portfolio()
print(f"Discrete allocation for the initial investment ${INVESTMENT} performed with ${leftover:.2f} leftover")
alloc

