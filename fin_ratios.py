from main import get_stock_data
import numpy as np

cac_rets = get_stock_data('^FCHI')['Close'].pct_change()

def sharpe(stock_returns):
    return np.sqrt(252)*stock_returns.mean() / stock_returns.std()

def sortino(stock_returns):
    return np.sqrt(252)*stock_returns.mean() / stock_returns[stock_returns<0].std()

def upside_potential(stock_returns):
    return np.sqrt(252)*stock_returns[stock_returns>0].mean() / stock_returns[stock_returns<0].std()

def treynor(stock_returns):
    return np.sqrt(252)*stock_returns.mean() / beta(stock_returns)

def var(stock_returns, alpha = 0.05):
    sorted_returns = np.sort(stock_returns)
    ind_alpha = int(alpha*len(sorted_returns))
    return sorted_returns[ind_alpha] #-ind alpha for for maximum returns (with 1-alpha confidence)

def cvar(stock_returns, alpha = 0.05):
    sorted_returns = np.sort(stock_returns)
    ind_alpha = int(alpha*len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, ind_alpha):
        sum_var += sorted_returns[i]
    return sum_var / ind_alpha

def beta(stock_returns):
    minind = stock_returns.index.min()
    maxind = stock_returns.index.max()
    benchmark = cac_rets.loc[minind:maxind]
    cov = stock_returns.cov(benchmark)
    market_var = benchmark.var()
    return cov / market_var

def add_risk_measures(stock_data, stock_ret):  #= stock_data['Returns']
    for window in [252, 126, 52, 26]:
        stock_data[f'Sharpe {window}'] = stock_ret.rolling(window).apply(sharpe)
        stock_data[f'Sortino {window}'] = stock_ret.rolling(window).apply(sortino)
        stock_data[f'Treynor {window}'] = stock_ret.rolling(window).apply(treynor)
        stock_data[f'Beta {window}'] = stock_ret.rolling(window).apply(beta)
    for window in [252, 126, 52]:    
        stock_data[f'Var {window}'] = stock_ret.rolling(window).apply(var)
        stock_data[f'CVar {window}'] = stock_ret.rolling(window).apply(cvar)
    return stock_data

def add_fin_ratios_and_commodities(stock_data): #FAMA, Beta, Omega, Sortino, Calmar
    stock_data["Gold Close"] = get_stock_data('GC=F')['Close']
    stock_data["WTI Oil Close"] = get_stock_data('CL=F')['Close']
    stock_data["5Y TY ^FVX"] = get_stock_data('^FVX')['Close']
    stock_data["CAC 40"] = get_stock_data('^FCHI')['Close']
    stock_data["CAC Returns"] = stock_data["CAC 40"].pct_change()
    return stock_data

def annual_risk_measures(stock_data, stock_ret):
    stock_data['Sortino'] = stock_ret.rolling(252).apply(sortino)
    stock_data['Beta'] = stock_ret.rolling(252).apply(beta)
    stock_data['CVar'] = stock_ret.rolling(252).apply(cvar)
    return stock_data

def range_80(df, ratio, alpha = 0.1):
    df_ratio = df[ratio]
    sorted_df = np.sort(df_ratio)
    ind_alpha = int(alpha*len(sorted_df))
    return sorted_df[ind_alpha], sorted_df[-ind_alpha]

#results were compared with https://www.profitspi.com/stock/view.aspx?v=stock-chart&uv=131900#&&vs=638057908951172050