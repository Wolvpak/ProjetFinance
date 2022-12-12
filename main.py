import fundamentalanalysis as fa
from tqdm import tqdm
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from ta import add_all_ta_features
import warnings
warnings.simplefilter(action='ignore')

#saves a dictionnary containing the financials and metrics from every company, for every available year
def download_stocks_fundamentals(stocks_list, filename, api_key):
    data = {}
    for stock in tqdm(stocks_list) :
        stock_financial_ratios = fa.financial_ratios(stock,api_key,period="annual").iloc[1:,:].to_dict()
        stock_key_metrics = fa.key_metrics(stock,api_key,period="annual").iloc[1:,:].to_dict()
        stock_data_merged = {key: dict(stock_financial_ratios[key], **stock_key_metrics[key]) for key in stock_financial_ratios}
        data.update({stock : stock_data_merged})
    np.save(filename + '.npy', data)
    

#return each company's yearly informations from dictionnary database (created by download_stocks_fundamentals())   
def db_fundamentals_summary(stock_list, database, year = '2021'):
    yearly_data = pd.DataFrame()
    for stock in stock_list:
        try:
            stock_data = pd.DataFrame({stock : database[stock][year]})
            yearly_data = pd.concat([yearly_data,stock_data], axis = 1)
        except:
            None
    return yearly_data.transpose()

#adds TA to price dataframe from yfinance
def add_classic_indicators(df):
    np.seterr(invalid='ignore')
    # EMA
    df['ema5']=ta.trend.ema_indicator(close=df['Close'], window=5)
    df['ema8']=ta.trend.ema_indicator(close=df['Close'], window=8)
    df['ema12']=ta.trend.ema_indicator(close=df['Close'], window=12)
    df['ema16']=ta.trend.ema_indicator(close=df['Close'], window=16)
    df['ema20']=ta.trend.ema_indicator(close=df['Close'], window=20)
    df['ema26']=ta.trend.ema_indicator(close=df['Close'], window=26)
    
    # SMA
    df['sma5']=ta.trend.sma_indicator(close=df['Close'], window=5)
    df['sma8']=ta.trend.sma_indicator(close=df['Close'], window=8)
    df['sma12']=ta.trend.sma_indicator(close=df['Close'], window=12)
    df['sma16']=ta.trend.sma_indicator(close=df['Close'], window=16)
    df['sma20']=ta.trend.sma_indicator(close=df['Close'], window=20)
    df['sma26']=ta.trend.sma_indicator(close=df['Close'], window=26)

    
    # MACD
    macd2452 = ta.trend.MACD(close=df['Close'], window_fast=24, window_slow=52, window_sign=9)
    macd1226 = ta.trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd_24_52'] = macd2452.macd_diff()
    df['macd_12_26'] = macd1226.macd_diff()

    # ADX
    df['adx7'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx10'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 10)
    df['adx14'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx20'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 20)
    df['adx30'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 30)
    
    # RSI
    df['rsi7'] = ta.momentum.RSIIndicator(close=df['Close'], window=7).rsi()
    df['rsi10'] = ta.momentum.RSIIndicator(close=df['Close'], window=10).rsi()
    df['rsi14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['rsi17'] = ta.momentum.RSIIndicator(close=df['Close'], window=17).rsi()
    df['rsi20'] = ta.momentum.RSIIndicator(close=df['Close'], window=20).rsi()
    df['rsi25'] = ta.momentum.RSIIndicator(close=df['Close'], window=25).rsi()

    # STOCHASTIC RSI
    df['stochrsi7'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=7).stochrsi()
    df['stochrsi10'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=10).stochrsi()
    df['stochrsi14'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df['stochrsi17'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=17).stochrsi()
    df['stochrsi20'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=20).stochrsi()
    df['stochrsi25'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=25).stochrsi()
    
    #filtrer avec EMA?
    #WilliamsR
    df['willr10'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=10).williams_r()
    df['willr14'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=14).williams_r()
    df['willr17'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=17).williams_r()
    df['willr20'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=20).williams_r()
    df['willr25'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=25).williams_r()
    
    # CCI
    df['CCI5'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=5).cci()
    df['CCI7'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=7).cci()
    df['CCI10'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=10).cci()
    df['CCI14'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=14).cci()
    df['CCI20'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    
    #data_FT = df[['Date', 'GS']]
    #close_fft = np.fft.fft(np.asarray(data_FT['GS'].tolist()))
    #fft_df = pd.DataFrame({'fft':close_fft})
    #fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    #fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    #fft_list = np.asarray(fft_df['fft'].tolist())
    #for num_ in [2, 7, 15, 100]:
        #fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        #df['Fourier' + num_] = fft_list_m10


    #add talib features (volume, trend, momentum, volatility)
    ta_df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    ta_df.index = ta_df.index.date
    return pd.concat([df,ta_df],axis=1).T.drop_duplicates().T

def get_euronext_tickers():
    df = pd.read_csv('euronext_tickers.csv', encoding = 'unicode_escape')
    df = df[df['Exchange']=='Euronext Paris']['Ticker']
    return df.to_list()

def get_stock_data(stock_ticker, lookback = '5y'):
    stock_data = yf.Ticker(stock_ticker).history(period=lookback).drop(columns=['Dividends','Stock Splits'])
    stock_data['Returns'] = stock_data["Close"].pct_change()
    stock_data['Log Returns'] = np.log(stock_data["Close"]).diff()
    stock_data.index = pd.to_datetime(stock_data.index.date)
    stock_data.index.name = 'date'
    return stock_data

def generate_lagged_variables(df):
    df['Returns n-1'] = df['Returns'].shift(1)
    df['Returns n-2'] = df['Returns'].shift(2)
    df['Returns n-3'] = df['Returns'].shift(3)
    df['Returns n-4'] = df['Returns'].shift(4)
    df['Returns n-5'] = df['Returns'].shift(5)
    df['Returns n-6'] = df['Returns'].shift(6)
    df['Returns n-7'] = df['Returns'].shift(7)
    df['Returns n-10'] = df['Returns'].shift(10)
    df['Returns n-15'] = df['Returns'].shift(15)
    df['Returns n-20'] = df['Returns'].shift(20)
    df['Returns n-25'] = df['Returns'].shift(25)
    df['RSI n-1'] = df['rsi14'].shift(1)
    df['RSI n-2'] = df['rsi14'].shift(2)
    df['RSI n-3'] = df['rsi14'].shift(3)
    df['RSI n-4'] = df['rsi14'].shift(4)
    df['RSI n-5'] = df['rsi14'].shift(5)
    df['RSI n-6'] = df['rsi14'].shift(6)
    df['RSI n-7'] = df['rsi14'].shift(7)
    #https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/04_alpha_factor_research/01_feature_engineering.ipynb
    #peut se faire avec toutes les variables mais a eviter avec lstm et DRL peut etre
    return df


#https://admiralmarkets.com/education/articles/forex-indicators/macd-indicator-in-depth#:~:text=of%20future%20performance.-,MACD%20Indicator%20Settings%20for%20Intraday%20Trading,that%20works%20well%20on%20M30.
def add_strategy_signals(df):
    return df
   
def get_market_data(mean_volume):
    Open, High, Low, Close = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tickers = get_euronext_tickers()
    for ticker in tickers:
        try:
            stock_data = get_stock_data(ticker)
            #print(ticker, stock_data['Volume'].mean())
            if stock_data['Volume'].mean()> mean_volume:
                Open[ticker] = stock_data['Open']
                High[ticker] = stock_data['High']
                Low[ticker] = stock_data['Low']
                Close[ticker] = stock_data['Close']
        except:
            None   
    Open = Open.loc[:,Open.isnull().sum() < 1]   
    High = High.loc[:,High.isnull().sum() < 1]  
    Low = Low.loc[:,Low.isnull().sum() < 1]  
    Close = Close.loc[:,Close.isnull().sum() < 1]        
    print(f'{len(Close.columns)*100/len(tickers)} % of tickers are available')
    return Open, High, Low, Close
