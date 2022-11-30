import fundamentalanalysis as fa
from tqdm import tqdm
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from ta import add_all_ta_features
<<<<<<< HEAD
=======
import matplotlib as plt
>>>>>>> 67ae792bb6c174824e77ff6aeff932cf3eeab747

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


#librarie qui permet de calculer tous les indicateurs une fois le fichier recupéré
def add_classic_indicators(df):
    np.seterr(invalid='ignore')
    df['adx 4'] = ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 4)
    
#moyennes mobiles
    # EMA
    df['ema7']=ta.trend.ema_indicator(close=df['Close'], window=7)
    df['ema30']=ta.trend.ema_indicator(close=df['Close'], window=30)
    df['ema50']=ta.trend.ema_indicator(close=df['Close'], window=50)
    df['ema100']=ta.trend.ema_indicator(close=df['Close'], window=100)
    df['ema150']=ta.trend.ema_indicator(close=df['Close'], window=150)
    df['ema200']=ta.trend.ema_indicator(close=df['Close'], window=200)
    
    # SMA
    df['sma7']=ta.trend.sma_indicator(close=df['Close'], window=7)
    df['sma30']=ta.trend.sma_indicator(close=df['Close'], window=30)
    df['sma50']=ta.trend.sma_indicator(close=df['Close'], window=50)
    df['sma100']=ta.trend.sma_indicator(close=df['Close'], window=100)
    df['sma150']=ta.trend.sma_indicator(close=df['Close'], window=150)
    df['sma200']=ta.trend.sma_indicator(close=df['Close'], window=200)
    
    
#oscillateurs
    # MACD
    macd2452 = ta.trend.MACD(close=df['Close'], window_fast=24, window_slow=52, window_sign=9)
    df['macd_24_52'] = macd2452.macd_diff()
    macd1226 = ta.trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd_12_26'] = macd1226.macd_diff()

    # ADX
    df['adx5'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx10'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx14'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx20'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    df['adx30'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    
    # RSI
    df['rsi3'] = ta.momentum.RSIIndicator(close=df['Close'], window=3).rsi()
    df['rsi5'] = ta.momentum.RSIIndicator(close=df['Close'], window=5).rsi()
    df['rsi7'] = ta.momentum.RSIIndicator(close=df['Close'], window=7).rsi()
    df['rsi10'] = ta.momentum.RSIIndicator(close=df['Close'], window=10).rsi()
    df['rsi14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['rsi17'] = ta.momentum.RSIIndicator(close=df['Close'], window=17).rsi()
    df['rsi20'] = ta.momentum.RSIIndicator(close=df['Close'], window=20).rsi()
    df['rsi25'] = ta.momentum.RSIIndicator(close=df['Close'], window=25).rsi()
    df['rsi30'] = ta.momentum.RSIIndicator(close=df['Close'], window=30).rsi()   

    # STOCHASTIC RSI
    df['stochrsi4'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=4).stochrsi()
    df['stochrsi7'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=7).stochrsi()
    df['stochrsi10'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=10).stochrsi()
    df['stochrsi14'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df['stochrsi20'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=20).stochrsi()
    df['stochrsi25'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=25).stochrsi()
    
    #filtrer avec EMA?
    #WilliamsR
    df['willr7'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=7).williams_r()
    df['willr14'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=14).williams_r()
    df['willr21'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=21).williams_r()
    df['willr60'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['Close'],lbp=60).williams_r()
    
    # CCI
    df['CCI5'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    df['CCI7'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    df['CCI10'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    df['CCI14'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    df['CCI20'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['Close'],window=20).cci()
    
    data_FT = df[['Date', 'GS']]
    close_fft = np.fft.fft(np.asarray(data_FT['GS'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [2, 7, 15, 100]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(data_FT['GS'],  label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()

    #add talib features (volume, trend, momentum, volatility)
    ta_df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    ta_df.index = ta_df.index.date
    
    return pd.concat([df,ta_df],axis=1).T.drop_duplicates().T.iloc[200:,:]




#comprendre comment evaluer la qualité de variables (forward stock return)
#https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/07_linear_models/05_predicting_stock_returns_with_linear_regression.ipynb
#https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/07_linear_models/04_statistical_inference_of_stock_returns_with_statsmodels.ipynb

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
    #https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/04_alpha_factor_research/01_feature_engineering.ipynb
    #peut se faire avec toutes les variables mais a eviter avec lstm et DRL peut etre
    return df

#mean reversion
#https://admiralmarkets.com/education/articles/forex-indicators/macd-indicator-in-depth#:~:text=of%20future%20performance.-,MACD%20Indicator%20Settings%20for%20Intraday%20Trading,that%20works%20well%20on%20M30.
#et d'autres
def add_strategy_signals(df):
    return df

def add_fin_ratios_and_commodities(df): #FAMA, Beta, Omega, Sortino, Calmar
    df["Gold Close"] = get_stock_data('GC=F')['Close']
    df["WTI Oil Close"] = get_stock_data('CL=F')['Close']
    df["5Y TY ^FVX"] = get_stock_data('^FVX')['Close']
    df["CAC 40"] = get_stock_data('^FCHI')['Close']
    return df

   
def get_market_data():
    Movements, Open, High, Low, Close = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tickers = get_euronext_tickers()
    for ticker in tickers:
        try:
            stock_data = get_stock_data(ticker)
            Movements[ticker] = stock_data['Close']-stock_data['Open']
            Open[ticker] = stock_data['Open']
            High[ticker] = stock_data['High']
            Low[ticker] = stock_data['Low']
            Close[ticker] = stock_data['Close']
        except:
            None
    Movements = Movements.loc[:,Movements.isnull().sum() < 1]     
    Open = Open.loc[:,Open.isnull().sum() < 1]   
    High = High.loc[:,High.isnull().sum() < 1]  
    Low = Low.loc[:,Low.isnull().sum() < 1]  
    Close = Close.loc[:,Close.isnull().sum() < 1]        
    print(f'{len(Movements.columns)*100/len(tickers)} % of tickers are available')
    return Movements, Open, High, Low, Close
            
    
#reste : VIX, supertrend, analyse de sentiment(BERT), et strategies (vol anomaly), varmax