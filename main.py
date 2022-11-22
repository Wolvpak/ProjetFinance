import fundamentalanalysis as fa
from tqdm import tqdm
import pandas as pd
import numpy as np
import ta
import requests

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
def add_custom_indicators(df):
    df['adx 4'] = ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 4)
    df.drop(columns = df.columns.difference(['timestamp','Open','High','Low','Close','Volume']), inplace=True)
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
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histo'] = macd.macd_diff() #Histogramme MACD
    
    #Awesome Oscillator
    df['awesome_oscilllator'] = ta.momentum.awesome_oscillator(high=df['High'], low=df['Low'], window1=5, window2=34)

    # ADX
    df['adx'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['Close'], window = 14)
    
    # Fear and Greed 
    # Défintion
    def fear_and_greed(close):
        ''' Fear and greed indicator
        '''
        response = requests.get("https://api.alternative.me/fng/?limit=0&format=json")
        dataResponse = response.json()['data']
        fear = pd.DataFrame(dataResponse, columns = ['timestamp', 'value'])
    
        fear = fear.set_index(fear['timestamp'])
        fear.index = pd.to_datetime(fear.index, unit='s')
        del fear['timestamp']
        df = pd.DataFrame(close, columns = ['Close'])
        df['fearResult'] = fear['value']
        df['FEAR'] = df['fearResult'].ffill()
        df['FEAR'] = df.FEAR.astype(float)
        return pd.Series(df['FEAR'], name="FEAR")
    
    # Récupération des valeurs
    df["f_g"] = fear_and_greed(df["Close"])
    
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    
    # STOCHASTIC RSI
    df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14)
    df['stochastic'] = ta.momentum.stoch(high=df['High'],low=df['Low'],close=df['Close'], window=14,smooth_window=3)
    df['stoch_signal'] =ta.momentum.stoch_signal(high =df['High'],low=df['Low'],close=df['Close'], window=14, smooth_window=3)
    
    # WilliamsR
    df['max_21'] = df['High'].rolling(21).max()
    df['min_21'] = df['Low'].rolling(21).min()
    df['william_r'] = (df['Close'] - df['max_21']) / (df['max_21'] - df['min_21']) * 100
    df['emaw'] = ta.trend.ema_indicator(close=df['william_r'], window=13)
    
    # CCI
    df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3 
    df['sma_cci'] = df['hlc3'].rolling(40).mean()
    df['mad'] = df['hlc3'].rolling(40).apply(lambda x: pd.Series(x).mad())
    df['cci'] = (df['hlc3'] - df['sma_cci']) / (0.015 * df['mad']) 


    # PPO
    df['ppo'] = ta.momentum.ppo(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['ppo_signal'] = ta.momentum.ppo_signal(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['ppo_histo'] = ta.momentum.ppo_hist(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)

   
    # PVO
    df['pvo'] = ta.momentum.pvo(volume = df['Volume'], window_slow=26, window_fast=12, window_sign=9)
    df['pvo_signal'] = ta.momentum.pvo_signal(volume = df['Volume'], window_slow=26, window_fast=12, window_sign=9)
    df['pvo_histo'] = ta.momentum.pvo_hist(volume = df['Volume'], window_slow=26, window_fast=12, window_sign=9)


    # Aroon
    df['aroon_up'] = ta.trend.aroon_up(close=df['Close'], window=25)
    df['aroon_dow'] = ta.trend.aroon_down(close=df['Close'], window=25)
    return df