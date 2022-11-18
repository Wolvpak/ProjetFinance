import fundamentalanalysis as fa
from tqdm import tqdm
import pandas as pd
import numpy as np
import ta

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
    df.drop(columns = df.columns.difference(['timestamp','Open','High','Low','Close','volume']), inplace=True)
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
    return df