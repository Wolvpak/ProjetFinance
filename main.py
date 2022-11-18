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
    
    # VMC
    #Class pour nos indicateurs
    class VMC():
        """ VuManChu Cipher B + Divergences 
            Args:
                high(pandas.Series): dataset 'High' column.
                low(pandas.Series): dataset 'Low' column.
                close(pandas.Series): dataset 'Close' column.
                wtChannelLen(int): n period.
                wtAverageLen(int): n period.
                wtMALen(int): n period.
                rsiMFIperiod(int): n period.
                rsiMFIMultiplier(int): n period.
                rsiMFIPosY(int): n period.
        """
        def __init__(
            self: pd.Series,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            open: pd.Series,
            wtChannelLen: int = 9,
            wtAverageLen: int = 12,
            wtMALen: int = 3,
            rsiMFIperiod: int = 60,
            rsiMFIMultiplier: int = 150,
            rsiMFIPosY: int = 2.5
        ) -> None:
            self._high = high
            self._low = low
            self._close = close
            self._open = open
            self._wtChannelLen = wtChannelLen
            self._wtAverageLen = wtAverageLen
            self._wtMALen = wtMALen
            self._rsiMFIperiod = rsiMFIperiod
            self._rsiMFIMultiplier = rsiMFIMultiplier
            self._rsiMFIPosY = rsiMFIPosY
            self._run()
            self.wave_1()
    
        def _run(self) -> None:
            try:
                self._esa = ta.trend.ema_indicator(
                    close=self._close, window=self._wtChannelLen)
            except Exception as e:
                print(e)
                raise
    
            self._esa = ta.trend.ema_indicator(
                close=self._close, window=self._wtChannelLen)
            self._de = ta.trend.ema_indicator(
                close=abs(self._close - self._esa), window=self._wtChannelLen)
            self._rsi = ta.trend.sma_indicator(self._close, self._rsiMFIperiod)
            self._ci = (self._close - self._esa) / (0.015 * self._de)
    
        def wave_1(self) -> pd.Series:
            """VMC Wave 1 
            Returns:
                pandas.Series: New feature generated.
            """
            wt1 = ta.trend.ema_indicator(self._ci, self._wtAverageLen)
            return pd.Series(wt1, name="wt1")
    
        def wave_2(self) -> pd.Series:
            """VMC Wave 2
            Returns:
                pandas.Series: New feature generated.
            """
            wt2 = ta.trend.sma_indicator(self.wave_1(), self._wtMALen)
            return pd.Series(wt2, name="wt2")
    
        def money_flow(self) -> pd.Series:
            """VMC Money Flow
                Returns:
                pandas.Series: New feature generated.
            """
            mfi = ((self._close - self._open) /
                    (self._high - self._low)) * self._rsiMFIMultiplier
            rsi = ta.trend.sma_indicator(mfi, self._rsiMFIperiod)
            money_flow = rsi - self._rsiMFIPosY
            return pd.Series(money_flow, name="money_flow")
        
    # Récupération des données
    df['hlc3'] = (df['High'] +df['Close'] + df['Low'])/3
    vmc = VMC(high =df['High'],low = df['Low'],close=df['hlc3'],open=df['Open'])
    df['vmc_wave1'] = vmc.wave_1()
    df['vmc_wave2'] = vmc.wave_2()
    vmc = VMC(high=df['High'], low=df['Low'], close=df['Close'], open=df['Open'])
    df['money_flow'] = vmc.money_flow()
    
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
    df['pvo'] = ta.momentum.pvo(volume = df['volume'], window_slow=26, window_fast=12, window_sign=9)
    df['pvo_signal'] = ta.momentum.pvo_signal(volume = df['volume'], window_slow=26, window_fast=12, window_sign=9)
    df['pvo_histo'] = ta.momentum.pvo_hist(volume = df['volume'], window_slow=26, window_fast=12, window_sign=9)



    # Aroon
    df['aroon_up'] = ta.trend.aroon_up(close=df['Close'], window=25)
    df['aroon_dow'] = ta.trend.aroon_down(close=df['Close'], window=25)
    return df