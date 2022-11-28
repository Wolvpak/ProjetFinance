import pandas as pd
import numpy as np


ohlcv = ['open', 'high', 'low', 'close', 'volume']
   
def rank(df):
    return df.rank(pct=True)

def scale(df):
    return df.div(df.abs().sum(), axis=0)

def log(df):
    return np.log1p(df)

def sign(df):
    return np.sign(df)

def power(df, exp):
    return df.pow(exp)

def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return df.shift(t)

def ts_delta(df, period=1):
    return df.diff(period)

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()

def ts_mean(df, window=10):
    return df.rolling(window).mean()

#def ts_weighted_mean(df, period=10):
#    return (df.apply(lambda x: WMA(x, timeperiod=period)))

def ts_std(df, window=10):
    return (df.rolling(window).std())

def ts_rank(df, window=10):
    return (df.rolling(window).apply(lambda x: x.rank().iloc[-1]))

def ts_product(df, window=10):
    return (df .rolling(window).apply(np.prod))

def ts_min(df, window=10):
        return df.rolling(window).min()
    
def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).add(1)

def ts_argmin(df, window=10):
    return (df.rolling(window).apply(np.argmin).add(1))

def ts_corr(x, y, window=10):
    return x.rolling(window).corr(y)

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)

def add_artificial_variables(df):
#https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/24_alpha_factor_library/03_101_formulaic_alphas.ipynb

    vwap = df['Open'].add(df['High']).add(df['Low']).add(df['Close']).div(4)
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    r = df['Returns']
    v = df['Volume']
    adv180 = ts_mean(v, 180)
    adv81 = ts_mean(v, 81)
    adv150= ts_mean(v, 150)
    adv50 = ts_mean(v, 50)
    adv10 = ts_mean(v, 10)
    adv40 = ts_mean(v, 40)
    adv30 = ts_mean(v, 30)
    adv20 = ts_mean(v, 20)
    
    #Alpha 001
    c[r < 0] = ts_std(r , 20)
    df['alpha1'] = (rank(ts_argmax(power(c, 2), 5)).mul(-.5))
    
    #Alpha 002
    s1 = rank(ts_delta(log(df['Volume']), 2))
    s2 = rank((df['Close'] / df['Open']) - 1)
    alpha = -ts_corr(s1, s2, 6)
    df['alpha2'] = alpha.replace([-np.inf, np.inf], np.nan)
    
    #Alpha 003
    df['alpha3'] = (-ts_corr(rank(df['Open']), rank(df['Volume']), 10).replace([-np.inf, np.inf], np.nan))
    
    #Alpha 004
    df['alpha4'] = (-ts_rank(rank(df['Low']), 9))
    
    #Alpha 005
    df['alpha5'] = (rank(df['Open'].sub(ts_mean(vwap, 10))).mul(rank(df['Close'].sub(vwap)).mul(-1).abs()))
    
    #Alpha 006
    df['alpha6'] = (-ts_corr(o, v, 10))
    
    #Alpha 007
    delta7 = ts_delta(c, 7)

    df['alpha7'] = (-ts_rank(abs(delta7), 60)
            .mul(sign(delta7))
            .where(adv20<v, -1))
    
    #Alpha 008
    df['alph8'] = (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                       ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10)))))
    
    #Alpha 009
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    df['alpha9'] = (alpha)
    
    #Alpha 010
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0,
                                              -close_diff))
    df['alpha10'] = (rank(alpha))
    
    #Alpha 011
    df['alpha11'] = (rank(ts_max(vwap.sub(c), 3)).add(rank(ts_min(vwap.sub(c), 3))).mul(rank(ts_delta(v, 3))))
    
    #Alpha 012
    df['alpha12'] = (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1)))
    
    #Alpha 013
    df['alpha13'] = (-rank(ts_cov(rank(c), rank(v), 5)))
    
    #Alpha 014
    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10).replace([-np.inf,np.inf],np.nan))
    df['alpha14'] = (alpha)
    
    #Alpha 015
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3).replace([-np.inf, np.inf], np.nan)), 3))
    df['alpha15'] = (alpha)
    
    #Alpha 016
    df['alpha16'] = (-rank(ts_cov(rank(h), rank(v), 5)))
    
    #Alpha 017
    adv20 = ts_mean(v, 20)
    df['alpha17'] = (-rank(ts_rank(c, 10)).mul(rank(ts_delta(ts_delta(c, 1), 1))).mul(rank(ts_rank(v.div(adv20), 5))))

    #Alpha 018
    df['alpha18'] = (-rank(ts_std(c.sub(o).abs(), 5).add(c.sub(o)).add(ts_corr(c, o, 10).replace([-np.inf,np.inf],np.nan))))
    
    #Alpha 019
    df['alpha19'] = (-sign(ts_delta(c, 7) + ts_delta(c, 7)).mul(1 + rank(1 + ts_sum(r, 250))))
    
    #Alpha 020
    df['alpha20'] = (rank(o - ts_lag(h, 1)).mul(rank(o - ts_lag(c, 1))).mul(rank(o - ts_lag(l, 1))).mul(-1))
    
    #Alpha 021
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)
    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1
    val = np.ones_like(c)
    alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],choicelist=[-1, 1, -1], default=1),index=c.index)
    df['alpha21'] = (alpha)
    
    #Alpha 022
    df['alpha22'] = (ts_delta(ts_corr(h, v, 5).replace([-np.inf,np.inf],np.nan), 5).mul(rank(ts_std(c, 20))).mul(-1))
    
    #Alpha 023
    df['alpha23'] = (ts_delta(h, 2).mul(-1).where(ts_mean(h, 20) < h, 0))
    
    #Alpha 024
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05
    df['alpha24'] = (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3)))
    
    #Alpha 025
    df['alpha25'] = (rank(-r.mul(adv20).mul(vwap).mul(h.sub(c))))
    
    #Alpha 026
    df['alpha26'] = (ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5).replace([-np.inf, np.inf], np.nan), 3).mul(-1))
    
    #Alpha 027
    cond = rank(ts_mean(ts_corr(rank(v),rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    df['alpha27'] = (alpha.where(cond <= 0.5, -alpha))
    
    #Alpha 028
    df['alpha28'] = (scale(ts_corr(adv20, l, 5).replace([-np.inf, np.inf], 0).add(h.add(l).div(2).sub(c))))
    
    #Alpha 029
    df['alpha29'] = (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))), 5).add(ts_rank(ts_lag((-1 * r), 6), 5)))
    
    #Alpha 030
    close_diff = ts_delta(c, 1)
    df['alpha30'] = (rank(sign(close_diff).add(sign(ts_lag(close_diff, 1))).add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20)))
    
    #Alpha 031
#    df['alpha31'] = (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
#            .add(rank(ts_delta(c, 3).mul(-1)))
#            .add(sign(scale(ts_corr(adv20, l, 12)
#                            .replace([-np.inf, np.inf],
#                                     np.nan))))
#            .stack('ticker')
#            .swaplevel())
    
    #Alpha 032
    df['alpha32'] = (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230))))
    
    #Alpha 033
    df['alpha33'] = (rank(o.div(c).mul(-1).add(1).mul(-1)))
    
    #Alpha 034
    df['alpha34'] = (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan))
                 .mul(-1)
                 .sub(rank(ts_delta(c, 1)))
                 .add(2)))
    
    #Alpha 035
    df['alpha35'] = (ts_rank(v, 32)
            .mul(1 - ts_rank(c.add(h).sub(l), 16))
            .mul(1 - ts_rank(r, 32)))
    
    #Alpha 036
    df['alpha36'] = (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
            .add(rank(o.sub(c)).mul(.7))
            .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
            .add(rank(abs(ts_corr(vwap, adv20, 6))))
            .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6)))
    
    #Alpha 037
    df['alpha37'] = (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c))))
    
    #Alpha 038
    df['alpha38'] = (rank(ts_rank(o, 10))
            .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
            .mul(-1))
    
    #Alpha 039
#    df['alpha39'] = (rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1))).mul(-1)
#            .mul(rank(ts_mean(r, 250).add(1)))
#            .stack('ticker')
#            .swaplevel())
    
    #Alpha 040
    df['alpha40'] = (rank(ts_std(h, 10))
            .mul(ts_corr(h, v, 10))
            .mul(-1))

    
    #Alpha 041
    df['alpha41'] = (power(h.mul(l), 0.5)
            .sub(vwap))
    
    #Alpha 042
    df['alpha42'] = (rank(vwap.sub(c))
            .div(rank(vwap.add(c))))
    
    #Alpha 043
    df['alpha43'] = (ts_rank(v.div(adv20), 20)
            .mul(ts_rank(ts_delta(c, 7).mul(-1), 8)))
    
    #Alpha 044
    df['alpha44'] = (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1))
    
    #Alpha 045
    df['alpha45'] = (rank(ts_mean(ts_lag(c, 5), 20))
            .mul(ts_corr(c, v, 2)
                 .replace([-np.inf, np.inf], np.nan))
            .mul(rank(ts_corr(ts_sum(c, 5),
                              ts_sum(c, 20), 2)))
            .mul(-1))
    
    #Alpha 046
#    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
#    alpha = pd.DataFrame(-np.ones_like(cond),
#                         index=c.index,)
#    alpha[cond.isnull()] = np.nan
#    df['alpha46'] = (cond.where(cond > 0.25,
#                       -alpha.where(cond < 0,
#                       -ts_delta(c, 1))))
    
    #Alpha 047
    df['alpha47'] = (rank(c.pow(-1)).mul(v).div(adv20)
            .mul(h.mul(rank(h.sub(c))
                       .div(ts_mean(h, 5)))
                 .sub(rank(ts_delta(vwap, 5)))))
    
    #Alpha 048
    #df['alpha48'] = (indneutralize(((ts_corr(ts_delta(c, 1), ts_delta(ts_lag(c, 1), 1), 250) * 
    #   ts_delta(c, 1)) / c), IndClass.subindustry) / 
    #   ts_sum(((ts_delta(c, 1) / ts_lag(c, 1))^2), 250))
    
    #Alpha 049
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    df['alpha49'] = (-ts_delta(c, 1)
            .where(cond, 1))
    
    #Alpha 050
    df['alpha50'] = (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1))
    
    #Alpha 51
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
           .sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
    df['alpha51'] = (-ts_delta(c, 1)
            .where(cond, 1))
    
    #Alpha 052
    df['alpha52'] = (ts_delta(ts_min(l, 5), 5)
            .mul(rank(ts_sum(r, 240)
                      .sub(ts_sum(r, 20))
                      .div(220)))
            .mul(ts_rank(v, 5)))
    
    #Alpha 053
    inner = (c.sub(l)).add(1e-6)
    df['alpha53'] = (ts_delta(h.sub(c)
                     .mul(-1).add(1)
                     .div(c.sub(l)
                          .add(1e-6)), 9)
            .mul(-1))
    
    #Alpha 054
    df['alpha54'] = (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5)))
    
    #Alpha 055
    df['alpha55'] = (ts_corr(rank(c.sub(ts_min(l, 12))
                         .div(ts_max(h, 12).sub(ts_min(l, 12))
                              .replace(0, 1e-6))),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1))
    
    #Alpha 056
    #df['alpha56'] = -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * rank((returns * cap))
    
    #Alpha 057
#    df['alpha57'] = (c.sub(vwap.add(1e-5))
#            .div(ts_weighted_mean(rank(ts_argmax(c, 30)))).mul(-1)
#            .stack('ticker')
#            .swaplevel())
    
    #Alpha 058
    #df['alpha58'] = (-1 * ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.sector), v, 3), 7), 5))
    
    #Alpha 059
    #df['alpha59'] = -ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), v, 4), 16), 8)
    
    #Alpha 060
    df['alpha60'] = (scale(rank(c.mul(2).sub(l).sub(h)
                       .div(h.sub(l).replace(0, 1e-5))
                       .mul(v))).mul(2)
            .sub(scale(rank(ts_argmax(c, 10)))).mul(-1))
    
    #Alpha 061
    df['alpha61'] = (rank(vwap.sub(ts_min(vwap, 16)))
            .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
            .astype(int))
    
    #Alpha 062
    df['alpha62'] = (rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
            .lt(rank(
                rank(o).mul(2))
                .lt(rank(h.add(l).div(2))
                    .add(rank(h))))
            .mul(-1))
    
    #Alpha 063
    #df['alpha63'] = ((rank(ts_weighted_mean(ts_delta(IndNeutralize(c, IndClass.industry), 2), 8)) - 
    #    rank(ts_weighted_mean(ts_corr(((vwap * 0.318108) + (o * (1 - 0.318108))), 
    #                                    ts_sum(adv180, 37), 13), 12))) * -1)
    
    #Alpha 064
    w = 0.178404
    df['alpha64'] = (rank(ts_corr(ts_sum(o.mul(w).add(l.mul(1 - w)), 12),
                         ts_sum(ts_mean(v, 120), 12), 16))
            .lt(rank(ts_delta(h.add(l).div(2).mul(w)
                               .add(vwap.mul(1 - w)), 3)))
            .mul(-1))
    
    #Alpha 065
    w = 0.00817205
    df['alpha65'] = (rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)),
                         ts_mean(ts_mean(v, 60), 9), 6))
            .lt(rank(o.sub(ts_min(o, 13))))
            .mul(-1))

    #Alpha 066
#    w = 0.96633
#    df['alpha66'] = (rank(ts_weighted_mean(ts_delta(vwap, 4), 7))
#            .add(ts_rank(ts_weighted_mean(l.mul(w).add(l.mul(1 - w))
#                                           .sub(vwap)
#                                           .div(o.sub(h.add(l).div(2)).add(1e-3)), 11), 7))
#            .mul(-1))
    
    #Alpha 067
    #df['alpha67'] = ((rank((h - ts_min(h, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1) 
    
    #Alpha 068
    w = 0.518371
    df['alpha68'] = (ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
            .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
            .mul(-1))

    
    #Alpha 069
    #df['alpha69'] = ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1) 
    
    #Alpha 070
    #df['alpha70'] = ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    
    #Alpha 071
#    s1 = (ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 3),
#                                       ts_rank(ts_mean(v, 180), 12), 18), 4), 16))
#    s2 = (ts_rank(ts_weighted_mean(rank(l.add(o).
#                                    sub(vwap.mul(2)))
#                               .pow(2), 16), 4))

    df['alpha71'] = (s1.where(s1 > s2, s2))
    
    #Alpha 072
#    df['alpha72'] = (rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 9), 10))
#            .div(rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2))))
    
    #Alpha 073
    w = 0.147155
#    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
#    s2 = (ts_rank(ts_weighted_mean(ts_delta(o.mul(w).add(l.mul(1 - w)), 2)
#                                   .div(o.mul(w).add(l.mul(1 - w)).mul(-1)), 3), 16))
#    df['alpha73'] = (s1.where(s1 > s2, s2)
#            .mul(-1)
#            .stack('ticker')
#            .swaplevel())
    
    #Alpha 074
    w = 0.0261661
    df['alpha74'] = (rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
            .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
            .mul(-1))
    
    #Alpha 075
    df['alpha75'] = (rank(ts_corr(vwap, v, 4))
            .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
            .astype(int))
    
    #Alpha 076
    #df['alpha76'] =  (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(l, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1) 
    
    #Alpha 077
#    s1 = rank(ts_weighted_mean(h.add(l).div(2).sub(vwap), 20))
#    s2 = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 3), 5))
#    df['alpha77'] = (s1.where(s1 < s2, s2))
    
    #Alpha 078
    w = 0.352233
    df['alpha78'] = (rank(ts_corr(ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                         ts_sum(ts_mean(v, 40), 19), 6))
            .pow(rank(ts_corr(rank(vwap), rank(v), 5))))

    
    
    
    #Alpha 079
    #df['alpha79'] = (rank(delta(IndNeutralize(((c * 0.60733) + (o * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644))) 
    
    #Alpha 080
    #df['alpha80'] = ((rank(Sign(delta(IndNeutralize(((o * 0.868128) + (h * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(h, adv10, 5.11456), 5.53756)) * -1) 
    
    #Alpha 081
    df['alpha81'] = (rank(log(ts_product(rank(rank(ts_corr(vwap,
                                                  ts_sum(ts_mean(v, 10), 50), 8))
                                     .pow(4)), 15)))
            .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
            .mul(-1))
    
    #Alpha 082
    #df['alpha82'] = (min(rank(decay_linear(delta(o, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(v, IndClass.sector), ((o * 0.634196) + (o * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1) 
    
    #Alpha 083
    s = h.sub(l).div(ts_mean(c, 5))
    df['alpha83'] = (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .replace((np.inf, -np.inf), np.nan))
    
    #Alpha 084
    df['alpha84'] = (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
                       ts_delta(c, 6))))
    
    #Alpha 085
    w = 0.876703
    df['alpha85'] = (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
            .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
                              ts_rank(v, 10), 7))))

    
    #Alpha 086
    df['alpha86'] = (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
            .lt(rank(c.sub(vwap)))
            .mul(-1))
    
    #Alpha 087
    #df['alpha87'] = (max(rank(decay_linear(delta(((c * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), c, 13.4132)), 4.89768), 14.4535)) * -1) 
    
    #Alpha 088
#    s1 = (rank(ts_weighted_mean(rank(o)
#                                .add(rank(l))
#                                .sub(rank(h))
#                                .add(rank(c)), 8)))
#    s2 = ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 8),
#                                          ts_rank(ts_mean(v, 60), 20), 8), 6), 2)
#    df['alpha88'] = (s1.where(s1 < s2, s2))
    
    #Alpha 089
    #df['alpha89'] = (Ts_Rank(decay_linear(correlation(((l * 0.967285) + (l * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012)) 
    
    #Alpha 090
    #df['alpha90'] = ((rank((c - ts_max(c, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), l, 5.38375), 3.21856)) * -1) 
    
    #Alpha 091
    #df['alpha91'] = ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(c, IndClass.industry), v, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    
    #Alpha 092
#    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
#    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)
#    df['alpha92'] = (p1.where(p1<p2, p2))
    
    #Alpha 093
    #df['alpha93'] = (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((c * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664))) 
    
    #Alpha 094
    df['alpha94'] = (rank(vwap.sub(ts_min(vwap, 11)))
            .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
                                 ts_rank(ts_mean(v, 60), 4), 18), 2))
            .mul(-1))
    
    #Alpha 095
    df['alpha95'] = (rank(o.sub(ts_min(o, 12)))
            .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
                                     ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
            .astype(int))

    
    #Alpha 096
#    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
#    s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
#                                                    ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
    df['alpha96'] = (s1.where(s1 > s2, s2)
            .mul(-1))

    
    #Alpha 097
    #df['alpha97'] = ((rank(decay_linear(delta(IndNeutralize(((l * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(l, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1) 
    
    #Alpha 098
#    adv5 = ts_mean(v, 5)
#    adv15 = ts_mean(v, 15)
#    df['alpha98'] = (rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
#            .sub(rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(o),
#                                                                 rank(adv15), 20), 8), 6))))
#            .stack('ticker')
#            .swaplevel())

    
    #Alpha 099
    df['alpha99'] = ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1)))
    
    #Alpha 100
    #df['alpha100'] = (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((c - l) - (h - c)) / (h - l)) * v)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(c, rank(adv20), 5) - rank(ts_argmin(c, 30))), IndClass.subindustry))) * (v / adv20)))) 
    
    #Alpha 101
    df['alpha101'] = (c.sub(o).div(h.sub(l).add(1e-3)))

    return df