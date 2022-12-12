from fin_ratios import add_fin_ratios_and_commodities, add_risk_measures
from alpha101 import add_artificial_variables
from main import add_classic_indicators, get_stock_data, generate_lagged_variables

def get_all_stock_indicators(stock_ticker, drop_first_rows = 252, max_missing_data = 30):
    data = get_stock_data(stock_ticker)
    data = add_classic_indicators(data)
    data = generate_lagged_variables(data)
    data = add_artificial_variables(data)
    data = add_risk_measures(data, data['Returns'])
    data = add_fin_ratios_and_commodities(data)
    data = data.iloc[drop_first_rows:,:]
    print(f'dropped {data.columns[data.isna().sum()>max_missing_data]}')
    data.drop(columns = data.columns[data.isna().sum()>max_missing_data], inplace=True)
    data = data.ffill().backfill()
    return data

