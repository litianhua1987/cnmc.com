import pandas as pd
import quandl as qua
import os
from dotenv import load_dotenv, find_dotenv

from src import utils

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


root_path = os.environ.get("LOCAL_PATH")
data_path = os.path.join(root_path, "data", "")
raw_path = os.path.join(data_path, "raw", "")
interim_path = os.path.join(data_path, "interim", "")
QUANDL_KEY = os.environ.get("QUANDL_KEY")


conf = utils.get_conf(path=root_path)

def get_quandl_prices(ticker: str, start_date: str, end_date: str, *args, **kwargs):

    renaming = kwargs.get('renaming', {})
    freq = kwargs.get('freq', None)

    df_raw = qua.get(ticker, start_date=start_date, end_date=end_date, authtoken=QUANDL_KEY).rename(columns=renaming)
    df_raw.index.freq = freq


    return df_raw

def get_cnmc_petromkt_data(start_date: str, end_date: str, *args, **kwargs):

    create_date = lambda yr, mon, day:\
        pd.to_datetime(10000 * yr.astype(float) + 100 * mon.astype(float) + day, format='%Y%m%d')

    cf_cnmc_data = conf['files']['raw']['cnmc']
    cnmc_raw_path = os.path.join(raw_path, cf_cnmc_data['file'])

    freq = cf_cnmc_data['freq']
    cons_renaming = {'GASÓLEO A': 'cons_GOA',
                     'GASOLINA  AUTO. S/PB 95 I.O.': 'cons_GNA95',
                     'GASOLINA  AUTO. S/PB 98 I.O.': 'cons_GNA98'}

    cons_select_cols = ['cons_GOA', 'cons_GNA95', 'cons_GNA98']

    month_parser = pd.read_excel(cnmc_raw_path, sheet_name='month_dict').set_index('MES')['month']
    cons = pd.read_excel(cnmc_raw_path, sheet_name='Con')
    cons = cons[cons['MES'] != 'ANUAL'].copy()
    cons['MES'] = cons['MES'].replace(month_parser).astype(int)
    cons['Date'] = cons[['AÑO', 'MES']].apply(lambda x: create_date(yr=x[0], mon=x[1], day=1), axis=1)

    cons.rename(inplace=True, columns=cons_renaming)
    cons.set_index('Date', inplace=True)
    cons.index.freq = freq

    return cons.loc[start_date: end_date, cons_select_cols].copy()

def create_comm_df(df_dict: dict, freq: str = 'B'):


    df_lst = [df.rename(columns={'Close': ticker}).resample(freq).interpolate() for ticker, df in df_dict.items()]
    comm_df = pd.concat(df_lst, join='inner', axis=1)

    return comm_df


def read_mkt_data():
    comm_intm_path = os.path.join(interim_path, conf['files']['interim']['market'])

    comm_df = pd.read_csv(comm_intm_path, parse_dates=['Date'], index_col='Date')

    return comm_df


def read_cnmc_data():
    cnmc_intm_path = os.path.join(interim_path, conf['files']['interim']['cnmc'])

    cnmc_df = pd.read_csv(cnmc_intm_path, parse_dates=['Date'], index_col='Date')

    return cnmc_df