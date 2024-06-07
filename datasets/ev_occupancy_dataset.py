import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient
import urllib3
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import json
import seaborn as sns

urllib3.disable_warnings()


def do_query(qy, dfc):
    try:
        res_df = dfc.query(qy)
    except Exception as e:
        print('EXCEPTION: %s' % str(e))
        return None
    return res_df

def replace_key(key):
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], tuple):
        return key[1][0][1]  # Extract the value from the nested tuple
    else:
        return key

def results_to_df(results):
    new_dict = dict(map(lambda item: (replace_key(item[0]), item[1]), results.items()))
    combined_df = pd.concat([df.assign(key=key) for key, df in new_dict.items()], ignore_index=False).reset_index(drop=False)
    combined_df.rename(columns={'index': 'time'}, inplace=True)
    return combined_df


def query_availability(df_client, db_conf, lat_min, lat_max, lon_min, lon_max, t_start, t_end, dt='15m'):
    """
    :param df_client:
    :param db_conf:
    :param lat_min:
    :param lat_max:
    :param lon_min:
    :param lon_max:
    :param t_start:
    :param t_end:
    :param state: 0=AVAILABLE, 1=OCCUPIED
    :param dt:
    :return:
    """

    # query one day at a time
    results_list = []
    for t in tqdm(pd.date_range(t_start, t_end, freq='D')):
        t_start = t.tz_localize(None).isoformat() + "Z"
        t_end = (t + pd.to_timedelta('1d')).tz_localize(None).isoformat() + "Z"

        query = (f"SELECT MEAN(value) "
                 f"FROM {db_conf['measurement']} "
                 f"WHERE time>='{t_start}' "
                 f"AND time<'{t_end}' "
                 f"AND latitude>={lat_min} "
                 f"AND latitude<={lat_max} "
                 f"AND longitude>={lon_min} "
                 f"AND longitude<={lon_max} "
                 f"GROUP BY time({dt}), evse_id"
                 )

        results = do_query(query, df_client)
        if results is None:
            continue
        else:
            results_list.append(results_to_df(results))

    df = pd.concat(results_list)
    df.reset_index(drop=True, inplace=True)
    df['time'] = pd.Index(df['time']).tz_convert('Europe/Zurich')

    return df

def get_client(cfg_path='conf/dbs_conf.json'):
    with open(cfg_path) as f:
        db_conf = json.load(f)['evs']
    client = DataFrameClient(host=db_conf['host'], port=db_conf['port'], password=db_conf['password'],
                    username=db_conf['user'], database=db_conf['database'], ssl=db_conf['ssl'])
    return client, db_conf

def get_data(pars):
    df_client, db_conf = get_client()
    t_start = pd.Timestamp(pars['t_start']).isoformat() + "Z"
    t_end = pd.Timestamp(pars['t_end']).isoformat() + "Z"
    data = query_availability(df_client, db_conf, pars['lat_min'], pars['lat_max'], pars['lon_min'], pars['lon_max'], t_start, t_end, dt=pars['dt'])
    return data


if __name__ == '__main__':

    # BBOX Swiss coordinates
    #lat_max, lon_max = 47.808464, 10.492294
    #lat_min, lon_min = 45.817920, 5.956085

    # BBOX
    lat_min, lat_max = 46.00, 46.03
    lon_min, lon_max = 8.91, 8.96

    # Time range and sampling time
    t_start = '2024-05-14'
    t_end = '2024-05-15'
    dt = '15m'

    pars = {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max,
                         't_start': t_start, 't_end': t_end, 'dt':dt}

    data = get_data(pars)

    now = pd.Timestamp.now()
    data.to_pickle('datasets/ev_occupancy_{}.zip'.format(now.strftime('%Y-%m-%d, %H-%M-%S')))
    pd.DataFrame(pars, index = [0]).to_pickle('datasets/pars_{}.pk'.format(now.strftime('%Y-%m-%d, %H-%M-%S')))

    # plot raw data
    plt.figure()
    for n in np.unique(data['key']):
        plt.plot(data.loc[data['key'] == n, ['time', 'mean']].set_index("time"))
    plt.show()