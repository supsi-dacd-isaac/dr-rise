from os.path import join, exists, isdir
from tqdm import tqdm
import os
import shutil
import zipfile
import pandas as pd
from os import remove
import missingno as msno
import matplotlib.pyplot as plt

def keggle_data_preprocessing():
    base_path = 'datasets/kaggle_smart_meters_london/'
    zip_filepath = join(base_path,'archive.zip')

    block_list = ['block_{}.csv'.format(i) for i in range(111)]
    csv_list = block_list + ['weather_hourly_darksky.csv']
    # unzip all the files from data path which are in csv_list
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for filename in tqdm(csv_list):
            with zip_ref.open(join('halfhourly_dataset/halfhourly_dataset/', filename)) as zf, open(join(base_path, filename), 'wb') as f:
                shutil.copyfileobj(zf, f)



    # read all the csv files, parse them, concat them and save them as a single compresed dataframe
    n_splits = 10
    files_per_split = len(block_list) // n_splits
    df_list = []
    k = 0
    # read weather data and keep only the float columns
    weather = pd.read_csv(join(base_path, 'weather_hourly_darksky.csv'), index_col=3, parse_dates=True, na_values=('Null', 'nan'))
    weather = weather.loc[:, weather.dtypes == 'float64']
    weather = weather.resample('30min').mean().ffill()
    for filename in tqdm(block_list):
        df = pd.read_csv(join(base_path, filename), index_col=1, parse_dates=True, na_values=('Null', 'nan'))
        df_list.append(df)
        if len(df_list) == files_per_split:
            df = pd.concat(df_list, axis=0)
            df.to_pickle(join(base_path, 'data_{}.zip'.format(k)))

            df_matrix = df.reset_index().pivot(index='tstp', columns='LCLid', values='energy(kWh/hh)')
            df_matrix = df_matrix[df_matrix.isna().sum().sort_values().index]
            df_matrix = pd.concat([df_matrix, weather.loc[(weather.index>=df_matrix.index[0]) & (weather.index<=df_matrix.index[-1])]], axis=1)
            msno.matrix(df_matrix, freq='10D', fontsize=6)
            plt.savefig(join(base_path, 'missing_data_{}.png'.format(k)))
            df_matrix.to_pickle(join(base_path, 'data_matrix_{}.zip'.format(k)))
            df_list = []
            k += 1

    # delete all the csv files in csv_list
    for filename in tqdm(csv_list):
        remove(join(base_path, filename))

