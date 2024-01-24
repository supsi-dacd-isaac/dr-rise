# --------------------------------------------------------------------------------------------------------------------
# To download the datasets, we use the Kaggle API. To use it, you need to create an account on Kaggle and download
# your API credentials. You need to genrate a new token and download the kaggle.json file. Then, you need to move
# the file to the .kaggle folder in your home directory under ./kaggle.
# If the folder does not exist, you need to create it.
# You can find more information here: https://www.kaggle.com/docs/api
# --------------------------------------------------------------------------------------------------------------------

from os.path import join, exists
from os import mkdir
from urllib.request import urlretrieve
import requests
from zipfile import ZipFile
from os import listdir
import os

from urllib.parse import urlparse


urls = {'electricity_hourly': 'https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1',
        'rolle': 'https://zenodo.org/records/3463137/files/power_data.p?download=1'}

for name, url in urls.items():
    # make a directory with the name of the dataset, if not there
    data_dir = join('datasets', name)
    if not exists(data_dir):
        mkdir(data_dir)

    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    name += ext


    print('Downloading {}...'.format(name))
    urlretrieve(url, join(data_dir, name))
    # get list of all files in data_dir
    files = listdir(data_dir)
    # unzip the files if needed
    for file in files:
        if file.endswith('.zip'):
            print('Unzipping {}...'.format(file))
            with ZipFile(join(data_dir, file), 'r') as zipObj:
                zipObj.extractall(data_dir)
    print('Done!')

