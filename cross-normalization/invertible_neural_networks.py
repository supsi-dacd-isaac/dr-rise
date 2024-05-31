from pyforecaster.forecasting_models.neural_models.INN import CausalInvertibleNN
from pyforecaster.forecasting_models.neural_models.base_nn import FFNN
from pyforecaster.forecaster import LinearForecaster
from pyforecaster.trainer import cross_validate
from pyforecaster.formatter import Formatter
from pyforecaster.metrics import nmae, make_scorer
import pandas as pd
from os.path import join
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
embedding = '7D'
n_max_series = 100
k_cv = 3

# read cfg file
pars = pd.read_json('conf/cross_normalize_conf.json')

# List of datasets
data_paths = {'ireland':'datasets/ireland',
                'rolle':'datasets/rolle',
              'portugal':'datasets/portugal',
              'london':'datasets/london',
              }

# Loop over || datasets ||
results = {}
for dataset_name, data_path in data_paths.items():
    df = pd.read_pickle(join(data_path, 'power.pk'))
    df = df.resample('1h').mean()
    df = df.dropna()
    sampling_time = np.nanmedian(df.index.diff().total_seconds().values).astype(int)
    n_embeddings = int(pd.Timedelta(embedding).total_seconds() // sampling_time)

    # Keep only a random subset of the meters
    df_sample = df.iloc[:, np.random.choice(df.shape[1], np.minimum(n_max_series, df.shape[1]), replace=False)]

    # Normalize across series
    df_sample -= df_sample.mean()
    df_sample /= df_sample.std()

    # format dataset using global form - add past day and past day of last week, forecast next day
    fr = Formatter(n_parallel=0).add_transform(['target'], lags=np.arange(1, 3*n_embeddings-1))
    #fr.add_transform(['target'], lags=np.arange(6*n_embeddings, 7*n_embeddings))
    fr.add_target_transform(['target'], lags=-np.arange(1, n_embeddings))
    x, y = fr.transform(df_sample, global_form=True, parallel=False, reduce_memory=False, time_features=False)

    # reorder x columns to be consistent with causal filter (from far away in the past to the present)
    x = x[[c for c in np.sort(x.columns)[::-1] if c.startswith('target')]]

    # Create the models
    pars['forecasters_kwargs']['ffnn']['n_out'] = y.shape[1]
    pars['forecasters_kwargs']['ffnn']['n_hidden_x'] = x.shape[1]
    pars['forecasters_kwargs']['cinn']['n_out'] = y.shape[1]
    pars['forecasters_kwargs']['cinn']['n_hidden_x'] = x.shape[1]
    pars['forecasters_kwargs']['cinn']['unnormalized_inputs'] = x.columns
    cinn_pars = pars['forecasters_kwargs']['cinn']
    cinn_pars_quasi_end_to_end = cinn_pars.copy()
    cinn_pars_quasi_end_to_end.update({'end_to_end':'quasi'})
    cinn_pars_full_end_to_end = cinn_pars.copy()
    cinn_pars_full_end_to_end.update({'end_to_end': 'full'})
    models = {'cinn':CausalInvertibleNN(**cinn_pars_quasi_end_to_end),
        'cinn_end_to_end':CausalInvertibleNN(**cinn_pars_full_end_to_end),
              'ffnn':FFNN(**pars['forecasters_kwargs']['ffnn']),
              'linear':LinearForecaster()}

    # Cross validate over || models ||
    cv_results = {}
    for model_name, model in tqdm(models.items()):
        cv_res, score = cross_validate(x, y, model, fr.time_kfcv(x.index, int(k_cv)), scoring=make_scorer(nmae), cv_type='full')
        cv_results[model_name] = pd.DataFrame(cv_res)
        plt.show()
        print('Model {} on {} dataset - score: {:.2f} '.format(model_name, dataset_name, score))
    plt.close('all')
    results[dataset_name] = pd.concat(cv_results, axis=1)

# Save results
pd.to_pickle(results, 'cross-normalization/results.pkl')