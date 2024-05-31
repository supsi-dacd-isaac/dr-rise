from pyforecaster.trainer import cross_validate
from pyforecaster.forecaster import LinearForecaster
from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti, Fourier_es, FK, FK_multi, tune_hyperpars
from pyforecaster.metrics import nmae, make_scorer, rmse, crps
from pyforecaster.formatter import Formatter
import pandas as pd
from os.path import join
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from pyforecaster.utilities import get_logger
from tqdm import tqdm
from copy import deepcopy
logger = get_logger()

# Parameters
parallel = True
past_embedding = '7D'
pred_embedding = '1D'
n_max_series = 100
hyperpar_training_period = '30D'
k_cv = 3
max_training_period, max_test_period = '21D', '14D'
optimization_budget = 200

datasets = {'rolle': 'datasets/rolle',
            'ireland': 'datasets/ireland',
            'portugal': 'datasets/portugal',
            'london': 'datasets/london'}

def test_single_series(x, y, model_class, model_kwargs, fr, k_cv, max_training_size, max_test_size, dataset_name='ireland'):
    folds = get_folds(fr, x, k_cv, max_training_size, max_test_size)
    model = model_class(**model_kwargs)
    res_, score = cross_validate(x, y, model, folds, scoring={'nmae': make_scorer(nmae), 'rmse': make_scorer(rmse),
                                                              'crps': make_scorer(crps)}, cv_type='full', storage_fun=lambda x:x)

    # plot last predictions on each fold
    try:
        fig, ax = plt.subplots(3, 1, figsize=(10, 5), layout='tight')
        [a.plot(res_['y_hat']['fold_{}'.format(i)].values[-1, :] if isinstance(res_['y_hat']['fold_{}'.format(i)], pd.DataFrame) else res_['y_hat']['fold_{}'.format(i)][-1, :]) for i, a in enumerate(ax)]
        [a.plot(res_['y_te']['fold_{}'.format(i)].values[-1, :]) for i, a in enumerate(ax)]
        [a.set_xticks([]) for i, a in enumerate(ax[:-1])]
        ax[0].set_title('Last preds on 3 folds, model: {}, nmae: {:0.2f}'.format(model_class.__name__, score))
        now_time_dd_mm_hh = pd.Timestamp.now().strftime('%d_%m_%H')
        plt.savefig('evaluate/figs/model: {}, nmae: {:0.2f}_time_{}.png'.format(model_class.__name__, score, now_time_dd_mm_hh))
    except:
        logger.info('plotting broke')

    # remove y_hat, y_te and estimator from results before storing
    del res_['y_hat']
    del res_['y_te']
    del res_['estimator']

    temp_res = pd.DataFrame(res_).mean()
    return temp_res

def get_folds(fr, x, k_cv, max_training_size=5000, max_test_size=10000):
    folds = fr.time_kfcv(x.index, int(k_cv))
    new_folds = []
    for f in folds:
        tr_fold, te_fold = f[0], f[1]
        where_tr = np.where(tr_fold)[0]
        where_te = np.where(te_fold)[0]
        if max_training_size<len(where_tr):
            tr_fold[where_tr[:len(where_tr)-max_training_size]] = False
        if max_test_size<len(where_te):
            te_fold[where_te[max_test_size:]] = False
        new_folds.append((tr_fold, te_fold))
    return new_folds


logger.info('started')
results = {}
for dataset_name, data_path in datasets.items():
    logger.info('#'*50)
    logger.info('Dataset: {}'.format(dataset_name))
    logger.info('#'*50)

    df = pd.read_pickle(join(data_path, 'power.pk'))
    sampling_time = np.nanmedian(df.index.diff().total_seconds().values).astype(int)
    n_past_embeddings = int(pd.Timedelta(past_embedding).total_seconds() // sampling_time)
    n_pred_embeddings = int(pd.Timedelta(pred_embedding).total_seconds() // sampling_time)
    n_tr_hyperpars = int(pd.Timedelta(hyperpar_training_period).total_seconds() // sampling_time)
    max_training_size, max_test_size = int(pd.Timedelta(max_training_period).total_seconds() // sampling_time), int(pd.Timedelta(max_test_period).total_seconds() // sampling_time)

    # Keep only a random subset of the meters. 20% is used to tune hyperpars
    df_sample = df.iloc[:, np.random.choice(df.shape[1], np.minimum(int(n_max_series*1.2), df.shape[1]), replace=False)]

    # Normalize across series
    df_sample -= df_sample.mean()
    df_sample /= df_sample.std()

    # split data to define the training set for the hyperpars
    df_tr = df_sample.iloc[:, :int(df_sample.shape[1]*0.8/1.2)]
    df_hyperpars = df_sample.iloc[:, int(df_sample.shape[1]*0.8/1.2):]

    # set models pars based on dataset properties
    model_classes= {"linear": LinearForecaster,
                    "FK_multi": FK_multi,
                    "HoltWintersMulti": HoltWintersMulti,
                "Fourier_es": Fourier_es,
              "HoltWinters": HoltWinters}


    models_kwargs = {"HoltWinters": {"periods": [n_past_embeddings, n_pred_embeddings], "n_sa": n_pred_embeddings},
                        "linear": {},
                        "HoltWintersMulti": {"periods": [n_past_embeddings, n_pred_embeddings], "n_sa": n_pred_embeddings,
                                             "models_periods":[1, 2, n_pred_embeddings//2, n_pred_embeddings]},
                        "Fourier_es": {"n_sa": n_pred_embeddings, "m": n_past_embeddings},
                        "FK_multi": {"n_sa": n_pred_embeddings, "m": n_past_embeddings}}


    # format dataset using global form - add past day and past day of last week, forecast next day
    fr = Formatter(n_parallel=0).add_transform(['target'], lags=np.arange(1, n_past_embeddings-1))
    #fr.add_transform(['target'], lags=np.arange(6*n_embeddings, 7*n_embeddings))
    fr.add_target_transform(['target'], lags=-np.arange(1, n_pred_embeddings+1))
    x_all, y_all = fr.transform(df_tr, global_form=True, parallel=False, reduce_memory=False, time_features=False)

    logger.info('#'*50)
    logger.info('millions of elements in x_all: \033[91m{}\033[0m'.format(int(x_all.size/1e6)))
    logger.info('#'*50)


    # Cross validate over || models ||
    cv_results = {}
    for model_name, model_class in model_classes.items():
        logger.info('#' * 50)
        logger.info('Model: {}'.format(model_name))
        logger.info('#' * 50)


        # tune hyperpars on a subset of the sampled timeseries. The hyperpars will be fixed in the CV for all the series
        if model_name in ['HoltWinters', 'Fourier_es', 'FK_multi', 'HoltWintersMulti']:
            m_kwargs = deepcopy(models_kwargs[model_name])
            m_kwargs.update({'optimize_hyperpars': True,
                                 'optimization_budget':optimization_budget,
                                 'targets_names':df_hyperpars.columns,
                                 "target_name":df_hyperpars.columns[0]})

            model = model_class(**m_kwargs)

            model.fit(df_hyperpars.iloc[:n_tr_hyperpars, :])
            m_kwargs = deepcopy({k:v for k, v in model.get_params().items() if k in model.init_pars.keys()})
            m_kwargs = model.get_params()
            m_kwargs.update({'optimize_hyperpars': False,
                                 'target_name':'target',
                             'optimize_submodels_hyperpars':False})

        else:
            m_kwargs = models_kwargs[model_name]

        xs = []
        ys = []
        for col in df_tr.columns:
            xs.append(x_all.loc[x_all['name'] == col, [c for c in x_all.columns if c != 'name']] if model_name in [
                'linear'] else x_all.loc[x_all['name'] == col, ['target']])
            ys.append(y_all.loc[x_all['name'] == col, :])

        if parallel and model_name not in ['linear']:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=len(xs)*2) as executor:
                cv_results_ts = list(tqdm(executor.map(partial(test_single_series, model_class=model_class,
                                                          model_kwargs=deepcopy(m_kwargs), fr=fr,
                                                          k_cv=k_cv, max_training_size=max_training_size,
                                                          max_test_size=max_test_size, dataset_name=dataset_name)
                                                  , xs, ys), total=len(xs)))
                cv_results_ts = {c:res for c, res in zip(df_tr.columns, cv_results_ts)}
        else:
            cv_results_ts = {}
            for col, x, y in tqdm(zip(df_tr.columns, xs, ys), total=len(xs)):
                cv_results_ts[col] = test_single_series(x, y, model_class, deepcopy(m_kwargs), fr, k_cv, max_training_size, max_test_size)
        cv_results[model_name] = pd.concat(cv_results_ts, axis=1)

        logger.info('Dataset: {}, Model: {}, mean nmae: {}'.format(dataset_name, model_name, cv_results[model_name].loc['test_nmae'].mean()))

    results[dataset_name] = pd.concat(cv_results, axis=1)

import pickle as pk
with open('evaluate/fast_adaptive_models_results.pkl', 'wb') as f:
    pk.dump(results, f, protocol=pk.HIGHEST_PROTOCOL)


for k in results.keys():
    print('{} RMSE'.format(k))
    print(results[k].loc['test_rmse'].groupby(level=0).mean())