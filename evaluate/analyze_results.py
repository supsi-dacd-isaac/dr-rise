import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
res = pd.read_pickle('evaluate/fast_adaptive_models_results.pkl')

def error_dist_plot(df, score_key, save_path=None):
    sns.set()

    df = df.loc[score_key, :]
    df = df.reset_index()
    df.rename(columns={'level_0': 'index', 'level_1': 'time_series'}, inplace=True)
    ordered_names = df[['index', score_key]].groupby('index').mean().sort_values(by=score_key, ascending=False).index


    fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout='tight')
    sns.boxplot(x=score_key, y='index', data=df, ax=ax, order=ordered_names,
                notch=True, showcaps=False,
                boxprops={"facecolor": (.3, .5, .7, .5)},
                medianprops={"color": "r", "linewidth": 2},
                )
    ax.set_title('Error distribution')
    ax.set_ylabel('Model')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


score_key = 'test_rmse'
for k in res.keys():
    error_dist_plot(res[k], score_key, save_path='evaluate/figs/{}_error_dist.png'.format(k))