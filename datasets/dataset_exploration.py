import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.dates as mdates

def plot_stats_and_sample(df, title, steps_to_plot=300, series_to_plot=None, ax1=None, ax2=None):
    series_to_plot = series_to_plot or len(df.columns)
    stats = pd.concat([df.mean(), df.std()], axis=1)
    stats.columns = ['means', 'stds']
    ax1.boxplot(stats)
    ax1.set_title(title)


    (df/df.std()).iloc[:steps_to_plot, :series_to_plot].plot(alpha=0.1, legend=None, ax=ax2)
    ax2.set_xticks([])
    ax2.set_xticklabels([])


data_paths = {'portugal':'datasets/portugal/power.pk',
              'rolle': 'datasets/rolle/power.pk',
              'london':'datasets/london/power.pk',
              'ireland':'datasets/ireland/power.pk'}


# save both fig and fig2
fig, ax = plt.subplots(1, len(data_paths.keys()), figsize=(10, 10))
fig2, ax2 = plt.subplots(len(data_paths.keys()), 1, figsize=(10, 5))
plt.gca()
fig2.subplots_adjust(wspace=0.3, left=0.05, right=0.98, top=0.95, bottom=0.05)



for k, a1, a2 in zip(data_paths.keys(), ax, ax2):
    df = pd.read_pickle(data_paths[k])
    steps_to_plot = int(24*7/df.index.diff().median().total_seconds()*3600)
    plot_stats_and_sample(df, title=k, steps_to_plot=steps_to_plot, series_to_plot=200, ax1=a1, ax2=a2)


plt.savefig('datasets/figures/time_series.png')
plt.close(fig2)
plt.savefig('datasets/figures/stats.png')


