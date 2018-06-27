import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#%%

os.makedirs('figures', exist_ok=True)
#%%
dataframes = []
episodes = []
for repeat in [0,1,2,4]:
    data = pd.read_csv('logs/control-{}.csv'.format(repeat))
    data['repeat'] = repeat
    dataframes.append(data)
    episodes.append((max(data['episodes'])))
datas = pd.concat(dataframes)
episodes = min(episodes)
datas = datas[datas['episodes'] < episodes].reset_index(drop=True)
ylabels = ['Episode Reward', 'Reward validation', 'State validation']
for metric, ylabel in zip(['total_rewards', 'reward_val', 'state_val'], ylabels):
    fig, ax = plt.subplots(figsize=(7, 7))
    if metric is not 'total_rewards':
        ax.set(yscale="log")
    sns.tsplot(data=datas, value=metric, unit='repeat', time='episodes',err_style="unit_traces", ax=ax)
    ax.set(xlabel="Episodes")
    ax.set(ylabel=ylabel)
    fig.savefig('figures/{}.pdf'.format(ylabel), bbox_inches='tight')

#%%
mean_rewards, mean_reward_val, mean_state_val = np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))
for repeat in [0,1,2,4]:
    mean_rewards = mean_rewards + np.array(datas[datas['repeat'] == repeat]['total_rewards'])
    mean_reward_val = mean_reward_val + np.array(datas[datas['repeat'] == repeat]['reward_val'])
    mean_state_val = mean_state_val + np.array(datas[datas['repeat'] == repeat]['state_val'])
mean_rewards = mean_rewards/4
mean_reward_val = mean_reward_val/4
mean_state_val = mean_state_val/4
print(np.argmax(mean_rewards), np.max(mean_rewards))
print(np.argmin(mean_reward_val))
print(np.argmin(mean_state_val))

#%%
def plot_experiments(experiments, name):
    fig, ax = plt.subplots(figsize=(7, 7))
    metric = 'total_rewards'
    dataframes = []
    episodes = []
    for experiment in experiments:
        data = pd.read_csv('logs/{}.csv'.format(experiment))
        data['repeat'] = experiment.split('-')[-1]
        data['condition'] = experiment.split('-')[0]
        dataframes.append(data)
        episodes.append((max(data['episodes'])))
    datas = pd.concat(dataframes)
    episodes = min(episodes)
    datas = datas[datas['episodes'] < episodes].reset_index(drop=True) 
    sns.tsplot(data=datas, value=metric, unit='repeat', time='episodes', ax=ax, condition='condition')   
    plt.show()
    ax.set(xlabel="Episodes")
    ax.set(ylabel='Episode Reward')
    fig.savefig('figures/{}.pdf'.format(name), bbox_inches='tight')
    
plot_experiments(experiments = ['control-{}'.format(i) for i in range(5)] + ['H5-{}'.format(i) for i in range(5)] + ['H10-{}'.format(i) for i in range(5)], name='changingH')
plot_experiments(experiments = ['control-{}'.format(i) for i in range(5)] + ['K500-{}'.format(i) for i in range(5)] + ['K250-{}'.format(i) for i in range(5)], name='changingK')
plot_experiments(experiments = ['control-{}'.format(i) for i in range(5)] + ['hu250-{}'.format(i) for i in range(5)] + ['hu125-{}'.format(i) for i in range(5)], name='changinghu')

#%%
# =============================================================================
# Policy gradients
# =============================================================================

def running_mean(x, N):
    x = list(x)
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%
# learning rate experiments with H=20, K=1000
experiments = ['reinforce-lr-0.00001', 'reinforce-lr-0.000001','reinforce-lr-0.0000001']
metric = 'total_rewards'
labels = [r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$']
N=10
fig, ax = plt.subplots(1, 2, figsize = (12,7), sharey=True)
for experiment, label in zip(experiments, labels):
    data = pd.read_csv('logs/{}.csv'.format(experiment))
    ax[0].plot(running_mean(data[metric], N), label = label, alpha = 0.8, linewidth=1)
    print(data.shape[0])

experiments = ['reinforce-H2-K500-hu125-lr1e-5', 'reinforce-H2-K500-hu125-lr1e-6','reinforce-H2-K500-hu125-lr1e-7']
metric = 'total_rewards'
labels = [r'$10^{-5}$', r'$10^{-6}$', r'$10^{-7}$']
for experiment, label in zip(experiments, labels):
    data = pd.read_csv('logs/{}.csv'.format(experiment))
    ax[1].plot(running_mean(data[metric], N), label = label, alpha = 0.8, linewidth=1)
    print(data.shape[0])
ax[0].legend(loc=2)
ax[0].set(xlim=[0,10000]), ax[1].set(xlim=[0,10000])
ax[0].set(xlabel="Episodes"), ax[1].set(xlabel="Episodes")
ax[0].set(ylabel='Episode Reward')
ax[0].set(title='Large network and trajectories')
ax[1].set(title='Small network and trajectories')
plt.show()
fig.savefig('figures/{}.pdf'.format('learningrates'), bbox_inches='tight')

#%%


N=10
metric = 'total_rewards'
experiments = ['reinforce-H2-K1000-hu125-lr1e-7-bs32-shuffle', 'reinforce-H2-K1000-hu125-lr1e-7-bs32', 'reinforce-H2-K1000-hu125-lr1e-7', 'reinforce-H2-K1000-hu125-lr1e-7-shuffle']
labels = ['B32-Shuffle', 'B32', 'Control', 'Shuffle']
fig, ax = plt.subplots(figsize = (7,7))
for experiment, label in zip(experiments, labels):
    data = pd.read_csv('logs/{}.csv'.format(experiment))
    ax.plot(running_mean(data[metric], N), label = label, alpha = 0.8, linewidth=1)
    print(data.shape[0])
ax.legend()
ax.set(xlim=[0,15000])
handles, labels = ax.get_legend_handles_labels()
handles[2], handles[3] = handles[3], handles[2]
labels[2], labels[3] = labels[3], labels[2]
ax.legend(handles, labels)
ax.set(xlabel="Episodes")
ax.set(ylabel='Episode Reward')
plt.show()
fig.savefig('figures/{}.pdf'.format('batchshuffle'), bbox_inches='tight')

# =============================================================================
# HK robustness
# =============================================================================
#%% 

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

import json
data = json.load(open('results/HKrobust.json'))
xs, ys = [], []
for key in data:
    x, y = key.split('-')
    xs.append(int(x))
    ys.append(int(y))
xs, ys = sorted(list(set(xs))), sorted(list(set(ys)))
array = np.zeros((len(xs), len(ys)))
for key in data:
    x, y = key.split('-')
    x, y = int(x), int(y)
    xi = np.where(np.array(xs) == x)[0][0]
    yi = np.where(np.array(ys) == y)[0][0]
    array[xi][yi] = float(data[key])
    
fig, ax = plt.subplots(figsize = (6,6))
im, cbar = heatmap(array, xs, ys, ax = ax, cbarlabel = 'Episode Reward')
fig.savefig('figures/{}.pdf'.format('HKrobustness'), bbox_inches='tight')

#%%

# =============================================================================
# Finetune 
# =============================================================================

fig, ax = plt.subplots()
N=5
metric = 'total_rewards'
for suffix in ['','-H2', '-H4']:
    lengths = []
    for i in range(5):
        prefix = 'finetune-control-{}'.format(i)
        fname = '{}{}'.format(prefix, suffix)
        data = list(pd.read_csv('logs/{}.csv'.format(fname))[metric])
        lengths.append(len(data))
    length = min(lengths)
    all_data = np.zeros((length,))
    for i in range(5):
        prefix = 'finetune-control-{}'.format(i)
        fname = '{}{}'.format(prefix, suffix)
        data = list(pd.read_csv('logs/{}.csv'.format(fname))[metric])[:length]
        all_data = all_data + np.array(data)
    all_data = all_data/5
    ax.plot(running_mean(all_data, N), alpha = 0.8, linewidth=1, label = suffix)
        #print(data.shape)
ax.legend()

#%%
N=25
episodes = []
datas = []
H = {'' : 20, '-H4' : 4, '-H2' : 2}
for i in range(5):
    prefix = 'finetune-control-{}'.format(i)
    for suffix in H:
        fname = '{}{}'.format(prefix, suffix)
        data = pd.read_csv('logs/{}.csv'.format(fname))
        data['H'] = H[suffix]
        data['repeat'] = i
        datas.append(pd.rolling_mean(data, N))
        episodes.append(max(data['episodes']))
#episodes = min(episodes)
data = pd.concat(datas)      
#data = data[data['episodes'] < episodes].reset_index(drop=True)  
fig, ax = plt.subplots()
sns.tsplot(data=data, value=metric, unit='repeat', time='episodes', ax=ax, condition='H', ci="sd")   
ax.set(xlabel="Episodes")
ax.set(ylabel='Episode Reward')
plt.show()
fig.savefig('figures/{}.pdf'.format('finetune'), bbox_inches='tight')
