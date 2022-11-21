#@title Choose your experiments { run: "auto" }

#@markdown Choose algorithms
qmix = True #@param {type:"boolean"}
qmc = False #@param {type:"boolean"}
vdn = False #@param {type:"boolean"}
vdnc = False #@param {type:"boolean"}
iql = False #@param {type:"boolean"}
coma = False #@param {type:"boolean"}
qtran = False #@param {type:"boolean"}

#@markdown Choose environments
three_marines = False #@param {type:"boolean"}
eight_marines = False #@param {type:"boolean"}
twentyfive_marines = False #@param {type:"boolean"}
three_stalkers_five_zealots = False #@param {type:"boolean"}
mpe_SimpleAdversary_v0 = True #@param {type:"boolean"}

# Fetch form data
envs = []
if three_marines: envs.append('3m')
if eight_marines: envs.append('8m')
if twentyfive_marines: envs.append('25m')
if three_stalkers_five_zealots: envs.append('3s5z')
if mpe_SimpleAdversary_v0: envs.append('mpe_SimpleAdversary-v0')

algos = []
if qmix: algos.append('qmix')
if qmc: algos.append('qmc')
if vdn: algos.append('vdn')
if vdnc: algos.append('vdc')
if iql: algos.append('iql')
if coma: algos.append('coma')
if qtran: algos.append('qtran')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy.signal import lfilter
from pathlib import Path
# from google.colab import files
import os
import glob
from zipfile import ZipFile
import os
from os.path import basename

sns.set()
def extract_plot_data(path, label):
    # # # Fetch all paths to info.json of every commited experiment
    # # for path in Path('../results').rglob('*info*.json'):
    # # Extract label from path
    # label = extract_label(path, algos+envs)
    #
    # # If the path corresponds to a env or an algo we want to plot
    # if is_algo(path, algos) and is_env(path, envs):
    #     # Read json
    plot_data = {}
    metrics = []
    with open(path/'info.json') as json_file:
        data = json.load(json_file)
        for metric in data:
            # Transposed corresponds to xs values. We identify metrics by ys
            if '_T' not in metric and '_std' not in metric:
                # If we have not already seen this metric in a json -> add to plots
                if metric not in metrics:
                    metrics.append(metric)

                # Get data for this metric and the current experiment
                transposed_identifier = metric+'_T'
                xs = data[transposed_identifier]
                ys = data[metric]

                # If data is return, then plot data with std
                if 'return' in metric:
                    transposed_identifier_std = metric.replace('mean','std')
                    stds = data[transposed_identifier_std]

                # Save the experiment data in the plot dict and provide label
                if metric in plot_data:
                    if 'return' not in metric:
                        plot_data[metric].append((xs, ys, label))
                    else:
                        plot_data[metric].append((xs, ys, label, stds))
                else:
                    plot_data[metric] = []
                    if 'return' not in metric:
                        plot_data[metric].append((xs, ys, label))
                    else:
                        plot_data[metric].append((xs, ys, label, stds))
    return plot_data, metrics

# Smoothing parameters
# n = 30  # the larger n is, the smoother curve will be
# b = [1.0 / n] * n
# a = 1
# not_smooth = ['epsilon', 'td_error_abs', 'loss']
def plot(plot_data, metrics, path):
    for metric in metrics:
        # Exclude for now since weird jsons are returned
        if metric == 'grad_norm':
            continue

        for i, data in enumerate(plot_data[metric]):
            # If plot data is hidden in a dict property 'value' -> unpack
            ys = list(map(lambda x: x['value'] if isinstance(x,dict) else x, data[1]))
            xs = data[0]

            if 'return' in metric:
                y_std = list(map(lambda x: x['value'] if isinstance(x,dict) else x, data[3]))
                plt.plot(xs, ys, label=data[2])
                plt.fill_between(xs, [a - b for a,b in zip(ys,y_std)], [a + b for a,b in zip(ys,y_std)],
                                alpha=0.15)
            else:
                # Confidence interval if we have multiple values
                sns.lineplot(x=xs, y=ys, errorbar=('ci', 95), label=data[2])

            plt.xlabel('steps')
            # plt.xlim(right=2000000) # Experiments end at 2 million steps
            metric_label = metric.replace('_', ' ')
            plt.ylabel(metric_label)

        plt.legend(loc='upper left')
        if not (path / 'plots').exists():
            os.makedirs(path / 'plots')
        plt.savefig(path / 'plots'/ (metric+'.png'))
        plt.show()

# def is_env(path, envs):
#     if len(envs) == 0:
#         return True
#     # Test for all paths if they correspond to a selected environment
#     return any(env in str(path) for env in envs)
#
# def is_algo(path, algos):
#     if len(algos) == 0:
#         return True
#     # Test for all paths if they correspond to a selected algorithm
#     return any(algo in str(path) for algo in algos)
#
# def extract_label(path, search):
#     # Match folder name which will serve as label
#     return re.search(".*(results/)(.*)(/source).*",str(path))[2]

if __name__ == '__main__':
    # main
    alg = "qmix"
    env = "mpe_SimpleAdversary-v0"
    run_id = "3"
    path = Path("results/sacred")/alg/env/run_id
    plot_data, metrics = extract_plot_data(path, alg)
    print(metrics)
    plot(plot_data, metrics, path)

    # # Delete previous zip to prevent rezipping
    # if os.path.isfile("/content/plots.zip"):
    #     os.remove("/content/plots.zip")

    # # Create a ZipFile
    # with ZipFile('plots.zip', 'w') as zip:
    #     # Iterate over all the files in directory
    #     print("Zipping:", filename)
    #     for folderName, subfolders, filenames in os.walk("/content/plots/"):
    #         for filename in filenames:
    #             print("Adding:", filename)
    #             #create complete filepath of file in directory
    #             filePath = os.path.join(folderName, filename)
    #             # Add file to zip
    #             zip.write(filePath, basename(filePath))
    #     assert zip.testzip() is None
    #
    #
    #
    # # Download
    # files.download("/content/plots.zip")
    #
    # # Clear plots
    # files = glob.glob('/content/plots/*')
    # for f in files:
    #     os.remove(f)