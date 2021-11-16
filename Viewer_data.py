import json
import numpy as np
import matplotlib.pyplot as plt

shot_N = 40827

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
    shot_options = json.load(f)

N_pages_total = shot_options["N_pages_get"]
# path = 'c:/work/Data/Эксперименты с плазмой/Полихроматор 34/2020.11.12 (первые сигналы с плазмы)/Данные/'

for n_file in range(0, N_pages_total, 50):
    if N_pages_total - n_file > 50:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                n_file + 50) + '.json', 'r') as f:
            read_data = json.load(f)
    elif 50 > N_pages_total - n_file > 0:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                N_pages_total - n_file) + '.json', 'r') as f:
            read_data = json.load(f)

    plt.figure(figsize=(18, 12))
    p = 1
    for i in range(8):
        plt.subplot(3, 3, p)
        plt.title(str(n_file) + '_channel #' + str(i + 1))
        for page in range(50):
            if n_file + page < N_pages_total:
                signal = read_data['data'][i][page * 1024:(page + 1) * 1024]
                base_line = sum(signal[0:200]) / len(signal[0:200])
                for k in range(len(signal)):
                    signal[k] = signal[k] - base_line
                plt.plot(signal, alpha=0.3)
        p += 1

    plt.show()
