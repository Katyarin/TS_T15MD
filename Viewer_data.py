import json
import numpy as np
import matplotlib.pyplot as plt


shot_N = 39491
N_pages_total = 292
path = 'c:/work/Data/Эксперименты с плазмой/Полихроматор 34/2020.11.12 (первые сигналы с плазмы)/Данные/'

def find_start_integration(signal):
    maximum = signal.index(max(signal))
    print(max(signal))
    for i in range(0, maximum):
        if signal[maximum - i] > 0 and signal[maximum - i - 1] <= 0:
            return maximum - i - 1

for n_file in range(0, N_pages_total, 50):
    with open(path + str(shot_N) + '_' + str(n_file) + '_to_' + str(n_file + 50) +'.json', 'r') as f:
        read_data = json.load(f)

    plt.figure(figsize=(10, 16))
    p = 1
    for i in range(8):
        plt.subplot(4, 2, p)
        plt.title(str(n_file) + '_channel #'+str(i+1))
        #for page in range(49):
        page = 21
        signal = read_data['data'][i][page*1024:(page+1)*1024]
        base_line = sum(signal[0:200]) / len(signal[0:200])
        for k in range(len(signal)):
            signal[k] = signal[k] - base_line
        start_index = find_start_integration(signal)
        plt.plot(signal, alpha=0.3)
        p += 1

    plt.show()