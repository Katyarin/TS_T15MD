import json
import numpy as np
import matplotlib.pyplot as plt


shot_N = '39545'
N_pages_total = 59
#path = 'c:/work/Data/Эксперименты с плазмой/Полихроматор 34/2020.11.12 (первые сигналы с плазмы)/Данные/'

for n_file in range(0, N_pages_total, 50):
    with open(str(shot_N) + '_' + str(n_file) + '_to_' + str(n_file + 50) +'.json', 'r') as f:
        read_data = json.load(f)

    plt.figure(figsize=(10, 16))
    p = 1
    for i in range(8):
        plt.subplot(4, 2, p)
        plt.title(str(n_file) + '_channel #'+str(i+1))
        for page in range(50):
            if n_file + page < N_pages_total:
                signal = read_data['data'][i][page*1024:(page+1)*1024]
                base_line = sum(signal[0:200]) / len(signal[0:200])
                for k in range(len(signal)):
                    signal[k] = signal[k] - base_line
                plt.plot(signal, alpha=0.3)
        p += 1

    plt.show()