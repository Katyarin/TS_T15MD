import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os


shotn = 41776
polyn = 34

'''create folder to shot recording'''
path = 'Files/' + str(shotn)
try:
    os.mkdir(path)
except OSError:
    print('Не удалось создать папку')


filename = 'c:/work/Data/T-15/New_prog/%d.json' % shotn

M = 100
el_charge = 1.6 * 10 ** (-19)
G = 10
R_sv = 10000
freq = 5  # GS/s
time_step = 1 / freq  # nanoseconds
event_len = 1024


def find_start_integration(signal):
    maximum = signal.index(max(signal))
    for i in range(0, maximum):
        if signal[maximum - i] > 0 and signal[maximum - i - 1] <= 0:
            return maximum - i - 1
    return 0


def find_end_integration(signal):
    maximum = signal.index(max(signal))
    for i in range(maximum, len(signal) - 1):
        if signal[i] > 0 and signal[i + 1] <= 0:
            return i + 1
    return len(signal) - 1

delta = {0: 0, 1: 150, 2: 170, 3: 190, 4: 200, 5: 210, 6: 0}
timestamps = []
N_photo_el = {}
var_phe = {}

timeline = [i * time_step * 1e-9 for i in range(event_len)]
print(timeline)

for ch in range(7):
    N_photo_el[ch] = []
    var_phe[ch] = []


with open(filename, 'r') as file:
    raw_data = json.load(file)
plt.figure()
for event in raw_data:
    if event['t'] == 0:
        continue
    print(event['t'])
    timestamps.append(event['t']/1000)
    for ch in range(7):
        signal = event['ch'][ch]
        print(len(signal))
        if ch == 0:
            pre_sig = 100
        else:
            pre_sig = 200
        base_line = sum(signal[0:pre_sig]) / len(signal[0:pre_sig])
        for i in range(len(signal)):
            signal[i] = signal[i] - base_line
        if ch == 0:
            index_0 = 0
            p = 0
            for i, s in enumerate(signal):
                if s > 0.250:
                    index_0 = i - 20
                    break
        plt.plot(signal)
        if index_0 == 0:

            plt.show()
            stop
        var_in_sr = np.var(signal[0:pre_sig])
        var_test = sum([(i - sum(signal[0:pre_sig]) / len(signal[0:pre_sig])) ** 2 for i in signal[0:pre_sig]]) / len(signal[0:pre_sig])
        print('var from var: ', var_in_sr)
        print('var from formula: ', var_test)
        '''start_index = find_start_integration(signal)
        end_index = find_end_integration(signal)'''
        start_index = index_0 + delta[ch]
        end_index = start_index + 200
        if max(signal) > 0.2:
            plt.vlines(start_index, 0, 1, 'r')
            plt.vlines(end_index, 0, 1, 'g')
        Ni = np.trapz(signal[start_index:end_index],
                            timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5)
        N_photo_el[ch].append(Ni)
        var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr * 1e6 - 1.14e4 * 0.0625) + math.fabs(Ni) * 4)
        print('Ni: ', Ni, 'var: ', var)
        var_phe[ch].append(var)
plt.figure(figsize=(10, 3))
plt.title('Shot #' + str(shotn))
for ch in N_photo_el.keys():
    #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
    #if ch != 0:
    #plt.errorbar([i for i in range(len(N_photo_el[ch]))], N_photo_el[ch], yerr=var_phe[ch], label='ch' + str(ch))
    plt.plot(timestamps, N_photo_el[ch], '^-', label='ch' + str(ch))
plt.ylabel('N, phe')
plt.xlabel('time')
plt.legend()
plt.show()

with open('Files/' + str(shotn) + '/' + 'N_phe.json', 'w') as f:
    for_temp = {'timeline': timestamps, 'data': N_photo_el, 'err': var_phe}
    json.dump(for_temp, f)

