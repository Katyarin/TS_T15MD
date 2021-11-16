import json
import numpy as np
import matplotlib.pyplot as plt
import math

shot_N = 40827

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
    shot_options = json.load(f)

N_pages_total = shot_options["N_pages_get"]

#delta_with_combiscope = 3.2
delta_with_combiscope = 0

M = 100
el_charge = 1.6 * 10 ** (-19)
G = 10
R_sv = 10000 #Ом

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

timeline = []
end_time = 0
N_photo_el = {}
N_plot = {}
var_phe = {}
var_plot = {}

for n_file in range(0, N_pages_total + 50, 50):
    if N_pages_total - n_file > 50:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                n_file + 50) + '.json', 'r') as f:
            read_data = json.load(f)
    elif 50 > N_pages_total - n_file > 0:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                N_pages_total - n_file) + '.json', 'r') as f:
            read_data = json.load(f)
    freq = 5  # GS/s
    time_step = 1 / freq  # nanoseconds
    event_len = 1024
    timeline_prototype = [0]
    while len(timeline_prototype) != event_len:
        timeline_prototype.append((timeline_prototype[-1] + time_step)) #in seconds

    times = np.array(read_data['timestampsSelection'])

    if n_file == 0:
        start_times = times[0]
    times = times - start_times
    times = times * 2E-8 * 1000  # relative timestamps in ms
    times = times + delta_with_combiscope
    timeline.extend(times)

    p = 1
    for ch in range(6):
        if n_file == 0:
            N_photo_el[ch] = []
            N_plot[ch] = []
            var_phe[ch] = []
            var_plot[ch] = []
        for page in range(50):
            signal = read_data['data'][ch][page*1024:(page+1)*1024]
            if len(signal) != 0:
                base_line = sum(signal[0:200]) / len(signal[0:200])
                for i in range(len(signal)):
                    signal[i] = signal[i] - base_line
                var_in_sr = np.var(signal[0:400])
                start_index = find_start_integration(signal)
                end_index = find_end_integration(signal)
                integration_timeline = [i * (10 ** (-9)) for i in timeline_prototype]
                Ni = np.trapz(signal[start_index:end_index],
                             integration_timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5)
                if max(signal) > 0.95:
                    N_photo_el[ch].append(None)
                    N_plot[ch].append(float('NaN'))
                    var_phe[ch].append(None)
                    var_plot[ch].append(float('NaN'))
                else:
                    N_photo_el[ch].append(Ni)
                    N_plot[ch].append(Ni)
                    var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr * 1e6 - 1.14e4 * 0.0625) + math.fabs(Ni) * 4)
                    var_phe[ch].append(var)
                    var_plot[ch].append(var)
                #k = 6715 * 0.0625 (фотоэлектронов^2 / mv^2)
                #b = -1.14e4 * 0.0625(фотоэлектронов^2)




        p += 1
print(timeline)

p = 1
plt.figure(figsize=(10, 3))
plt.title('Shot #' + str(shot_N))
for ch in N_photo_el.keys():
    #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
    if ch != 0:
        plt.errorbar([i for i in range(len(N_plot[ch]))] , N_plot[ch], yerr=var_plot[ch], label='ch' + str(ch))
        #plt.plot(timeline, N_photo_el[ch], '^-', label='ch' + str(ch))
plt.ylabel('N, phe')
plt.xlabel('time')
plt.legend()
plt.show()



with open('Files/' + str(shot_N) + '/' + 'N_phe.json', 'w') as f:
    for_temp = {'timeline': timeline, 'data': N_photo_el, 'err': var_phe}
    json.dump(for_temp, f)

print('end')