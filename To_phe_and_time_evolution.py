import json
import numpy as np
import matplotlib.pyplot as plt

shot_N = 39846

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
    shot_options = json.load(f)

N_pages_total = shot_options["N_pages_get"]

delta_with_combiscope = 3.2


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

for n_file in range(0, N_pages_total + 50, 50):
    if N_pages_total - n_file > 50:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                n_file + 50) + '.json', 'r') as f:
            read_data = json.load(f)
    elif 50 > N_pages_total - n_file > 0:
        with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(n_file) + '_to_' + str(
                N_pages_total - n_file) + '.json', 'r') as f:
            read_data = json.load(f)
    freq = 3.2  # GS/s
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
    #if n_file == 0:
        #delta_with_combiscope = 9.85 - times[1]
        #delta_times = delta_with_combiscope - times[1]
        #print(delta_with_combiscope)
    times = times + delta_with_combiscope
    timeline.extend(times)
    print(times)

    #plt.figure(figsize=(10, 16))
    p = 1
    for ch in range(8):
        #plt.subplot(4, 2, p)
        #plt.title(str(n_file) + '_channel #' + str(ch + 1))
        if n_file == 0:
            N_photo_el[ch] = []
        for page in range(50):
            signal = read_data['data'][ch][page*1024:(page+1)*1024]
            if len(signal) != 0:
                base_line = sum(signal[0:200]) / len(signal[0:200])
                for i in range(len(signal)):
                    signal[i] = signal[i] - base_line
                    #timeline_prototype[i] = timeline_prototype[i] * (10 ** (-9))
                #var = np.sqrt(np.var(signal[0:200]))
                start_index = find_start_integration(signal)
                end_index = find_end_integration(signal)
                #plt.plot(timeline_prototype, signal, alpha=0.3)
                #plt.axvline(timeline_prototype[start_index], color='r')
                #plt.axvline(timeline_prototype[end_index], color='g')
                integration_timeline = [i * (10 ** (-9)) for i in timeline_prototype]
                N_photo_el[ch].append(np.trapz(signal[start_index:end_index],
                             integration_timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5))
        p += 1
    #plt.show()
#timeline_for_phe = [i for i in range(0, (len(N_photo_el[0])) * delta_times, delta_times)]
#print(timeline_for_phe)

#combiscope_time = [i + (delta_with_combiscope - timeline_for_phe[2]) for i in timeline_for_phe]
print(timeline)
p = 1
plt.figure(figsize=(20, 6))
for ch in N_photo_el.keys():
    plt.title('Shot #' + str(shot_N))
    #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
    if ch != 0:
        plt.plot(timeline, N_photo_el[ch], 'r^-', label='ch' + str(ch))
    plt.ylabel('N, phe')
    plt.xlabel('time')
    plt.legend()
    plt.show()
    #plt.savefig(path + 'Shot #' + str(shot_N) +', ' 'ch #' + str(ch), dpi=600)
#plt.savefig(path + 'Shot #' + str(shot_N), dpi=600)

'''with open('Files/' + start_options["data"] + '/' + str(shot_N) + 'N_phe.json') as f:
    for_temp = {'data': N_photo_el, 'timeline': timeline}
    json.dump(for_temp, f)'''