import json
import matplotlib.pyplot as plt
import numpy as np

PATH = 'Results/slow/temp_measure/220622/'

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


dyn_data = {}
time = []

for ch in range(1,9):
    dyn_data[ch] = []

for shot in range(1, 52):
    print(shot)
    data = []
    with open(PATH + str(shot) + '.json', 'r') as file:
        data = json.load(file)

    '''plt.figure()
    for ch in range(1, 9):
        plt.plot([i[ch] for i in data['data']], label=ch)
    plt.legend()
    plt.grid()'''

    if shot == 1:
        start_time = data['header']['h']*60*60 + data['header']['m']*60 + data['header']['s']

    time.append(data['header']['h']*60*60 + data['header']['m']*60 + data['header']['s'] - start_time)
    for ch in range(1, 3):
        dyn_data[ch].append(sum([i[ch] for i in data['data']]) / len([i[ch] for i in data['data']]))

    list_max = {}
    list_min = {}
    for ch in range(3, 9):
        #dyn_data[ch].append(max([i[ch] for i in data['data']]) - min([i[ch] for i in data['data']]))
        list_min[ch] = []
        list_max[ch] = []

    ds_dt = [(data['data'][i][7] - data['data'][i-1][7]) / (data['data'][i][0] - data['data'][i-1][0]) for i in range(len(data['data']))]
    if shot == 1:
        sm_coeff = 25
    else:
        sm_coeff = 5

    ds_dt_smooth = smooth(ds_dt, sm_coeff)

    '''plt.figure()
    plt.plot([i[7] for i in data['data']])
    plt.twinx()
    plt.plot(ds_dt_smooth, 'r')'''

    bound = 0.05
    start = 0
    finish = 1
    for i in range(1, len(ds_dt_smooth) - 1):
        if ds_dt_smooth[i-1] < -bound and ds_dt_smooth[i+1] > -bound and i-start > 3:
            start = i
        if ds_dt_smooth[i-1] < bound and ds_dt_smooth[i+1] > bound and i-finish > 3:
            finish = i
            for ch in range(3,9):
                list_min[ch].append(sum([i[ch] for i in data['data'][start:finish]]) / len([i[ch] for i in data['data'][start:finish]]))
    start = 0
    finish = 1
    for i in range(1, len(ds_dt_smooth) - 1):
        if ds_dt_smooth[i - 1] > bound and ds_dt_smooth[i + 1] < bound and i - start > 3:
            start = i
        if ds_dt_smooth[i - 1] > -bound and ds_dt_smooth[i + 1] < -bound and i - finish > 3:
            finish = i
            for ch in range(3, 9):
                list_max[ch].append(
                    sum([i[ch] for i in data['data'][start:finish]]) / len([i[ch] for i in data['data'][start:finish]]))

    for ch in range(3, 9):
        dyn_data[ch].append(sum(list_max[ch]) / len(list_max[ch]) - sum(list_min[ch]) / len(list_min[ch]))
        #print(ch, sum(list_max[ch]) / len(list_max[ch]) - sum(list_min[ch]) / len(list_min[ch]))

    '''print(len(list_max[7]))
    plt.figure()
    for ch in range(3,9):
        plt.plot([i[ch] for i in data['data']])
        plt.hlines(sum(list_max[ch])/ len(list_max[ch]), 0, len([i[5] for i in data['data']]))
        plt.hlines(sum(list_min[ch]) / len(list_min[ch]), 0, len([i[5] for i in data['data']]))
        plt.grid()
        plt.show()'''




plt.figure()
for ch in range(1,8):
    plt.plot([i / 60 / 60 for i in time], [i / dyn_data[ch][0] for i in dyn_data[ch]], 'o-', label=ch)
plt.legend()
plt.grid()
plt.xlabel('time, h')
plt.ylabel('Signal / Signal(0)')
#plt.savefig('Results/slow/temp_measure/220623/all_sig.png', dpi=300)

with open(PATH + 'slow_sig.json', 'w') as file2:
    data_write = {'t': time, 'data': dyn_data}
    json.dump(data_write, file2)

plt.figure()
for ch in range(1,8):
    plt.plot([i / 60 / 60for i in time], dyn_data[ch], 'o-', label=ch)
plt.legend()
plt.grid()


'''plt.figure()
for ch in range(1,8):
    plt.plot([i / max(dyn_data[2]) for i in dyn_data[2]], [i / max(dyn_data[ch]) for i in dyn_data[ch]], 'o-', label=ch)
plt.show()'''

Temp = []
for v in dyn_data[2]:
    Temp.append(-1481.96 + (2.1962e6 + (1.8639 - v/1000) / 3.88e-6)**0.5)

plt.figure()
plt.plot([i / 60 / 60 for i in time], Temp, 'o-')
plt.grid()
plt.xlabel('time, h')
plt.ylabel('T, C')
#plt.savefig('Results/slow/temp_measure/220623/temp.png', dpi=300)


plt.figure()

plt.plot(Temp, dyn_data[6], 'o-')
plt.grid()
plt.xlabel('T, C')
plt.ylabel('6 ch, 4 spectral')
#plt.savefig('Results/slow/temp_measure/220623/sig_6_temp.png', dpi=300)
plt.show()