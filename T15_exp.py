import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import requests


#shotn = 41783
#shots = list(range(41744, 41755)) + list(range(41756, 41761)) + [41764, 41765, 41766, 41768] + \
        #list(range(41770, 41773)) + list(range(41775, 41781)) + [41782, 41783, 41786]
shots = [41770]
print(shots)
polyn = 34


URL = 'https://172.16.12.130:443/api'
TSpath = 'TS_core/'

def TS_res(dev, N_phe, sigma, i):
    ne = []
    chi = []
    for j in range(len(dev['ch']['1'])):
        num = 0
        den = 0
        for ch in dev['ch'].keys():
            if N_phe[ch][i] == None:
                continue
            num += N_phe[ch][i] * dev['ch'][ch][j] / math.pow(sigma[ch][i], 2)
            den += dev['ch'][ch][j] * dev['ch'][ch][j] / math.pow(sigma[ch][i], 2)
        ne_loc = num / den
        ne.append(num / den)
        chi_local = 0
        for ch in dev['ch'].keys():
            if N_phe[ch][i] == None:
                continue
            chi_local += (N_phe[ch][i] - ne_loc * dev['ch'][ch][j]) ** 2 / math.pow(sigma[ch][i],2)
        chi.append(chi_local)
    chi_min = min(chi)
    index = chi.index(chi_min)
    if index == len(dev['Te']) - 1:
        index = len(dev['Te']) - 2
        #print('index out of range')
    Te = dev['Te'][chi.index(chi_min)]
    ne_res = ne[chi.index(chi_min)]

    '''sigma_Te'''
    dN_dTe = {}
    for ch in dev['ch'].keys():
        dN_dTe[ch] = np.diff([i * ne_res for i in dev['ch'][ch]]) / np.diff(dev['Te'])
        #dN_dTe[ch].append(dev['ch'][ch][-1] * ne_res)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for ch in dev['ch'].keys():
        if N_phe[ch][i] == None:
            continue
        sum1 += (dev['ch'][ch][index] / sigma[ch][i]) ** 2
        sum2 += (dN_dTe[ch][index] / sigma[ch][i]) ** 2
        sum3 += dN_dTe[ch][index] / sigma[ch][i] * dev['ch'][ch][index] / sigma[ch][i]
    sum3 = sum3 ** 2
    sigma_Te = (sum1 / (sum1 * sum2 - sum3)) ** 0.5

    return Te, ne_res, chi_min, sigma_Te


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


def to_phe(shotn):
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

    timestamps = []
    N_photo_el = {}
    var_phe = {}
    timeline = [i * time_step * 1e-9 for i in range(event_len)]
    #print(timeline)

    for ch in range(7):
        N_photo_el[ch] = []
        var_phe[ch] = []

    with open(filename, 'r') as file:
        raw_data = json.load(file)

    for event in raw_data:
        timestamps.append(event['t']/1000)
        for ch in range(7):
            signal = event['ch'][ch]
            if ch == 0:
                pre_sig = 100
            else:
                pre_sig = 200
            base_line = sum(signal[0:pre_sig]) / len(signal[0:pre_sig])
            for i in range(len(signal)):
                signal[i] = signal[i] - base_line
            var_in_sr = np.var(signal[0:pre_sig])
            start_index = find_start_integration(signal)
            end_index = find_end_integration(signal)
            Ni = np.trapz(signal[start_index:end_index],
                                timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5)
            N_photo_el[ch].append(Ni)
            var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr * 1e6 - 1.14e4 * 0.0625) + math.fabs(Ni) * 4)
            var_phe[ch].append(var)
    '''plt.figure(figsize=(10, 3))
    plt.title('Shot #' + str(shotn))
    for ch in N_photo_el.keys():
        #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
        #if ch != 0:
        plt.errorbar([i for i in range(len(N_photo_el[ch]))], N_photo_el[ch], yerr=var_phe[ch], label='ch' + str(ch))
        #plt.plot(timestamps, N_photo_el[ch], '^-', label='ch' + str(ch))
    plt.ylabel('N, phe')
    plt.grid()
    plt.xlabel('time')
    plt.legend()
    #plt.show()'''

    with open('Files/' + str(shotn) + '/' + 'N_phe.json', 'w') as f:
        for_temp = {'timeline': timestamps, 'data': N_photo_el, 'err': var_phe}
        json.dump(for_temp, f)


def Temp(shot_N, poly):
    if shot_N < 41600:
        '''reading_options'''
        with open('config.json', 'r') as file:
            start_options = json.load(file)

        with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
            shot_options = json.load(f)

    '''waiting signals'''
    with open('dev_num_' + str(poly) + '.json', 'r') as file:
        dev_num = json.load(file)

    '''data in phe'''
    with open('Files/' + '/' + str(shot_N) + '/' + 'N_phe.json') as fp:
        data = json.load(fp)

    N_phe = data['data']
    timeline = data['timeline']
    sigma = data['err']

    Te = []
    Te_err = []
    ne = []
    chi2 = []
    start_temp = 18
    end_temp = 65
    for i in range(len(N_phe['0'])):
        # for i in range(start_temp, end_temp):
        Te_loc, ne_loc, chi2_loc, Te_err_loc = TS_res(dev_num, N_phe, sigma, i)
        '''if Te_err_loc/Te_loc < 0.5:
            plt.figure()
            plt.title(timeline[i])
            for ch in dev_num['ch'].keys():
                plt.plot(dev_num['Te'], [i * ne_loc for i in dev_num['ch'][ch]], label=ch)
                plt.scatter(Te_loc, N_phe[ch][i])
            #plt.ylim(0, max([i * ne_loc for i in dev_num['ch']['4']]) * 1.25)
            plt.legend()
        #plt.show()'''
        Te.append(Te_loc)
        ne.append(ne_loc)
        chi2.append(chi2_loc)
        Te_err.append(Te_err_loc)

    '''print(Te_err)
    plt.figure()
    plt.plot(timeline, Te, 'ro-')
    plt.grid()
    plt.ylim(0, 2000)'''

    '''plt.figure()
    plt.errorbar(timeline, Te, yerr=Te_err)
    plt.grid()
    plt.ylim(0, 2000)

    plt.figure()
    plt.plot(timeline, ne)
    plt.grid()

    plt.figure()
    plt.plot(timeline, chi2)
    plt.grid()'''

    with open('Files/' + str(shot_N) + '/' + 'Te.json', 'w') as res_file:
        for_temp = {'timeline': timeline, 'Te': Te, 'Te_err': Te_err}
        json.dump(for_temp, res_file)

    return {'timeline': timeline, 'Te': Te, 'Te_err': Te_err, 'chi2': chi2}

def TS_temp(shot_N, poly):
    with open(TSpath + str(shot_N) + '.json', 'r') as fp:
        TSdata = json.load(fp)

    N_phe = {}
    sigma = {}
    for ch in range(1,5):
        N_phe[str(ch)] = [event['poly'][poly]['ch'][ch-1]['ph_el'] for event in TSdata['data'] if event['error'] == None]
        sigma[str(ch)] = [event['poly'][poly]['ch'][ch-1]['err'] for event in TSdata['data'] if event['error'] == None]
    timeline = [event['timestamp'] for event in TSdata['data'] if event['error'] == None]


    with open(TSpath + '2021.10.19_1064.4.json', 'r') as fc:
        div_ts = json.load(fc)

    dev_num = {}
    dev_num['Te'] = div_ts['T_arr']
    dev_num['ch'] = {}
    for ch in range(1,5):
        dev_num['ch'][str(ch)] = div_ts['poly'][int(poly)]['expected'][ch-1]

    Te = []
    Te_err = []
    ne = []
    chi2 = []
    start_temp = 18
    end_temp = 65
    for i in range(len(timeline)):
        # for i in range(start_temp, end_temp):
        Te_loc, ne_loc, chi2_loc, Te_err_loc = TS_res(dev_num, N_phe, sigma, i)
        '''p = 0
        if Te_err_loc/Te_loc < 0.3:
            plt.figure()
            plt.title(timeline[i])
            for ch in dev_num['ch'].keys():
                plt.plot(dev_num['Te'], [i * ne_loc for i in dev_num['ch'][ch]], label=ch)
                plt.scatter(Te_loc, N_phe[ch][i])
            #plt.ylim(0, max([i * ne_loc for i in dev_num['ch']['4']]) * 1.25)
            plt.legend()
            p+=1
            if p > 20:
                plt.show()
        #plt.show()'''
        Te.append(Te_loc)
        ne.append(ne_loc)
        chi2.append(chi2_loc)
        Te_err.append(Te_err_loc)

    return {'timeline': timeline, 'Te': Te, 'Te_err': Te_err, 'chi2': chi2, 'expected': div_ts}
"""___________________________________________________________________________________________________________________"""


TS = {'Te': [], 'err': []}
T15 = {'Te': [], 'err': [], 'chi2': []}
p = 0
for shotn in shots:
    response = requests.post(url=URL, verify=False, json={
        'subsystem': 'db',
        'reqtype': 'get_shot',
        'shotn': int(shotn)
    })
    to_phe(shotn)

    try:
        data = response.json()
        with open('dump.json', 'w') as file:
            json.dump(data, file)
    except:
        print('Not a json?')

    TS_by_me = TS_temp(shotn, '9')
    plt.vlines(data['data']['events'][69]['T_e'][9]['T'], 0, 10000)
    plt.vlines(TS_by_me['expected']['T_arr'][data['data']['events'][69]['T_e'][9]['index']], 0, 500)
    #plt.show()


    T15_data = Temp(shotn, polyn)


    #delta = [i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][8]['error'] == None][0]
    delta = 3.632

    Te_plot = []
    time_plot = []
    err_plot = []

    for j in range(len(T15_data['Te'])):
        if T15_data['Te_err'][j] / T15_data['Te'][j] < 0.5:
            #time_plot.append([i + delta for i in list(reversed(T15_data['timeline']))][j])
            time_plot.append(T15_data['timeline'][j] + delta)
            Te_plot.append(T15_data['Te'][j])
            err_plot.append(T15_data['Te_err'][j])


    plt.figure()
    for poly in range(10):
        if poly < 9:
            continue
        #plt.plot([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], [i['T_e'][poly]['T'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], label=poly)
        plt.errorbar([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None],
                 [TS_by_me['expected']['T_arr'][i['T_e'][poly]['index']] for i in data['data']['events'] if
                  i['error'] == None and i['T_e'][poly]['error'] == None], yerr=[i['T_e'][poly]['Terr'] for i in data['data']['events'] if
                  i['error'] == None and i['T_e'][poly]['error']== None], label=poly)
        for i, j in enumerate(data['data']['events']):
            try:
                print(i, j['T_e'][poly]['chi2'], TS_by_me['chi2'][i+1], T15_data['chi2'][i+1])
            except KeyError:
                pass
    #plt.errorbar([i + delta for i in list(reversed(T15_data['timeline']))], T15_data['Te'], T15_data['Te_err'], label='T15 34')
    plt.errorbar(time_plot, Te_plot, yerr=err_plot, label='T15 34')
    plt.errorbar(TS_by_me['timeline'], TS_by_me['Te'], fmt='--', yerr=TS_by_me['Te_err'], label='TS by me')
    plt.ylim(0,2000)
    plt.legend()
    plt.grid()
    plt.xlabel('time, ms')
    plt.ylabel('Te, eV')

    plt.figure()
    plt.hist(TS_by_me['chi2'], bins=10, range=(0,20))

    plt.figure()
    plt.hist(T15_data['chi2'], bins=10, range=(0,20))
    print(shotn)
    print(len(data['data']['events']))
    print(len(T15_data['Te']))
    index_start = 45
    for i in range(45, 76):
        if data['data']['events'][i]['T_e'][9]['error'] == None and data['data']['events'][i+1]['T_e'][9]['error'] == None:
            index_start = i
            break
    index_end = 76
    for i in range(50, 76):
        if data['data']['events'][i]['T_e'][9]['error'] != None:
            index_end = i
            break
    print(index_start, index_end)
    if index_end - index_start < 10:
        print('no plasma?')
        continue

    '''plt.figure()
    plt.plot([data['data']['events'][i]['T_e'][9]['T'] for i in range(index_start, index_end)])
    plt.plot([T15_data['Te'][i] for i in range(index_start, index_end)])'''

    TS['Te'].extend([data['data']['events'][i]['T_e'][9]['T'] for i in range(index_start, index_end)])
    TS['err'].extend([data['data']['events'][i]['T_e'][9]['Terr'] for i in range(index_start, index_end)])
    T15['Te'].extend([T15_data['Te'][i] for i in range(index_start, index_end)])
    T15['err'].extend([T15_data['Te_err'][i] for i in range(index_start, index_end)])
    T15['chi2'].extend([T15_data['chi2'][i] for i in range(index_start, index_end)])
    p+=1
    if p > 5:
        p = 0
        plt.show()
print(p)
plt.figure()
plt.errorbar(TS['Te'], T15['Te'], yerr=T15['err'], xerr=TS['err'], fmt='.')
plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.plot([i for i in range(2000)], [i for i in range(2000)])
plt.xlabel('TS 10 poly, eV')
plt.ylabel('T15, eV')
plt.grid()

with open('220330_for_comparison.txt', 'w') as file2:
    for i in range(len(TS['Te'])):
        file2.write(' %14.4f' % TS['Te'][i])
        file2.write(' %14.4f' % TS['err'][i])
        file2.write(' %14.4f' % T15['Te'][i])
        file2.write(' %14.4f' % T15['err'][i])
        file2.write(' %14.4f' % T15['chi2'][i])
        file2.write('\n')


'''plt.figure()
for event in range(len(data['data']['events'])):
    plt.plot()'''
plt.show()