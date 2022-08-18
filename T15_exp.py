import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import requests
from pathlib import Path

shots = [42048]
G = 10

plot_osc = False
plot_exp_sig = False
time_for_start_plot = 150 #ms

print(shots)
polyn = 34
thomson = 'ust' #or 'usual' or 'ust' or 'divertor'

sp_cal = '22.06.27'

bound1 = 180
bound2 = 195
ne_corr_mult_all = 1
ne_corr_mult_T15 = 5.87e40

URL = 'https://172.16.12.130:443/api'
TSpath = 'TS_core/'

def to_json(shotn, ch_count=6):
    # ! use only Your own local copy of raw data files!
    path = Path('c:/work/Data/T-15/New_prog/%d' % shotn)
    if not path.is_file():
        print('not found')

    data = [{
        't': 0,
        'ch': [[0 for cell in range(1024)] for ch in range(ch_count + 1)]
    }]
    with path.open(mode='r') as file:
        count = 0
        event = {
            't': 0,
            'ch': [[] for ch in range(ch_count + 1)]
        }
        for line in file:
            if count > 1023:
                count += 1
                if count == 1026:
                    data.append(event.copy())
                    event = {
                        't': 0,
                        'ch': [[] for ch in range(ch_count + 1)]
                    }
                    count = 0
                continue
            sp = line.split()

            for ch in range(ch_count + 1):
                event['ch'][ch].append(float(sp[1 + ch]))
            if count == 0:
                event['t'] = int(sp[-2])
            count += 1

    with open(str(path) + '.json', 'w') as file:
        json.dump(data, file)

    print('Code OK')

def TS_res(dev, N_phe, sigma, i, err):
    ne = []
    chi = []
    for j in range(len(dev['ch']['1'])):
        num = 0
        den = 0
        for ch in dev['ch'].keys():
            if err[ch][i] == 'off scale':
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
        sum1 += (dev['ch'][ch][index] * ne_res / sigma[ch][i]) ** 2
        sum2 += (dN_dTe[ch][index] / sigma[ch][i]) ** 2
        sum3 += dN_dTe[ch][index] / sigma[ch][i] * dev['ch'][ch][index] * ne_res / sigma[ch][i]
    sum3 = sum3 ** 2
    sigma_Te = (sum1 / (sum1 * sum2 - sum3)) ** 0.5
    sigma_ne = ne_res * (sum2 / (sum1 * sum2 - sum3)) ** 0.5

    return Te, ne_res, chi_min, sigma_Te, sigma_ne


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
    #G = 10
    R_sv = 10000
    freq = 5  # GS/s
    time_step = 1 / freq  # nanoseconds
    event_len = 1024

    delta = {0: 0, 1: 150, 2: 170, 3: 190, 4: 200, 5: 210, 6: 0}
    delta_divertor = {0: 0, 1: -480, 2: -470, 3: -460, 4: -450, 5: -440, 6: -25}
    timestamps = []
    N_photo_el = {}
    var_phe = {}
    timeline = [i * time_step for i in range(event_len)]
    calc_err = {}
    #print(timeline)

    for ch in range(7):
        N_photo_el[ch] = []
        var_phe[ch] = []
        calc_err[ch] = []

    with open(filename, 'r') as file:
        raw_data = json.load(file)

    p = 0
    for event in raw_data:
        timestamps.append(event['t']/1000 + 3.73)

        #if max(event['ch'][1]) > 0.030:
        if event['t']/1000 + 3.73 > time_for_start_plot and plot_osc:
            fig2, axs2 = plt.subplots(3, 3)
            fig2.suptitle(event['t']/1000 + 3.73)
            p = 1
        for ch in range(7):
            signal = event['ch'][ch]
            if max(signal) > 0.8:
                calc_err[ch].append('off scale')
                print(event['t']/1000 + 3.73)
            else:
                calc_err[ch].append('')
            if ch == 0:
                pre_sig = 100
            else:
                pre_sig = 200
            base_line = sum(signal[0:pre_sig]) / len(signal[0:pre_sig])
            for i in range(len(signal)):
                signal[i] = signal[i] - base_line

            if ch == 0:
                index_0 = 0
                for i, s in enumerate(signal[10:]):
                    if s > 0.250:
                        index_0 = i - 20
                        #print(index_0)
                        break

            for i in range(len(signal)):
                signal[i] = signal[i] * 1000
            var_in_sr = np.var(signal[0:pre_sig])
            delta_exp = {}
            if thomson == 'usual':
                width = 100
                delta_exp[ch] = delta[ch]
            elif thomson == 'ust':
                width = 250
                delta_exp[ch] = delta[ch] - 100
            elif thomson == 'divertor':
                width = 100
                delta_exp[ch] = delta_divertor[ch]
            else:
                print('something wrong! Unnown config')
                stop
            start_index = index_0 + delta_exp[ch]
            end_index = start_index + width


            if p:
                #print(max(event['ch'][5]), event['t']/1000)
                axs2[int(ch//3), int(ch%3)].set_title('ch = ' + str(ch))
                axs2[int(ch//3), int(ch%3)].plot(signal)
                axs2[int(ch//3), int(ch%3)].vlines(start_index, min(signal), max(signal))
                axs2[int(ch//3), int(ch%3)].vlines(end_index, min(signal), max(signal))


            Ni = np.trapz(signal[start_index:end_index],
                                timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5)
            N_photo_el[ch].append(Ni *1e-12)
            var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr - 1.14e4 * 0.0625) + math.fabs(Ni *1e-12) * 4)
            var_phe[ch].append(var)
        plt.show()
        p=0
    plt.figure(figsize=(10, 3))
    plt.title('Shot #' + str(shotn))
    for ch in N_photo_el.keys():
        #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
        if ch != 0:
            plt.errorbar(timestamps, N_photo_el[ch], yerr=var_phe[ch], label='ch' + str(ch))
            plt.scatter([t for i, t in enumerate(timestamps) if calc_err[ch][i] == 'off scale'],
                        [j for i, j in enumerate(N_photo_el[ch]) if calc_err[ch][i] == 'off scale'], marker='x', s=40, c='black')
        #plt.plot(timestamps, N_photo_el[ch], '^-', label='ch' + str(ch))
    plt.ylabel('N, phe')
    plt.grid()
    plt.xlabel('time')
    plt.legend()
    #plt.show()

    with open('Files/' + str(shotn) + '/' + 'N_phe.json', 'w') as f:
        for_temp = {'timeline': timestamps, 'data': N_photo_el, 'err': var_phe, 'culc_err': calc_err}
        json.dump(for_temp, f)


def Temp(shot_N, poly):
    if shot_N < 41600:
        '''reading_options'''
        with open('config.json', 'r') as file:
            start_options = json.load(file)

        with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
            shot_options = json.load(f)

    '''waiting signals'''
    with open('source/' + sp_cal + '_dev_num_' + str(poly) + '.json', 'r') as file:
        dev_num = json.load(file)

    '''data in phe'''
    with open('Files/' + '/' + str(shot_N) + '/' + 'N_phe.json') as fp:
        data = json.load(fp)

    N_phe = data['data']
    timeline = data['timeline']
    sigma = data['err']
    culc_errors = data['culc_err']

    Te = []
    Te_err = []
    ne = []
    ne_err = []
    chi2 = []
    start_temp = 18
    end_temp = 65
    for i in range(len(N_phe['0'])):
        # for i in range(start_temp, end_temp):
        Te_loc, ne_loc, chi2_loc, Te_err_loc, ne_err_loc = TS_res(dev_num, N_phe, sigma, i, culc_errors)
        if plot_exp_sig and timeline[i] > time_for_start_plot:
        #if Te_err_loc/Te_loc < 0.2:
            plt.figure()
            plt.title(timeline[i])
            for ch in dev_num['ch'].keys():
                plt.plot(dev_num['Te'], [i * ne_loc for i in dev_num['ch'][ch]], label=ch)
                if culc_errors[ch][i] == 'off scale':
                    plt.scatter(Te_loc, N_phe[ch][i], marker='x')
                else:
                    plt.scatter(Te_loc, N_phe[ch][i])
            #plt.ylim(0, max([i * ne_loc for i in dev_num['ch']['4']]) * 1.25)
            plt.legend()
            plt.show()
        Te.append(Te_loc)
        ne.append(ne_loc)
        chi2.append(chi2_loc)
        Te_err.append(Te_err_loc)
        ne_err.append(ne_err_loc)

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

    return {'timeline': timeline, 'Te': Te, 'Te_err': Te_err, 'chi2': chi2, 'ne': ne, 'ne_err': ne_err}

def TS_temp(shot_N, poly):
    with open(TSpath + str(shot_N) + '.json', 'r') as fp:
        TSdata = json.load(fp)

    N_phe = {}
    sigma = {}
    timeline = []
    for ch in range(1,5):
        N_phe[str(ch)] = []
        sigma[str(ch)] = []
        for event in TSdata['data']:
            if event['error'] == None:
                N_phe[str(ch)].append(event['poly'][poly]['ch'][ch-1]['ph_el'])
                sigma[str(ch)].append(event['poly'][poly]['ch'][ch-1]['err'])
            else:
                N_phe[str(ch)].append(0)
                sigma[str(ch)].append(1000)
    for event in TSdata['data']:
        if event['error'] == None:
            timeline.append(event['timestamp'])
        else:
            timeline.append(0)


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
    ne_err = []
    chi2 = []
    start_temp = 18
    end_temp = 65
    for i in range(len(timeline)):
        # for i in range(start_temp, end_temp):
        Te_loc, ne_loc, chi2_loc, Te_err_loc, ne_err_loc = TS_res(dev_num, N_phe, sigma, i)
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
        ne_err.append(ne_err_loc)

    return {'timeline': timeline, 'Te': Te, 'Te_err': Te_err, 'chi2': chi2, 'expected': div_ts, 'ne': ne, 'ne_err': ne_err}
"""___________________________________________________________________________________________________________________"""


TS = {'Te': [], 'Te_err': [], 'ne': [], 'ne_err': [], 'chi2': []}
T15 = {'Te': [], 'Te_err': [], 'ne': [], 'ne_err': [], 'chi2': []}
p = 0
for shotn in shots:
    to_json(shotn)
    response = requests.post(url=URL, verify=False, json={
        'subsystem': 'db',
        'reqtype': 'get_shot',
        'shotn': int(shotn)
    })
    to_phe(shotn)

    try:
        data = response.json()
        if data['ok'] == False:
            print(data['description'])
            plt.show()
        with open('dump.json', 'w') as file:
            json.dump(data, file)
    except:
        print('Not a json?')

    #TS_by_me = TS_temp(shotn, '9')
    '''plt.vlines(data['data']['events'][69]['T_e'][9]['T'], 0, 10000)
    plt.vlines(TS_by_me['expected']['T_arr'][data['data']['events'][69]['T_e'][9]['index']], 0, 500)
    #plt.show()'''


    T15_data = Temp(shotn, polyn)
    print(T15_data.keys())
    print(len(T15_data['timeline']))
    #print(len(data['data']['events']))
    plt.figure()
    plt.errorbar([t for i, t in enumerate(T15_data['timeline']) if T15_data['Te_err'][i] / T15_data['Te'][i] < 0.7],
                 [t for i, t in enumerate(T15_data['Te']) if T15_data['Te_err'][i] / T15_data['Te'][i] < 0.7],
                 yerr=[t for i, t in enumerate(T15_data['Te_err']) if T15_data['Te_err'][i] / T15_data['Te'][i] < 0.7], ls='--')
    plt.ylim(0, 2000)
    plt.grid()
    plt.xlabel('time, ms')
    plt.ylabel('Te, eV')

    plt.figure()
    plt.errorbar([t for i, t in enumerate(T15_data['timeline'])], T15_data['ne'], yerr=T15_data['ne_err'])
    #plt.ylim(0, 2000)
    plt.grid()
    plt.xlabel('time, ms')
    plt.ylabel('ne')
    #plt.show()

    if thomson == 'usual' or thomson == 'ust':
        for i, event in enumerate(data['data']['events']):
            if event['error'] == None:
                data['data']['events'][i]['T_e'].insert(10, {})

                event['T_e'][10]['T'] = T15_data['Te'][i]
                event['T_e'][10]['Terr'] = T15_data['Te_err'][i]
                event['T_e'][10]['chi2'] = T15_data['chi2'][i]
                event['T_e'][10]['n'] = T15_data['ne'][i] * ne_corr_mult_T15
                event['T_e'][10]['n_err'] = T15_data['ne_err'][i] * ne_corr_mult_T15
                if T15_data['Te_err'][i] / T15_data['Te'][i] < 0.5:
                    event['T_e'][10]['error'] = None
                else:
                    event['T_e'][10]['error'] = "high Te error"
                '''else:
                    event['T_e'][10]['error'] = "second run"'''

        poly_list = [str(i) for i in range(10)] + ['T15-34'] + [11]
        print(poly_list[11])

        fig, axs = plt.subplots(2,2)
        for poly in range(12):
            '''if poly < 9:
                continue'''
            #plt.plot([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], [i['T_e'][poly]['T'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], label=poly)
            axs[0, 0].errorbar([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None],
                     [i['T_e'][poly]['T'] for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error'] == None], yerr=[i['T_e'][poly]['Terr'] for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error']== None], label=poly_list[poly])
        axs[0, 0].set_ylim(0,2000)
        axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_xlabel('time, ms')
        axs[0, 0].set_ylabel('Te, eV')
        #axs[0, 0].twiny()
        #axs[0, 0].plot([0 for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], alpha=0)

        #plt.figure()
        for poly in range(12):
            '''if poly < 9:
                continue'''
            #plt.plot([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], [i['T_e'][poly]['T'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], label=poly)
            axs[0, 1].errorbar([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None],
                     [i['T_e'][poly]['n'] *ne_corr_mult_all for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error'] == None], yerr=[i['T_e'][poly]['n_err'] *ne_corr_mult_all for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error']== None], label=poly_list[poly])
        #plt.ylim(0,2000)
        axs[0, 1].legend()
        axs[0, 1].grid()
        axs[0, 1].set_xlabel('time, ms')
        #axs[0, 1].set_color_cycle(sns.color_palette("coolwarm_r"))


        fibers = ['1A', '1B'] + [str(i) for i in list(range(2, 10))] + ['VA_LFS'] + ['10']


        #plt.figure()
        for event in data['data']['events']:
            if event['error'] == None and bound1 < event['timestamp'] < bound2:
                axs[1, 0].errorbar([data['data']['config']['fibers'][i]['R'] for poly, i in enumerate(fibers) if event['T_e'][poly]['error'] == None],
                         [event['T_e'][poly]['T']for poly in range(12) if event['T_e'][poly]['error'] == None],
                             yerr=[event['T_e'][poly]['Terr']for poly in range(12) if event['T_e'][poly]['error'] == None])

        axs[1, 0].grid()
        axs[1, 0].set_xlabel('R, mm')
        axs[1, 0].set_ylabel('Te, eV')

        #plt.figure()
        for event in data['data']['events']:
            if event['error'] == None and bound1 < event['timestamp'] < bound2:
                axs[1, 1].errorbar([data['data']['config']['fibers'][i]['R'] for poly, i in enumerate(fibers) if
                              event['T_e'][poly]['error'] == None],
                             [event['T_e'][poly]['n']*ne_corr_mult_all for poly in range(12) if event['T_e'][poly]['error'] == None],
                             yerr=[event['T_e'][poly]['n_err']*ne_corr_mult_all for poly in range(12) if event['T_e'][poly]['error'] == None])

        axs[1, 1].grid()
        axs[1, 1].set_xlabel('R, mm')
        axs[1, 1].set_ylabel('ne, m^-3')





    '''for i in range(len(T15_data['Te'])):
        if T15_data['Te'][i] > 100 and TS_by_me['Te'][i] > 100 and T15_data['Te_err'][i]/T15_data['Te'][i] < 0.3:
            TS['Te'].append(TS_by_me['Te'][i])
            TS['Te_err'].append(TS_by_me['Te_err'][i])
            TS['ne'].append(TS_by_me['ne'][i])
            TS['ne_err'].append(TS_by_me['ne_err'][i])
            TS['chi2'].append(TS_by_me['chi2'][i])

            T15['Te'].append(T15_data['Te'][i])
            T15['Te_err'].append(T15_data['Te_err'][i])
            T15['ne'].append(T15_data['ne'][i])
            T15['ne_err'].append(T15_data['ne_err'][i])
            T15['chi2'].append(T15_data['chi2'][i])
    p+=1'''
    '''if p > 1:
        p = 0
        plt.show()'''
'''print(p)
plt.figure()
plt.errorbar(TS['Te'], T15['Te'], yerr=T15['Te_err'], xerr=TS['Te_err'], fmt='.')
plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.plot([i for i in range(2000)], [i for i in range(2000)])
plt.xlabel('TS 10 poly, eV')
plt.ylabel('T15, eV')
plt.grid()

plt.figure()
plt.hist(T15['chi2'], range=(0, 25), bins=20                                                                                                                                                        )

plt.figure()
plt.hist(TS['chi2'], range=(0, 25), bins=20)
plt.show()'''

'''with open('220411_for_comparison.txt', 'w') as file2:
    file2.write(' %14s' %'TS_Te')
    file2.write(' %14s' % 'TS_Terr')
    file2.write(' %14s' % 'T15_Te')
    file2.write(' %14s' % 'T15_Terr')
    file2.write(' %14s' % 'TS_ne')
    file2.write(' %14s' % 'TS_nerr')
    file2.write(' %14s' % 'T15_ne')
    file2.write(' %14s' % 'T15_nerr')
    file2.write(' %14s' % 'TS_chi2')
    file2.write(' %14s' % 'T15_chi2')
    file2.write('\n')
    for i in range(len(TS['Te'])):
        file2.write(' %14.4f' % TS['Te'][i])
        file2.write(' %14.4f' % TS['Te_err'][i])
        file2.write(' %14.4f' % T15['Te'][i])
        file2.write(' %14.4f' % T15['Te_err'][i])
        file2.write(' %14.4e' % TS['ne'][i])
        file2.write(' %14.4e' % TS['ne_err'][i])
        file2.write(' %14.4e' % T15['ne'][i])
        file2.write(' %14.4e' % T15['ne_err'][i])
        file2.write(' %14.4f' % TS['chi2'][i])
        file2.write(' %14.4f' % T15['chi2'][i])
        file2.write('\n')'''


'''plt.figure()
for event in range(len(data['data']['events'])):
    plt.plot()'''
plt.show()