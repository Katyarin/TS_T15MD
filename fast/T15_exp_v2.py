import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import requests
from pathlib import Path

shots = [42193]
G = 10

plot_osc = False
plot_exp_sig = True # False
time_for_start_plot = 180 #ms

print(shots)
polyn = 34
thomson = 'ust' #or 'usual' #or 'ust' or 'divertor'

sp_cal = '22.06.30'
abs_cal = '22.06.01'
laser_const = 625448756.1404866 #881940473.5373197
#laser_const = 485448756.1404866 #881940473.5373197

bound1_prof = 160
bound2_prof = 180
ne_corr_mult_all = 1
ne_corr_mult_T15 = 3.85e22 #8.33e40

URL = 'https://172.16.12.130:443/api'
TSpath = 'TS_core/'
raw_data_path = 'C:/Users/user/Desktop/t15/data/raw_data/'
res_data_path = 'C:/Users/user/Desktop/t15/data/results/'

with open('source/%s_abs_cal.json' %abs_cal, 'r') as abs_file:
    abs_data = json.load(abs_file)

A = abs_data['A']


#const
sigma_ts = 8/3 * math.pi * 2.8179403267 * 1e-15 * 2.8179403267 * 1e-15
lamd0 = 1064 *1e-9
q = 1.6e-19 #cl

const_for_ne = A * lamd0 * sigma_ts / q
print(const_for_ne)

def to_json(shotn, ch_count=6):
    # ! use only Your own local copy of raw data files!
    path0 = Path(res_data_path + '%d/%d.json' % (shotn,shotn))
    if path0.is_file():
        print('ok')
        return 0
    path2 = res_data_path + str(shotn)
    try:
        os.mkdir(path2)
    except OSError:
        print('Не удалось создать папку')

    path = Path(raw_data_path + '%d' % shotn)
    if not path.is_file():
        print('not found')
        stop



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

    with open(str(path2) + '/%d.json' %(shotn), 'w') as file:
        json.dump(data, file)

    print('Code OK')

def TS_res(dev, N_phe, sigma, i, err):
    ne = []
    chi = []
    for j in range(len(dev['ch']['1'])):
        num = 0
        den = 0
        ch_count = 0
        for ch in dev['ch'].keys():
            if err[ch][i] == 'off scale':
                continue
            num += N_phe[ch][i] * dev['ch'][ch][j] / math.pow(sigma[ch][i], 2)
            den += dev['ch'][ch][j] * dev['ch'][ch][j] / math.pow(sigma[ch][i], 2)
            ch_count+=1
        if ch_count < 2:
            return 1, 1, 1000, 100, 100
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
    wtf: float = 1e-40
    sigma_ne: float = 1e40
    if (sum1 * sum2 - sum3) != 0 and (N_phe['6'][i] * const_for_ne) != 0:
        sigma_ne = ne_res / (N_phe['6'][i] * const_for_ne) * (sum2 / (sum1 * sum2 - sum3)) ** 0.5
        wtf = ne_res / (N_phe['6'][i] * const_for_ne)
    #(i, ne_res, N_phe['6'][i])

    return Te, wtf, chi_min, sigma_Te, sigma_ne


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

    filename = res_data_path + '%d/%d.json' %(shotn, shotn)

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
                #print(event['t']/1000 + 3.73)
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
            if ch==6:
                Ni = Ni * (M * el_charge * G * R_sv * 0.5) * laser_const
            N_photo_el[ch].append(Ni *1e-12)
            var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr - 1.14e4 * 0.0625) + math.fabs(Ni *1e-12) * 4)
            var_phe[ch].append(var)
        plt.show()
        p=0
    plt.figure(figsize=(10, 3))
    plt.title('Shot #' + str(shotn))
    for ch in N_photo_el.keys():
        #color = ['r', 'g', 'b', 'm', 'black', 'orange', 'brown', 'pink']
        if ch != 0 and ch != 6:
            plt.errorbar(timestamps, N_photo_el[ch], yerr=var_phe[ch], label='ch' + str(ch))
            plt.scatter([t for i, t in enumerate(timestamps) if calc_err[ch][i] == 'off scale'],
                        [j for i, j in enumerate(N_photo_el[ch]) if calc_err[ch][i] == 'off scale'], marker='x', s=40, c='black', zorder=2.5)
        #plt.plot(timestamps, N_photo_el[ch], '^-', label='ch' + str(ch))
    N_photo_el[6][0] = N_photo_el[6][1]
    plt.ylabel('N, phe')
    plt.grid()
    plt.xlabel('time')
    plt.legend()
    plt.figure(figsize=(10, 3))
    plt.title('Shot #' + str(shotn))
    #plt.errorbar(timestamps, N_photo_el[6], yerr=var_phe[6], label='ch' + str(6))
    plt.plot(timestamps, N_photo_el[6], label='ch' + str(6))
    #print(1.25 / (sum(N_photo_el[6][2:])/len(N_photo_el[6][2:])))
    plt.ylabel('El, J')
    plt.grid()
    plt.xlabel('time')
    plt.legend()
    #plt.show()

    with open(res_data_path + '%d/N_phe.json' %shotn, 'w') as f:
        for_temp = {'timeline': timestamps, 'data': N_photo_el, 'err': var_phe, 'culc_err': calc_err, 'laser_en': N_photo_el[6]}
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
    with open(res_data_path + '%d/N_phe.json' %shotn, 'r') as fp:
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


    with open(res_data_path + '%d/Te.json' %shotn, 'w') as res_file:
        for_temp = {'timeline': timeline, 'Te': Te, 'Te_err': Te_err, 'ne': ne, 'ne_err': ne_err, 'chi2': chi2 }
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

    to_phe(shotn)

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
    plt.errorbar([t for i, t in enumerate(T15_data['timeline']) if T15_data['ne_err'][i] / T15_data['ne'][i] < 0.7],
                 [t for i, t in enumerate(T15_data['ne']) if T15_data['ne_err'][i] / T15_data['ne'][i] < 0.7],
                 yerr=[t for i, t in enumerate(T15_data['ne_err']) if T15_data['ne_err'][i] / T15_data['ne'][i] < 0.7], ls='--')
    #plt.ylim(0, 2000)
    plt.grid()
    plt.xlabel('time, ms')
    plt.ylabel('ne')
    #plt.show()

    with open(res_data_path + '%d/result_dynamic.txt' %shotn, 'w') as f_res1:
        f_res1.write(' %14s' % 'time_ms')
        f_res1.write(' %14s' % 'Te')
        f_res1.write(' %14s' % 'Te_err')
        f_res1.write(' %14s' % 'ne')
        f_res1.write(' %14s' % 'ne_err')
        f_res1.write(' %14s' % 'chi2')
        f_res1.write('\n')
        for i in range(len(T15_data['timeline'])):
            f_res1.write(' %14.4f' % T15_data['timeline'][i])
            f_res1.write(' %14.4f' % T15_data['Te'][i])
            f_res1.write(' %14.4f' % T15_data['Te_err'][i])
            f_res1.write(' %14.4f' % (T15_data['ne'][i]*ne_corr_mult_T15))
            f_res1.write(' %14.4f' % (T15_data['ne_err'][i]*ne_corr_mult_T15))
            f_res1.write(' %14.4f' % T15_data['chi2'][i])
            f_res1.write('\n')

    try:
        response = requests.post(url=URL, verify=False, json={
            'subsystem': 'db',
            'reqtype': 'get_shot',
            'shotn': int(shotn)
        })
        data = response.json()
        if data['ok'] == False:
            print(data['description'])
            plt.show()
        #with open('dump.json', 'w') as file:
        #json.dump(data, file)
    except:
        print('Not TS?')
        plt.show()

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
        #print(poly_list[11])

        fibers = ['1A', '1B'] + [str(i) for i in list(range(2, 10))] + ['VA_LFS'] + ['10']

        fig, axs = plt.subplots(2,2)
        fig.suptitle('shotn #' + str(shotn))
        for poly in range(12):
            '''if poly < 9:
                continue'''
            #plt.plot([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], [i['T_e'][poly]['T'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], label=poly)
            axs[0, 0].errorbar([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None],
                     [i['T_e'][poly]['T'] for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error'] == None], yerr=[i['T_e'][poly]['Terr'] for i in data['data']['events'] if
                      i['error'] == None and i['T_e'][poly]['error']== None], label=str(data['data']['config']['fibers'][fibers[poly]]['R']) + 'mm')
        axs[0, 0].set_ylim(0,2000)
        #axs[0, 0].legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc='lower left', ncol=4)
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
                      i['error'] == None and i['T_e'][poly]['error']== None], label=str(data['data']['config']['fibers'][fibers[poly]]['R']) + ' mm')
        #plt.ylim(0,2000)
        axs[0, 1].legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize='small')
        axs[0, 1].grid()
        axs[0, 1].set_xlabel('time, ms')
        #axs[0, 1].set_color_cycle(sns.color_palette("coolwarm_r"))



        #plt.figure()
        for event in data['data']['events']:
            if event['error'] == None and bound1_prof < event['timestamp'] < bound2_prof:
                axs[1, 0].errorbar([data['data']['config']['fibers'][i]['R'] for poly, i in enumerate(fibers) if event['T_e'][poly]['error'] == None],
                         [event['T_e'][poly]['T']for poly in range(12) if event['T_e'][poly]['error'] == None],
                             yerr=[event['T_e'][poly]['Terr']for poly in range(12) if event['T_e'][poly]['error'] == None])

        axs[1, 0].grid()
        axs[1, 0].set_xlabel('R, mm')
        axs[1, 0].set_ylabel('Te, eV')

        #plt.figure()
        for event in data['data']['events']:
            if event['error'] == None and bound1_prof < event['timestamp'] < bound2_prof:
                axs[1, 1].errorbar([data['data']['config']['fibers'][i]['R'] for poly, i in enumerate(fibers) if
                              event['T_e'][poly]['error'] == None],
                             [event['T_e'][poly]['n']*ne_corr_mult_all for poly in range(12) if event['T_e'][poly]['error'] == None],
                             yerr=[event['T_e'][poly]['n_err']*ne_corr_mult_all for poly in range(12) if event['T_e'][poly]['error'] == None],
                                   label=str(round(event['timestamp'], 1)) + ' ms')

        axs[1, 1].grid()
        axs[1, 1].legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize='small')
        axs[1, 1].set_xlabel('R, mm')
        axs[1, 1].set_ylabel('ne, m^-3')



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
