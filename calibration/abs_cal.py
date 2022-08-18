import numpy as np
import math
import matplotlib.pyplot as plt
import json
from scipy import interpolate as inter

lamd0 = 1064e-9 #m
E_las = 1.6 #J

h = 6.626e-34
c = 299792458
Bo = 1.99e2 #m-1
k = 1.38e-23
q = 1.6e-19 #cl
gamma2 = 0.51e-48 * 1e-12 #m

sp_cal_path = 'c:/work/Thomson Scattering/T-15/Spectral_cal/'


def gt(J):
    if J%2:
        return 6
    else:
        return 3


def Q(J_max, Tg):
    sum_F = 0
    for j in range(J_max):
        sum_F += gt(j) * (2*j + 1) * math.exp(-j*(j+1) *h * c * Bo / (k * Tg))
    return sum_F


def lamd(J):
    return 1 / (1/lamd0 + Bo * (4*J - 2))


def F(J, J_max, Tg):
    return (gt(J) * (2*J + 1) * math.exp(-J*(J+1) *h * c * Bo / (k * Tg))) / Q(J_max, Tg)


def sigma_gas(J):
    return 64*(math.pi**4) / 45 * (3*J *(J-1)) / (2*(2*J +1) * (2*J-1)) *gamma2 / lamd(J)

def absolute_cal(PATH, gas_signal_file, G, P, Tg, ch=1, data_sp_cal='22.04.13', thomson = 'ust'):

    #const
    J_max = 40
    Tg = Tg + 273.15

    with open(sp_cal_path + data_sp_cal + '_sp_cal_' + thomson + '.json', 'r') as sp_file:
        sp_cal = json.load(sp_file)

    dev_34 = {'lmd': [], 'ch': {1: [], 2: [], 3: [], 4: [], 5: []}}
    dev_34['lmd'] = [i * 1e-9 for i in sp_cal['l']] #m
    for ch2 in dev_34['ch'].keys():
        dev_34['ch'][ch2] = sp_cal[str(ch2)]

    K_lam = inter.interp1d(dev_34['lmd'], dev_34['ch'][ch])
    plt.figure()
    plt.plot(dev_34['lmd'], dev_34['ch'][ch])
    for j in range(J_max):
        #print(j, lamd(j))
        plt.scatter(lamd(j), K_lam(lamd(j)))
    #plt.show()

    p=0
    signals = {'t': [], 'data': {0: [], 1: []}, 'laser': {'t': [], 'data': []}}
    with open(PATH + gas_signal_file, 'r') as file1:
        for line in file1:
            if p <= 1:
                p+=1
                continue
            signals['t'].append(float(line.split(sep=',')[0]))
            signals['data'][1].append(float(line.split(sep=',')[1]))
    p=0
    if 'scope' in gas_signal_file:
        with open(PATH + 'scope_0_4.csv', 'r') as file2:
            for line in file2:
                if p <= 1:
                    p += 1
                    continue
                signals['data'][0].append(float(line.split(sep=',')[1]))
        p = 0
        with open(PATH + 'scope_8_3.csv', 'r') as file3:
            for line in file3:
                if p <= 1:
                    p += 1
                    continue
                signals['laser']['t'].append(float(line.split(sep=',')[0]))
                signals['laser']['data'].append(float(line.split(sep=',')[1]))
    '''plt.figure()
    plt.plot(signals['laser']['data'])
    plt.show()'''
    #to phe

    M = 100
    el_charge = 1.6 * 10 ** (-19)
    # G = 10
    R_sv = 10000
    freq = 5  # GS/s
    time_step = 1 / freq  # nanoseconds
    event_len = 1024

    delta = {0: 0, 1: 150, 2: 170, 3: 190, 4: 200, 5: 210, 6: 180}
    timestamps = []
    N_photo_el = []
    var_phe = []
    timeline = [i * time_step for i in range(event_len)]

    pre_sig = 100

    signal = signals['data'][1]
    base_line = sum(signal[0:pre_sig]) / len(signal[0:pre_sig])

    for i in range(len(signal)):
        signal[i] = signal[i] - base_line

    index_0 = 0
    for i, s in enumerate(signals['data'][0]):
        if s > 0.250:
            index_0 = i - 20
            break

    for i in range(len(signal)):
        signal[i] = signal[i] * 1000
    var_in_sr = np.var(signal[0:pre_sig])
    if thomson == 'usual':
        width = 100
    elif thomson == 'ust':
        width = 120
        delta[ch] = delta[ch] - 100
    else:
        print('something wrong! Unnown config')
        stop
    start_index = index_0 + delta[ch]
    end_index = start_index + width

    plt.figure()
    plt.plot(signal)
    plt.vlines(start_index, min(signal), max(signal))
    plt.vlines(end_index, min(signal), max(signal))


    Ni = np.trapz(signal[start_index:end_index],
                  timeline[start_index:end_index]) / (M * el_charge * G * R_sv * 0.5)
    N_photo_el.append(Ni * 1e-12)
    var = math.sqrt(math.fabs(6715 * 0.0625 * var_in_sr - 1.14e4 * 0.0625) + math.fabs(Ni * 1e-12) * 4)
    var_phe.append(var)

    laser_sig = [i * 1000 for i in signals['laser']['data']]
    base_las = sum(laser_sig[0:pre_sig]) / len(laser_sig[0:pre_sig])
    print(base_las)
    laser_sig = [i - base_las for i in laser_sig]


    Li = np.trapz(laser_sig[index_0 + delta[6]:index_0 + delta[6]+ width],
                  timeline[index_0 + delta[6]:index_0 + delta[6]+ width]) * 1e-12

    plt.figure()
    plt.plot(laser_sig)
    plt.vlines(index_0 + delta[6], min(laser_sig), max(laser_sig))
    plt.vlines(index_0 + delta[6]+ width, min(laser_sig), max(laser_sig))
    plt.show()

    print(N_photo_el, var_phe)

    #gas data

    n_gas = P / (k*Tg)

    f_J = 0

    for j in range(2, J_max):
        print('J=', j, ' sigma*F=', sigma_gas(j) * F(j, J_max, Tg), ' K=', K_lam(lamd(j)), 'lambda=', lamd(j))
        #print(sigma_gas(j), F(j, J_max, Tg), K_lam(lamd(j)), lamd(j))
        f_J += (sigma_gas(j) * F(j, J_max, Tg) * K_lam(lamd(j)) / lamd(j))
        #f_J += (sigma_gas(j) * F(j, J_max, Tg) * K_lam(lamd(j)))

    A = N_photo_el / (E_las * lamd0 *n_gas / q * f_J)
    laser_const = E_las / Li
    print(A, laser_const, f_J)
    print(N_photo_el, E_las, lamd0, n_gas)

    return A, laser_const




path = 'c:/work/Thomson Scattering/T-15/Abs_cal/220.06.01/'
A1, las_const1 =absolute_cal(path, 'scope_0_1.csv', 10, 90.8, 21.5)
A2, las_const2 = absolute_cal(path, 'scope_0_2.csv', 10, 90.8, 21.5, 2)

print(A1, A2)
print(las_const1, las_const2)

with open('source/22.06.01_abs_cal.json', 'w') as res_file:
    json.dump({'data': '22.08.16', 'poly': 34 ,'A': A1[0], 'E_las_const': las_const1}, res_file)
print(A1, A2)
print(las_const1, las_const2)
plt.show()
