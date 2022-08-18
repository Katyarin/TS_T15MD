import json
import scipy.constants as cnst
import matplotlib.pyplot as plt
import math as mt
import numpy as np

thetta = 123.5
fiber = 'ust'

sp_cal_path = 'c:/work/Thomson Scattering/T-15/Spectral_cal/'

name_list = [1, 2, 3, 4, 5]
dev_34 = {'lmd': [], 'ch': {1: [], 2: [], 3: [], 4: [], 5: []}}

with open(sp_cal_path + '22.04.13_sp_cal_' + fiber + '.json', 'r') as sp_file:
    sp_cal = json.load(sp_file)

dev_34['lmd'] = sp_cal['l']
for ch in dev_34['ch'].keys():
    dev_34['ch'][ch] = sp_cal[str(ch)]

'''with open('dev_#34.txt', 'r') as file:
    for line in file:
        all = line.split()
        dev_34['lmd'].append(float(all[0]))
        for i in range(1, len(name_list) + 1):
            dev_34['ch'][name_list[i-1]].append(float(all[i]))'''

for ch in dev_34['ch'].keys():
    plt.plot(dev_34['lmd'], dev_34['ch'][ch])
plt.show()



def waiting_signal(ch, Te, thetta, Spectral_calibr):
    '''Функция вычисляет ожидаемый сигнал при температуре Te в канале ch в полихроматоре с углом зондирования thetta'''
    '''константы'''
    m = cnst.electron_mass
    c = cnst.c
    h = cnst.h
    pi = cnst.pi
    re = 2.8179403267 * 10 ** (-15)
    L = 0.018  # длина участка, с которого собирается свет, м
    Tt = thetta * pi / 180  # угол рассеяния, радианы
    fi = 90 * pi / 180  # уол между поляризацией и направлением рассеяния, радианы
    Om = 0.015  # телесный угол, стерадианы
    ne = 10 ** 19  # концентрация электронов, м-3
    T = 0.3
    e = 2.71828

    '''pr = [0.79, 0.78, 0.75, 0.72, 0.62, 0.64]  # пропускание каналов
    QE = [0.5, 0.54, 0.6, 0.7, 0.7, 0.46]  # квантовый выход'''

    E = 1  # энергия лазера, Дж
    lo = 1064  # длина волны лазера (может быть 1055), нм

    l = Spectral_calibr['lmd']

    '''сечение рассеяния'''
    al = (m * c ** 2) / (2 * Te * 1.602 * (10 ** (-19)))
    x_list = [(i - lo) / lo for i in l]
    C = (al / pi) ** .5 * (1 - 15 / (16 * al) + 345 / (512 * al ** 2))
    B = [((1 + ((x ** 2) / (2 * (1 - mt.cos(Tt)) * (1 + x)))) ** 0.5) - 1 for x in x_list]
    A = [(1 + x) ** 3 * (2 * (1 - mt.cos(Tt)) * (1 + x) + x ** 2) ** .5 for x in x_list]
    Sigma = [C*((e)**(-2*al*B[i]))/(A[i] * lo * 10 ** (-9)) * (mt.sin(fi))**2 for i in range(len(x_list))]

    dl = [i * 10 ** (-9) for i in l]
    F = np.trapz([Sigma[i] * Spectral_calibr['ch'][ch][i] / dl[i] for i in range(len(l))], dl)
    return F

N_th = {}

Te_ex = [1.1 ** (i / 10) for i in range(20, 830)]
for ch in name_list:
    N_th[ch] = []
    for temp in Te_ex:
        N_th[ch].append(waiting_signal(ch, temp, thetta, dev_34))

plt.figure()
for ch in name_list:
    plt.plot(Te_ex, N_th[ch], label=str(ch))
plt.grid()
plt.legend()
plt.show()

Result = {'Te': Te_ex, 'ch': N_th}

with open('22.06.30_dev_num_34.json', 'w') as f_fin:
    json.dump(Result, f_fin)

