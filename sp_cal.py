import matplotlib.pyplot as plt
import numpy as np
import json
pi = 3.14159265359
new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


PATH = 'c:/work/Thomson Scattering/T-15/Spectral_cal/'

Id = 11.95*1e-6
poly = 34

R = 800 #mm
S = 4.4*1e-3*11.7*1e-3 #m
Om = pi * (135/2)**2 / R**2
print(Om)
Rsv = 10000
G_slow = 40
G_fast = 10

def linear(I1, I2, I3, listy1, listy2):
    listy3 = []
    for i in range(len(listy1)):
        listy3.append(listy2[i] + (listy1[i]-listy2[i])*(I2 - I3)/ (I2-I1))
    return listy3


def linear_approx(dict, l_new):
    data_new = []
    l_test = []
    for l in l_new:
        for i, l_old in enumerate(dict['l']):
            if dict['l'][i] <= l < dict['l'][i+1]:
                l_test.append(l)
                x1 = dict['l'][i]
                x2 = dict['l'][i+1]
                y1 = dict['data'][i]
                y2 = dict['data'][i+1]
                data_new.append(y1 + (y2 - y1) * (l - x1) / (x2 - x1))
    return data_new


def lamp_spectr(Id):
    lamp_data = {}
    detectors_i = []
    p = 0
    with open(PATH + 'lamp_sp.txt', 'r') as file:
        for line in file:
            data = line.split()
            if p == 0:
                for el in data:
                    lamp_data[float(el)] = []
                    detectors_i.append(float(el))
                p+=1
            else:
                for i, el in enumerate(data):
                    lamp_data[detectors_i[i]].append(float(el))

    print(lamp_data.keys())
    sp_in = 0
    for i in detectors_i:
        if Id > i:
            sp_in+=1
    print(sp_in)

    P = {'l': [i * 1000 for i in lamp_data[detectors_i[0]]], 'data': []}
    if sp_in < 3:
        print(detectors_i[1], detectors_i[2], Id, lamp_data[detectors_i[1]], lamp_data[detectors_i[2]])
        P['data'] = linear(detectors_i[1], detectors_i[2], Id, lamp_data[detectors_i[1]], lamp_data[detectors_i[2]])
    elif sp_in > 3:
        P['data'] = linear(detectors_i[3], detectors_i[4], Id, lamp_data[detectors_i[3]], lamp_data[detectors_i[4]])
    else:
        P['data'] = linear(detectors_i[2], detectors_i[3], Id, lamp_data[detectors_i[2]], lamp_data[detectors_i[3]])
    return P


def filters(poly):
    K = {'l': []}
    for ch in range(1, 7):
        K[ch] = []
    if poly == 34:
        with open(PATH + 'optical_#34.txt') as file3:
            for line in file3:
                data = line.split()
                K['l'].append(float(data[0]))
                for ch in range(1, 7):
                    K[ch].append(float(data[ch]))
    else: error
    return K


P = lamp_spectr(Id)
K = filters(poly)

LFD = {'l': [], 'data': []}

with open(PATH + 'hamamatsu.txt', 'r') as file2:
    for line in file2:
        data = line.split()
        LFD['l'].append(float(data[0]))
        LFD['data'].append(float(data[1]))

P_l = linear_approx(P, K['l'])
R_l = linear_approx(LFD, K['l'])

for ch in range(1, 7):
    plt.plot(K['l'], [K[ch][i] * R_l[i] * P_l[i] for i in range(len(K['l']))])
#plt.show()
plt.figure()
p_p_all = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
for file_n in range(1, 11):
    measure = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    with open(PATH + '20220512cal_with_lens/' + str(file_n), 'r') as file:
        p = 0
        for line in file:
            data = line.split()
            for i, el in enumerate(data):
                measure[i+1].append(float(el))
            p+=1
            if p == 511 or p == 1023:
                plt.plot(measure[6])
                for ch in range(3, 9):
                    p_p_all[ch - 2].append(max(measure[ch]) - min(measure[ch]))
                p_p_all[7].append(sum(measure[1])/len(measure[1]))
                p_p_all[8].append(sum(measure[2]) / len(measure[2]))
                measure = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

p_p = {}
for ch in range(1, 9):
    p_p[ch] = sum(p_p_all[ch])/len(p_p_all[ch])
print(p_p)
plt.figure()
for ch in range(1, 7):
    plt.scatter([i for i in range(len(p_p_all[ch]))], p_p_all[ch])
    plt.plot([p_p[ch] for i in range(len(p_p_all[ch]))])

plt.figure()
for ch in range(7, 9):
    plt.scatter([i for i in range(len(p_p_all[ch]))], p_p_all[ch])
    plt.plot([p_p[ch] for i in range(len(p_p_all[ch]))])


I = {}
for ch in range(1, 7):
    I[ch]= np.trapz([K[ch][i] * R_l[i] * P_l[i] * 1000 * S * Om for i in range(len(K['l']))], [i * 1e-9 for i in K['l']])
print(I)

alpha = []

for ch in range(1, 7):
    alpha.append(p_p[ch] * 1e-3 / (Rsv * G_slow * 0.5 * I[ch]))

print(alpha)

plt.figure()
for ch in range(1, 6):
    plt.plot(K['l'], [1e-9 * K[ch][i] * R_l[i] * alpha[ch-1] * Rsv * G_slow * 0.5 for i in range(len(K['l']))])
    plt.plot(K['l'], [1e-9 * K[ch][i] * R_l[i] * alpha[ch - 1] * Rsv * G_fast * 0.5 for i in range(len(K['l']))], '--',
                 color=new_colors[ch - 1])
plt.ylabel('S, V/W')
plt.xlabel('wavelenght, nm')
plt.grid()
plt.ylim(0,5)
#plt.savefig('20220512_VW_char.png', dpi=300)


al_dict_ust = {}

K_new_ust = {'l': K['l']}

for ch in range(1,6):
    K_new_ust[ch] = [K[ch][i] * R_l[i] * alpha[ch - 1] for i in range(len(K['l']))]

for ch in range(5):
    al_dict_ust[ch+1] = alpha[ch]

with open(PATH + '22.04.13_sp_cal_coef_ust.json', 'w') as file4:
    json.dump(al_dict_ust, file4)

with open(PATH + '22.04.13_sp_cal_ust.json', 'w') as file5:
    json.dump(K_new_ust, file5)
plt.show()
