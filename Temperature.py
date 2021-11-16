import json
import numpy as np
import matplotlib.pyplot as plt

shot_N = 40827

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
    shot_options = json.load(f)

'''waiting signals'''
with open('dev_num_34.json', 'r') as file:
    dev_num = json.load(file)

'''data in phe'''
with open('Files/' + '/' + str(shot_N) + '/' + 'N_phe.json') as fp:
    data = json.load(fp)

N_phe = data['data']
timeline = data['timeline']
sigma = data['err']


print(N_phe.keys())
print(dev_num['ch'].keys())
for ch in N_phe.keys():
    plt.plot(timeline, N_phe[ch])

plt.figure()
plt.plot(timeline, N_phe['0'])
plt.show()

'''Temperature'''
#здесь будет расчет температуры.


def TS_res(dev, N_phe, sigma, i):
    ne = []
    chi = []
    for j in range(len(dev['ch']['1'])):
        num = 0
        den = 0
        for ch in dev['ch'].keys():
            if N_phe[ch][i] == None:
                continue
            num += N_phe[ch][i] * dev['ch'][ch][j] / sigma[ch][i]
            den += dev['ch'][ch][j] * dev['ch'][ch][j] / sigma[ch][i]
        ne_loc = num / den
        ne.append(num / den)
        chi_local = 0
        for ch in dev['ch'].keys():
            if N_phe[ch][i] == None:
                continue
            chi_local += (N_phe[ch][i] - ne_loc * dev['ch'][ch][j]) ** 2 / sigma[ch][i]
        chi.append(chi_local)
    chi_min = min(chi)
    Te = dev['t'][chi.index(chi_min)]
    ne_res = ne[chi.index(chi_min)]
    return Te, ne_res, chi_min


Te = []
ne = []
chi2 = []
for i in range(len(N_phe['0'])):
    Te_loc, ne_loc, chi2_loc = TS_res(dev_num, N_phe, sigma, i)
    '''plt.figure()
    for ch in dev_num['ch'].keys():
        plt.plot(dev_num['t'], [i * ne_loc for i in dev_num['ch'][ch]])
        plt.scatter(Te_loc, N_phe[ch][i])
    plt.show()'''
    Te.append(Te_loc)
    ne.append(ne_loc)
    chi2.append(chi2_loc)

plt.figure()
plt.plot(timeline, Te)
plt.show()





