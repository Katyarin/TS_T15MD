import json
import numpy as np
import matplotlib.pyplot as plt

shot_N = 41617
poly = 34

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


print(N_phe.keys())
print(dev_num['ch'].keys())
for ch in dev_num['ch'].keys():
    plt.plot(timeline[16:32], N_phe[ch][16:32], 'o-', label=ch)
plt.legend()
plt.grid()


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
    index = chi.index(chi_min)
    if index == len(dev['Te']) - 1:
        index = len(dev['Te']) - 2
        print('index out of range')
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

Te = []
Te_err = []
ne = []
chi2 = []
start_temp = 18
end_temp = 65
for i in range(len(N_phe['0'])):
#for i in range(start_temp, end_temp):
    Te_loc, ne_loc, chi2_loc, Te_err_loc = TS_res(dev_num, N_phe, sigma, i)
    '''plt.figure()
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

print(Te_err)
plt.figure()
plt.plot(timeline, Te, 'ro-')
plt.grid()
plt.ylim(0, 2000)

plt.figure()
plt.errorbar(timeline, Te, yerr=Te_err)
plt.grid()
plt.ylim(0, 2000)

with open('Files/' + str(shot_N) + '/' + 'Te.json', 'w') as res_file:
    for_temp = {'timeline': timeline, 'Te': Te, 'Te_err': Te_err}
    json.dump(for_temp, res_file)


plt.show()





