import requests
import json
import matplotlib.pyplot as plt


shotn = 41615

URL = 'https://172.16.12.130:443/api'


response = requests.post(url=URL, verify=False, json={
    'subsystem': 'db',
    'reqtype': 'get_shot',
    'shotn': int(shotn)
})

try:
    data = response.json()
    '''with open('dump.json', 'w') as file:
        json.dump(data, file)'''
except:
    print('Not a json?')

with open('Files/' + str(shotn) + '/Te.json', 'r') as T15_file:
    T15_data = json.load(T15_file)

delta = [i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][8]['error'] == None][0]

Te_plot = []
time_plot = []
err_plot = []

for j in range(len(T15_data['Te'])):
    if T15_data['Te_err'][j] / T15_data['Te'][j] < 0.5:
        time_plot.append([i + delta for i in list(reversed(T15_data['timeline']))][j])
        Te_plot.append(T15_data['Te'][j])
        err_plot.append(T15_data['Te_err'][j])

for poly in range(10):
    #plt.plot([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], [i['T_e'][poly]['T'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None], label=poly)
    plt.errorbar([i['timestamp'] for i in data['data']['events'] if i['error'] == None and i['T_e'][poly]['error'] == None],
             [i['T_e'][poly]['T'] for i in data['data']['events'] if
              i['error'] == None and i['T_e'][poly]['error'] == None], yerr=[i['T_e'][poly]['Terr'] for i in data['data']['events'] if
              i['error'] == None and i['T_e'][poly]['error']== None], label=poly)
#plt.errorbar([i + delta for i in list(reversed(T15_data['timeline']))], T15_data['Te'], T15_data['Te_err'], label='T15 34')
plt.errorbar(time_plot, Te_plot, yerr=err_plot, label='T15 34')
plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.show()
