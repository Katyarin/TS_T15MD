import json
import requests
import os
import numpy as np

shot_N = 39787


'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + str(shot_N) + '/' + 'options.json', 'r') as f:
    shot_options = json.load(f)

N_pages_set = shot_options["N_pages_set"]

'''create folder to shot recording''' #not working now
path = 'Files/' + str(shot_N) + '/raw'
try:
    os.mkdir(path)
except OSError:
    print('Не удалось создать папку')

'''define function for RPC request'''
def doRequest(device, req):
#    req['subsystem'] = device['dev']
    print('http://' + device['ip'] + ':' + device['port'] +
          '/flugegeheimen')
    print(json.dumps(req))
    r = requests.post('http://' + device['ip'] + ':' + device['port'] +
                      '/flugegeheimen', data=json.dumps(req))
    #print(r.json())
    return r.json()

def writing_data(N_pages_get, shot_N):
    for N_pages in range(0, N_pages_get, 50):
        print(N_pages_get)
        if 0 <= N_pages_get - N_pages <= 50:
            req = {"reqtype": "regionGetData", "from": N_pages, "pages": N_pages_get - N_pages,
                   "subsystem": "drs"}
        elif N_pages_get - N_pages > 50:
            req = {"reqtype": "regionGetData", "from": N_pages, "pages": 50, "subsystem": "drs"}
        ret_data = doRequest(device, req)
        print('________________')
        print(ret_data.keys())
        print('Number of downloaded pages:', ret_data['readPagesCount'])
        # ret_data['data']
        # ret_data['stops']
        # ret_data['stopsSelection']
        # ret_data['timestamps']
        times = np.array(ret_data['timestampsSelection'])
        times = times - times[0]
        print('________________')
        print('relative timestamps in ms')
        times = times * 2E-8 * 1000  # relative timestamps in ms
        times = times + (9.85 - times[1])
        print(times)

        if N_pages_get - N_pages >= 50:
            with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(N_pages) + '_to_' + str(N_pages + 50) + '.json', 'w') as fp:
                json.dump(ret_data, fp)
        elif 50 > N_pages_get - N_pages > 0:
            with open('Files/' + str(shot_N) + '/' + 'raw' + '/' + str(shot_N) + '_' + str(N_pages) + '_to_' + str(N_pages_get - N_pages) + '.json',
                      'w') as fp:
                json.dump(ret_data, fp)


'''define device'''
device = {'ip': '192.168.10.21', 'port': '8080'}

'''Check number of pages '''
req = {"reqtype":"getPagesReady","subsystem":"drs"}
ret = doRequest(device, req)
print(ret)
N_pages_get = ret['pagesReady']
print('________________________________________')
print('Number of registered pages: ',N_pages_get)

if N_pages_get == N_pages_set:
    writing_data(N_pages_get, shot_N)
    shot_options['N_pages_get'] = N_pages_get
    with open('Files/' + str(shot_N) + '/options.json', 'w') as f:
        json.dump(shot_options, f)

else:
    print('Записанное количество данных не равно запрашиваемому. Остановить запись?')
    answer = int(input())
    if answer == 1:
        writing_data(N_pages_get, shot_N)
        shot_options['N_pages_get'] = N_pages_get
        with open('Files/' + str(shot_N) + '/options.json', 'w') as f:
            json.dump(shot_options, f)
        req = {"reqtype": "stopTriggerWait", "subsystem": "drs"}
        ret = doRequest(device, req)
        print(ret)
        print('___________________')
        if ret['status'] == 'success':
            print('Ожидание триггера остановлено')

    else:
        print('Система ожидает %d импульсов' %(N_pages_set - N_pages_get))


