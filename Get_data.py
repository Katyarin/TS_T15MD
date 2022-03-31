import json
import requests
import numpy as np

'''Download data'''
shot_N = '39547'
N_pages_total = 59

def doRequest(device, req):
#    req['subsystem'] = device['dev']
    print('http://' + device['ip'] + ':' + device['port'] +
          '/flugegeheimen')
    print(json.dumps(req))
    r = requests.post('http://' + device['ip'] + ':' + device['port'] +
                      '/flugegeheimen', data=json.dumps(req))
    #print(r.json())
    return r.json()

'''define device'''
device = {'ip': '192.168.10.34', 'port': '8080'}

for N_pages_get in range(0, N_pages_total + 50, 50):
    if 0 < N_pages_total - N_pages_get < 50:
        req = {"reqtype": "regionGetData", "from": N_pages_get, "pages": N_pages_total - N_pages_get,
               "subsystem": "drs"}
    elif N_pages_total - N_pages_get > 0:
        req = {"reqtype": "regionGetData", "from": N_pages_get, "pages": 50, "subsystem": "drs"}
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

    if N_pages_total - N_pages_get > 0:
        with open(str(shot_N) + '_' + str(N_pages_get) + '_to_' + str(N_pages_get + 50) + '.json', 'w') as fp:
            json.dump(ret_data, fp)
    elif 50 > N_pages_total - N_pages_get > 0:
        with open(str(shot_N) + '_' + str(N_pages_get) + '_to_' + str(N_pages_total - N_pages_get) + '.json',
                  'w') as fp:
            json.dump(ret_data, fp)
