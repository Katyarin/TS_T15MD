import json
import requests
import numpy as np

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

'''define device'''
device = {'ip': '192.168.10.21', 'port': '8080'}

'''get Info from device'''
req = {"reqtype": "drsInfo", "subsystem": "drs"}
ret = doRequest(device, req)
print(ret)

# Set trigger parameters

# {"reqtype":"triggerConfig","invertedFront":false,"type":"ch1","delay":0,"value":0.05,"subsystem":"drs"}:
# {"reqtype":"awaitTrigger","burstLength":2,"subsystem":"drs"}:
# {"reqtype":"drsInfo","subsystem":"drs"}:
# {"reqtype":"getPagesReady","subsystem":"drs"}:
#{"reqtype":"regionGetData","from":0,"pages":2,"subsystem":"drs"}:

req = {"reqtype": "triggerConfig", "subsystem": "drs","invertedFront": False,"type":"ch1","delay":0,"value":0.05,}
ret = doRequest(device, req)
print(ret)

'''Read pages'''
N_pages_set = 6 # number of pages to be recorded

req = {"reqtype":"awaitTrigger","burstLength":N_pages_set,"subsystem":"drs"}
ret = doRequest(device, req)
print(ret)
print('___________________')
if ret['status']=='success':
    print('Number of pages to be registered: ', N_pages_set)

'''Check number of pages '''
req = {"reqtype":"getPagesReady","subsystem":"drs"}
ret = doRequest(device, req)
print(ret)
N_pages_get = ret['pagesReady']
print('________________________________________')
print('Number of registered pages: ',N_pages_get)

'''Download data'''
shot_N = 39491
N_pages_total = 292

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
    times = np.round(times * 2E-8 * 1000)  # relative timestamps in ms
    print(times)

    if N_pages_total - N_pages_get > 0:
        with open(str(shot_N) + '_' + str(N_pages_get) + '_to_' + str(N_pages_get + 50) + '.json', 'w') as fp:
            json.dump(ret_data, fp)
    elif 50 > N_pages_total - N_pages_get > 0:
        with open(str(shot_N) + '_' + str(N_pages_get) + '_to_' + str(N_pages_total - N_pages_get) + '.json',
                  'w') as fp:
            json.dump(ret_data, fp)
