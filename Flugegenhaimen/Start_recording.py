import json
import requests
import os
import numpy as np

shot_N = '1'
N_pages_set = 40 # number of pages to be recorded


'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + shot_N + '/' + 'options.json', 'w') as f:
    shot_options = {'N_pages_set': N_pages_set, 'config': start_options}
    json.dump(shot_options, f)

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

req = {"reqtype":"awaitTrigger","burstLength":N_pages_set,"subsystem":"drs"}
ret = doRequest(device, req)
print(ret)
print('___________________')
if ret['status']=='success':
    print('Number of pages to be registered: ', N_pages_set)
