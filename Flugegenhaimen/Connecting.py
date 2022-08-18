import json
import requests
import os
import numpy as np

#shot_N = '1' #no plasma
shot_N = 41551 #plasma
N_pages_set = 56 #number of pages to be recorded
polyn = 34

'''shot update'''
with open('Files/shot.txt', 'w') as sh_file:
    sh_file.write(str(shot_N))

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

'''create folder to shot recording'''
path = 'Files/' + str(shot_N)
try:
    os.mkdir(path)
except OSError:
    print('Не удалось создать папку')

with open('Files/' + str(shot_N) + '/' + 'options.json', 'w') as f:
    shot_options = {'polyn': polyn, 'N_pages_set': N_pages_set, 'config': start_options}
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

'''get Info from device'''
req = {"reqtype": "drsInfo", "subsystem": "drs"}
ret = doRequest(device, req)
print(ret)

'''Calibrate'''
def amplitudeCalibration(baseLine = start_options['amplitude_range']):
    print('___________________')
    calibra = doRequest(device, {'reqtype':'calibrate', 'baseLine':baseLine})
    #m_A = calibra['data']
    #m_B = calibra['B_coef']
    print('Calibration set in renge: ', start_options['amplitude_range'])
    return calibra

amplitudeCalibration()

'''set trigger options'''
req = {"reqtype": "triggerConfig", "subsystem": "drs","invertedFront": False,"type":start_options["trigger_ch"],
       "delay":0,"value": start_options["trigger_lvl"],}
ret = doRequest(device, req)
print('___________________')
print('set trigger options')
print(ret)


'''start waiting signals'''
req = {"reqtype":"awaitTrigger","burstLength":N_pages_set,"subsystem":"drs"}
ret = doRequest(device, req)
print(ret)
print('___________________')
if ret['status']=='success':
    print('Number of pages to be registered: ', N_pages_set)