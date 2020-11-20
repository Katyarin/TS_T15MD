import json
import requests
import os
import numpy as np

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

'''create folder to today recording''' #not working now
path = 'Files/' + start_options['data']
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
    m_A = calibra['data']
    m_B = calibra['B_coef']
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
