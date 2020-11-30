import json
import numpy as np
import matplotlib.pyplot as plt

shot_N = 39542

'''reading_options'''
with open('config.json', 'r') as file:
    start_options = json.load(file)

with open('Files/' + start_options["data"] + '/' + str(shot_N) + '.json', 'r') as f:
    shot_options = json.load(f)

with open('Files/' + start_options["data"] + '/' + str(shot_N) + 'N_phe.json') as fp:
    data = json.load(fp)

N_phe = data['data']
timeline = data['timeline']

'''Temperature'''
#здесь будет расчет температуры.

#для него мне не хватает спектральных характеристик 34 прибора

#хотя можно сделать расчет по отношению сигналов в каналах:


