import json
import matplotlib.pyplot as plt

with open('dev_num_35.json', 'r') as file:
    dev_35 = json.load(file)
for ch in dev_35['ch'].keys():
    plt.plot(dev_35['t'], dev_35['ch'][ch])
plt.show()
