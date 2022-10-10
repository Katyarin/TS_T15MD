from pathlib import Path
import json


def to_json(shotn, ch_count=6, path_name='c:/work/Data/T-15/New_prog/'):
    # ! use only Your own local copy of raw data files!
    path = Path(path_name + str(shotn))
    if not path.is_file():
        print('not found')

    data = [{
        't': 0,
        'ch': [[0 for cell in range(1024)] for ch in range(ch_count + 1)]
    }]
    with path.open(mode='r') as file:
        count = 0
        event = {
            't': 0,
            'ch': [[] for ch in range(ch_count + 1)]
        }
        for line in file:
            if count > 1023:
                count += 1
                if count == 1026:
                    data.append(event.copy())
                    event = {
                        't': 0,
                        'ch': [[] for ch in range(ch_count + 1)]
                    }
                    count = 0
                continue
            sp = line.split()

            for ch in range(ch_count + 1):
                event['ch'][ch].append(float(sp[1 + ch]))
            if count == 0:
                event['t'] = int(sp[-2])
            count += 1

    with open(str(path) + '.json', 'w') as file:
        json.dump(data, file)

    print('Code OK')

'''shots = [42000]
for shotn in shots:
    to_json(shotn)'''

to_json('100pages', 7, 'c:/work/Code/SpectralCalibrStend/drs_raw_data/')