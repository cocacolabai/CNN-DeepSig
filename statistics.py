import os
import csv
def do_statistics():
    DATA_FILEPATH = './better/'
    print(os.listdir(DATA_FILEPATH))
    files = os.listdir(DATA_FILEPATH)
    files = [f for f in files if 'detail' not in f]
    data = []
    for f in files:
        print(os.path.join(DATA_FILEPATH, f))
        path = os.path.join(DATA_FILEPATH, f)
        with open(path, 'r') as g:
            while True:
                line = g.readline()
                if line == '':
                    break
                data.append(float(line.strip()))
    print(data)
    print('total: ', len(data))
    print('avg: ', sum(data)/len(data))
    print('min: ', min(data))
    print('max: ', max(data))

    stat = {str(i/100): 0 for i in range(85, 95, 1)}
    print(stat)
    for sample in data:
        for mid in stat.keys():
            if float(mid) - 0.005 < sample <= float(mid) + 0.005:
                stat[mid] += 1
    print(stat)
    print(sum([stat[key] for key in stat.keys()]))
    with open('best_result.csv', 'w') as f:
        fieldnames = ['MCC', 'AMOUNT']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in stat.items():
            writer.writerow({'MCC': float(key), 'AMOUNT': value})



do_statistics()
