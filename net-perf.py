#! /bin/python3
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

#os.system("cargo run --release -- net-perf --duration-per 2 --thread-count 10")
properties = []
net_configs = []
net_values = {

}

with open("net-perf.csv") as csvfile:
    spamreader = csv.reader(csvfile)
    iterator = iter(spamreader)
    properties = iterator.__next__()[1:]
    for p in properties:
        net_values[p] = []
    for row in iterator:
        net_configs.append(row[0])
        print("Row: " + ' | '.join(row))
        for i, p in enumerate(properties):
            net_values[p].append(float(row[i + 1]))
x = np.arange(len(net_configs))
width = 0.25
multiplier = 0

fig, ax = plt.subplots()
dpi = fig.get_dpi()
fig.set_size_inches(4000 / dpi, 2000 / dpi)
for attribute, value in net_values.items():
    #print(f"{attribute}: {' | '.join(value)}")
    offset = width * multiplier
    rects = ax.bar(x + offset, value, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1
ax.set_ylabel("Evals per second")
ax.set_title("net-perf results")
ax.set_xticks(x + width, net_configs, rotation=90)
ax.legend(loc="upper left", ncols=3)
plt.savefig("net-perf.svg")
plt.show()