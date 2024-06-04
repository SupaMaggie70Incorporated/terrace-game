#! /usr/bin/env python3
# This should be run separately from the main program for side by side data
from pynvml import *
import psutil
import time
import matplotlib.pyplot as plt

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)

os.system("clear")
print()
max_len = 0
cpu_usages = []
ram_usages = []
gpu_usages = []
xs = []

time_calculated = 0
while True:
    info = nvml.nvmlDeviceGetMemoryInfo(h)
    cpu_percent = int(psutil.cpu_percent() * 100) / 100
    ram_percent = int(psutil.virtual_memory()[2] * 100) / 100
    gpu_percent = int(10000 * info.used / info.total) / 100
    cpu_usages.append(cpu_percent)
    ram_usages.append(ram_percent)
    gpu_usages.append(gpu_percent)
    xs.append(time_calculated)
    str = f"\rCPU: {cpu_percent}%, RAM: {ram_percent}%, GPU: {gpu_percent}%"
    if len(str) > max_len:
        max_len = len(str)
    else:
        str += " " * (max_len - len(str))
    print(str, end="")
    time.sleep(0.5)
    if time_calculated % 10 == 0:
        plt.cla()
        plt.clf()
        plt.plot(xs, cpu_usages, label = "CPU")
        plt.plot(xs, ram_usages, label = "RAM")
        plt.plot(xs, gpu_usages, label = "GPU")
        plt.legend()
        plt.savefig("usage_diagrams.svg")
    time_calculated += 1