import subprocess
import time
from collections import deque

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        lines = output.decode('utf-8').strip().split('\n')
        return [line.split(', ') for line in lines]
    except:
        return []

def create_bar(percentage, width=20):
    filled = int(percentage / 100 * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"

max_points = 60
gpu_usages = {}
memory_usages = {}

try:
    while True:
        gpu_info = get_gpu_info()
        
        print("\033[2J\033[H")  # 清屏并将光标移到左上角
        
        for gpu in gpu_info:
            index, gpu_usage, memory_used, memory_total = gpu
            index = int(index)
            gpu_usage = float(gpu_usage)
            memory_used = float(memory_used)
            memory_total = float(memory_total)
            memory_usage = (memory_used / memory_total) * 100

            if index not in gpu_usages:
                gpu_usages[index] = deque(maxlen=max_points)
                memory_usages[index] = deque(maxlen=max_points)

            gpu_usages[index].append(gpu_usage)
            memory_usages[index].append(memory_usage)

            print(f"GPU {index}:")
            print(f"  GPU Usage:    {create_bar(gpu_usage)} {gpu_usage:.1f}%")
            print(f"  Memory Usage: {create_bar(memory_usage)} {memory_usage:.1f}%")
            print("  History (last 60 seconds):")
            print("  GPU Usage:    " + "".join(['▁▂▃▄▅▆▇█'[min(int(u/12.5), 7)] for u in gpu_usages[index]]))
            print("  Memory Usage: " + "".join(['▁▂▃▄▅▆▇█'[min(int(u/12.5), 7)] for u in memory_usages[index]]))
            print()

        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")