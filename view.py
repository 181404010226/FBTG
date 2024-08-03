import subprocess
import time
from collections import deque

def get_gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
        lines = output.decode('utf-8').strip().split('\n')
        return [line.split(', ') for line in lines]
    except:
        return [["N/A", "N/A", "N/A"]]

def create_bar(percentage, width=20):
    filled = int(percentage / 100 * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"

max_points = 60
gpu_usages = deque(maxlen=max_points)
memory_usages = deque(maxlen=max_points)

try:
    while True:
        gpu_info = get_gpu_info()[0]  # 假设只有一个GPU
        gpu_usage = float(gpu_info[0]) if gpu_info[0] != "N/A" else 0
        memory_used = float(gpu_info[1]) if gpu_info[1] != "N/A" else 0
        memory_total = float(gpu_info[2]) if gpu_info[2] != "N/A" else 1
        memory_usage = (memory_used / memory_total) * 100

        gpu_usages.append(gpu_usage)
        memory_usages.append(memory_usage)

        print("\033[2J\033[H")  # 清屏并将光标移到左上角
        print(f"GPU Usage:    {create_bar(gpu_usage)} {gpu_usage:.1f}%")
        print(f"Memory Usage: {create_bar(memory_usage)} {memory_usage:.1f}%")
        print("\nHistory (last 60 seconds):")
        print("GPU Usage:    " + "".join(['▁▂▃▄▅▆▇█'[min(int(u/12.5), 7)] for u in gpu_usages]))
        print("Memory Usage: " + "".join(['▁▂▃▄▅▆▇█'[min(int(u/12.5), 7)] for u in memory_usages]))
        
        print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")