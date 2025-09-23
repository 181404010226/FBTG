import torch
import psutil
import time
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class GPUMemoryMonitor:
    def __init__(self, log_file="gpu_memory_log.txt", plot_file="gpu_memory_plot.png"):
        self.log_file = log_file
        self.plot_file = plot_file
        self.memory_data = []
        self.time_data = []
        self.monitoring = False
        self.start_time = None
        
    def get_gpu_memory_info(self):
        """获取GPU显存信息"""
        if not torch.cuda.is_available():
            return None
            
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            # 获取显存信息（单位：MB）
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024 / 1024
            max_reserved = torch.cuda.max_memory_reserved(i) / 1024 / 1024
            
            # 获取GPU总显存
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            
            gpu_memory[f'GPU_{i}'] = {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'max_reserved': max_reserved,
                'total': total_memory,
                'free': total_memory - reserved,
                'usage_percent': (reserved / total_memory) * 100
            }
            
        return gpu_memory
    
    def get_system_memory_info(self):
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024 / 1024,  # MB
            'available': memory.available / 1024 / 1024,  # MB
            'used': memory.used / 1024 / 1024,  # MB
            'percent': memory.percent
        }
    
    def log_memory_info(self):
        """记录内存信息到文件"""
        gpu_info = self.get_gpu_memory_info()
        sys_info = self.get_system_memory_info()
        current_time = datetime.now()
        
        if self.start_time is None:
            self.start_time = current_time
            
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        log_entry = f"\n=== {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Elapsed: {elapsed_time:.1f}s) ===\n"
        
        # 系统内存信息
        log_entry += f"System Memory:\n"
        log_entry += f"  Total: {sys_info['total']:.1f} MB\n"
        log_entry += f"  Used: {sys_info['used']:.1f} MB ({sys_info['percent']:.1f}%)\n"
        log_entry += f"  Available: {sys_info['available']:.1f} MB\n\n"
        
        # GPU显存信息
        if gpu_info:
            for gpu_name, info in gpu_info.items():
                log_entry += f"{gpu_name} Memory:\n"
                log_entry += f"  Allocated: {info['allocated']:.1f} MB\n"
                log_entry += f"  Reserved: {info['reserved']:.1f} MB\n"
                log_entry += f"  Max Allocated: {info['max_allocated']:.1f} MB\n"
                log_entry += f"  Max Reserved: {info['max_reserved']:.1f} MB\n"
                log_entry += f"  Total: {info['total']:.1f} MB\n"
                log_entry += f"  Free: {info['free']:.1f} MB\n"
                log_entry += f"  Usage: {info['usage_percent']:.1f}%\n\n"
                
                # 保存数据用于绘图
                self.memory_data.append(info['reserved'])
                self.time_data.append(elapsed_time)
        else:
            log_entry += "No GPU available\n\n"
            
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        return gpu_info, sys_info
    
    def print_current_status(self):
        """打印当前内存状态"""
        gpu_info, sys_info = self.log_memory_info()
        
        print(f"\n{'='*50}")
        print(f"Memory Status - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        print(f"System RAM: {sys_info['used']:.1f}/{sys_info['total']:.1f} MB ({sys_info['percent']:.1f}%)")
        
        if gpu_info:
            for gpu_name, info in gpu_info.items():
                print(f"{gpu_name}: {info['reserved']:.1f}/{info['total']:.1f} MB ({info['usage_percent']:.1f}%)")
                print(f"  - Allocated: {info['allocated']:.1f} MB")
                print(f"  - Max Allocated: {info['max_allocated']:.1f} MB")
        else:
            print("No GPU available")
    
    def start_monitoring(self, interval=5):
        """开始监控（后台线程）"""
        self.monitoring = True
        
        # 清空日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"GPU Memory Monitoring Started at {datetime.now()}\n")
            f.write("="*60 + "\n")
        
        def monitor_loop():
            while self.monitoring:
                self.print_current_status()
                time.sleep(interval)
                
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f"Memory monitoring started. Logging to {self.log_file}")
        print(f"Monitoring interval: {interval} seconds")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        print("Memory monitoring stopped.")
        
        # 生成图表
        if len(self.memory_data) > 1:
            self.plot_memory_usage()
    
    def plot_memory_usage(self):
        """绘制显存使用图表"""
        if not self.memory_data:
            print("No data to plot")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_data, self.memory_data, 'b-', linewidth=2, label='GPU Memory Reserved')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        max_memory = max(self.memory_data)
        avg_memory = np.mean(self.memory_data)
        plt.axhline(y=max_memory, color='r', linestyle='--', alpha=0.7, label=f'Max: {max_memory:.1f} MB')
        plt.axhline(y=avg_memory, color='g', linestyle='--', alpha=0.7, label=f'Avg: {avg_memory:.1f} MB')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Memory usage plot saved to {self.plot_file}")
        print(f"Peak memory usage: {max_memory:.1f} MB")
        print(f"Average memory usage: {avg_memory:.1f} MB")

def main():
    """主函数 - 可以直接运行进行实时监控"""
    monitor = GPUMemoryMonitor()
    
    print("GPU Memory Monitor")
    print("Commands:")
    print("  's' - Start monitoring")
    print("  'p' - Print current status")
    print("  'q' - Quit")
    print("  'r' - Reset max memory stats")
    
    while True:
        try:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 's':
                interval = input("Enter monitoring interval (seconds, default=5): ").strip()
                interval = int(interval) if interval.isdigit() else 5
                monitor.start_monitoring(interval)
                
            elif cmd == 'p':
                monitor.print_current_status()
                
            elif cmd == 'r':
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_max_memory_allocated(i)
                        torch.cuda.reset_max_memory_cached(i)
                    print("GPU memory stats reset.")
                else:
                    print("No GPU available.")
                    
            elif cmd == 'q':
                monitor.stop_monitoring()
                break
                
            else:
                print("Invalid command. Use 's', 'p', 'r', or 'q'.")
                
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()