import time
import random
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
import datetime 

import matplotlib
matplotlib.use('TkAgg')  # Ensure we use a Tk-compatible backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LOG_FILE = Path(__file__).parent / "scheduling_results.log"
PROCESS_LOG_FILE = Path(__file__).parent / "process_history.log"
failed_tasks = {'value': 0, 'lock': threading.Lock()}

def show_history_logs():
    """Show process execution history in new window with simple syntax highlighting."""
    log_window = tk.Toplevel()
    log_window.title("Process Execution History")
    log_window.geometry("800x600")

    text_widget = tk.Text(log_window, wrap=tk.NONE)
    text_widget.pack(expand=True, fill=tk.BOTH)

    y_scroll = tk.Scrollbar(log_window, orient=tk.VERTICAL, command=text_widget.yview)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll = tk.Scrollbar(log_window, orient=tk.HORIZONTAL, command=text_widget.xview)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    text_widget.config(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    # Define color tags for highlighting
    text_widget.tag_config('timestamp', foreground='green')
    text_widget.tag_config('process_id', foreground='yellow')
    text_widget.tag_config('completed_event', foreground='green')
    text_widget.tag_config('expired_event', foreground='red')
    text_widget.tag_config('header', foreground='cyan', underline=1)

    try:
        with open(PROCESS_LOG_FILE, 'r') as f:
            lines = f.readlines()
            # Insert CSV header as cyan, underlined
            header_line = "Timestamp,ProcessID,Event,CPU,ArrivalTime,ExecTime,StartTime,EndTime\n"
            text_widget.insert(tk.END, header_line, 'header')

            for line in lines:
                line_str = line.strip()
                # Skip the original CSV header or any empty lines
                if not line_str or line_str.startswith("Timestamp,ProcessID,Event"):
                    continue

                parts = line_str.split(',')
                if len(parts) < 8:
                    text_widget.insert(tk.END, line + "\n")
                    continue

                timestamp_str = parts[0]
                pid_str = parts[1]
                event_str = parts[2]
                cpu_str = parts[3]
                arrival_str = parts[4]
                exec_str = parts[5]
                start_str = parts[6]
                end_str = parts[7]

                # Insert timestamp in green
                text_widget.insert(tk.END, f"{timestamp_str} ", 'timestamp')
                # Insert process ID in yellow
                text_widget.insert(tk.END, f"{pid_str} ", 'process_id')

                # Highlight events
                if "COMPLETED" in event_str:
                    text_widget.insert(tk.END, event_str, 'completed_event')
                elif "EXPIRED" in event_str:
                    text_widget.insert(tk.END, event_str, 'expired_event')
                else:
                    text_widget.insert(tk.END, event_str)

                info_line = f", {cpu_str}, {arrival_str}, {exec_str}, {start_str}, {end_str}\n"
                text_widget.insert(tk.END, info_line)

    except Exception as e:
        text_widget.insert(tk.END, f"Error reading log: {e}")

    text_widget.config(state=tk.DISABLED)

def log_final_results(system_state, executed_list, stats):
    """Log final results to file"""
    try:
        config = {
            'cpus': len(system_state['cpus']),
            'processes': stats['total_generated'],
            'duration': stats['duration'],
            'layers': stats['layers'],
            'algorithms': stats['algorithms']
        }
        
        results = {
            'total_generated': stats['total_generated'],
            'completed': len(executed_list),
            'expired': stats['expired'],
            'failed': stats['failed'],
            'hit_rate': stats['hit_rate'],
            'score': stats['total_score'],
            'avg_waiting': stats['avg_waiting'],
            'avg_response': stats['avg_response']
        }
        
        write_run_results(config, results)
        
    except Exception as e:
        print(f"Error logging results: {e}")

def write_run_results(config, results):
    """Write run results to log file and print full results to console."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'='*50}\n"
        header += f"Run Date: {timestamp}\n"
        header += "Configuration:\n"
        header += f"- CPUs: {config['cpus']}\n"
        header += f"- Processes: {config['processes']}\n"
        header += f"- Duration: {config['duration']}s\n"
        header += f"- Scheduling Layers: {config['layers']}\n"
        header += f"- Algorithms: {', '.join(config['algorithms'])}\n"

        results_text = "\nResults:\n"
        results_text += f"- Total Processes: {results['total_generated']}\n"
        results_text += f"- Successfully Executed: {results['completed']}\n"
        results_text += f"- Missed Deadlines: {results['expired']}\n"
        results_text += f"- Failed Tasks: {results['failed']}\n"
        results_text += f"- Hit Rate: {results['hit_rate']:.2f}\n"
        results_text += f"- Total Score: {results['score']:.2f}\n"
        results_text += f"- Avg Waiting Time: {results['avg_waiting']:.2f}s\n"
        results_text += f"- Avg Response Time: {results['avg_response']:.2f}s\n"

        with open(LOG_FILE, 'a') as f:
            f.write(header + results_text)

        print(header + results_text)

    except Exception as e:
        print(f"Error writing to log file: {e}")

def calculate_heuristic_score(process, current_time):
    """
    Heuristic scoring based on:
    - Process value (40%)
    - Urgency (time until deadline) (30%)
    - Progress (executed portion) (30%)
    Returns normalized score 0-1
    """
    # Time remaining until deadline
    time_to_deadline = process.end_deadline - (current_time - process.arrival_time)
    if time_to_deadline <= 0:
        return 0
        
    # Normalize factors
    deadline_factor = min(1.0, time_to_deadline / process.end_deadline)
    execution_factor = 1.0 - (process.remaining_time / process.exec_time)
    
    # Calculate weighted score
    score = (0.4 * process.value + 
             0.3 * deadline_factor +
             0.3 * execution_factor)
             
    return score

def log_process_execution(process, cpu_id, event_type):
    """Log process execution events"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time_str = f"{process.start_exec_time:.2f}" if process.start_exec_time is not None else "NA"
        finish_time_str = f"{process.finish_exec_time:.2f}" if process.finish_exec_time is not None else "NA"
        with open(PROCESS_LOG_FILE, 'a') as f:
            f.write(
                f"{timestamp},{process.pid},{event_type},CPU{cpu_id},"
                f"{process.arrival_time:.2f},{process.exec_time:.2f},"
                f"{start_time_str},{finish_time_str}\n"
            )
    except Exception as e:
        print(f"Error logging process: {e}")

class MainController(threading.Thread):
    def __init__(self, cpus, scheduler, process_generator):
        super().__init__()
        self.cpus = cpus
        self.scheduler = scheduler
        self.process_generator = process_generator
        self.running = True
        self.thread_id = None
        self.start_time = time.time()
        self.run_duration = self.process_generator.duration
        
    def run(self):
        self.thread_id = threading.get_ident()
        try:
            # Start managed threads
            self.process_generator.start()
            self.scheduler.start()
            for cpu in self.cpus:
                cpu.start()
                
            while self.running:
                current_time = time.time() - self.start_time
                if current_time >= self.run_duration:
                    print("Runtime expired, stopping system...")
                    self.stop()
                    break
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Controller error: {e}")
            self.stop()
            
    def stop(self):
        self.running = False
        try:
            self.process_generator.stop()
            self.scheduler.stop()
            for cpu in self.cpus:
                cpu.stop()
                
            self.process_generator.join()
            self.scheduler.join()
            for cpu in self.cpus:
                cpu.join()
        except Exception as e:
            print(f"Error stopping threads: {e}")

class Process:
    def __init__(self, arrival_time, exec_time, start_deadline, end_deadline, value):
        self.arrival_time = arrival_time
        self.exec_time = exec_time
        self.remaining_time = exec_time
        self.start_deadline = start_deadline
        self.end_deadline = end_deadline
        self.value = value
        self.start_exec_time = None
        self.finish_exec_time = None
        self.pid = id(self)
        self.score = 0.0  # Store heuristic score


class ProcessGenerator(threading.Thread):
    def __init__(self, incoming_queue, incoming_lock, count=50, duration=30):
        super().__init__()
        self.incoming_queue = incoming_queue
        self.incoming_lock = incoming_lock
        self.running = True
        self.count = count
        self.duration = duration
        self.start_time = time.time()
        self.generated = 0
        # Add rate control
        self.generation_interval = duration / count if count > 0 else 0.1

    def run(self):
        while self.running and self.generated < self.count:
            current_time = time.time() - self.start_time
            if current_time >= self.duration:
                break
                
            exec_time = random.uniform(0.5, 2.0)  # Reduced max exec time
            start_dl = random.uniform(0.5, 3.0)   # Reduced deadline range
            end_dl = start_dl + exec_time + random.uniform(0.5, 1.0)
            value = random.uniform(0.0, 1.0)
            
            proc = Process(current_time, exec_time, start_dl, end_dl, value)
            with self.incoming_lock:
                self.incoming_queue.append(proc)
            self.generated += 1
            
            # Sleep for generation interval
            time.sleep(self.generation_interval)

    def stop(self):
        self.running = False

def reorder_queue(algo, queue):
    """
    Implements scheduling algorithms:
    - RR: Round Robin with time slices
    - WeightedRR: Weighted Round Robin based on process value
    - SRTF: Shortest Remaining Time First
    - FCFS: First Come First Served
    - RateMonotonic: Priority based on period/deadline
    """
    if not queue:
        return []

    # Copy to avoid in-place conflicts
    result = queue.copy()

    if algo.upper() == "RR":
        # Basic round-robin: rotate the queue
        if len(result) > 1:
            result.append(result.pop(0))

    elif algo.upper() == "WEIGHTEDRR":
        # Sort by highest value first, then rotate
        result.sort(key=lambda p: (-p.value, p.remaining_time))
        # Assign time_quantum based on value
        for proc in result:
            proc.time_quantum = 0.1 * (1 + proc.value)
        if len(result) > 1:
            result.append(result.pop(0))

    elif algo.upper() == "SRTF":
        # Shortest remaining time first
        result.sort(key=lambda p: (p.remaining_time, p.end_deadline, -p.value))

    elif algo.upper() == "FCFS":
        # First come, first served
        result.sort(key=lambda p: (p.arrival_time, p.end_deadline, -p.value))

    elif algo.upper() == "RATEMONOTONIC":
        # Sort by earliest (deadline - arrival), then shortest remaining
        result.sort(key=lambda p: (p.end_deadline - p.arrival_time, p.remaining_time, -p.value))

    return result

class Scheduler(threading.Thread):
    """
    Multi-layer scheduling with up to 3 layers (selected by user).
    Each layer can use a chosen algorithm: RR, weightedRR, SRTF, FCFS, RateMonotonic.
    The ready_queue capacity is limited to 20.
    """
    def __init__(self, incoming_queue, incoming_lock, ready_queue, ready_lock, 
                 expired_count, layers=1, algorithms=("FCFS",)):
        super().__init__()
        self.incoming_queue = incoming_queue
        self.incoming_lock = incoming_lock
        self.ready_queue = ready_queue
        self.ready_lock = ready_lock
        self.expired_count = expired_count
        self.running = True
        self.start_time = time.time()
        self.layers = layers
        self.algorithms = algorithms
        self.thread_id = None 
        self.layer_ratios = [1.0, 0.6, 0.4]

    def run(self):
        self.thread_id = threading.get_ident()
        while self.running:
            time.sleep(0.1)
            now = time.time() - self.start_time
            to_ready = []
            to_remove = []

            # Safely move processes from incoming_queue
            with self.incoming_lock:
                while self.incoming_queue:
                    proc = self.incoming_queue.popleft()
                    if (now - proc.arrival_time) > proc.end_deadline:
                        to_remove.append(proc)
                    else:
                        proc.score = calculate_heuristic_score(proc, now)
                        if proc.score > 0:
                            to_ready.append(proc)
                        else:
                            to_remove.append(proc)

            with self.expired_count['lock']:
                self.expired_count['value'] += len(to_remove)

            # Lock ready_queue before reordering
            with self.ready_lock:
                result_queue = list(self.ready_queue) + to_ready

                # Apply multi-layer algorithms
                for layer, algo in enumerate(self.algorithms):
                    layer_size = int(len(result_queue) * self.layer_ratios[layer])
                    layer_processes = result_queue[:layer_size]
                    layer_result = reorder_queue(algo, layer_processes)
                    result_queue = layer_result + result_queue[layer_size:]

                # Final sort by score, ensure top 20
                result_queue.sort(key=lambda x: (-x.score, x.end_deadline))
                self.ready_queue.clear()
                self.ready_queue.extend(result_queue[:20])

    def stop(self):
        self.running = False

class CPU(threading.Thread):
    def __init__(self, cpu_id, ready_queue, ready_lock, executed_processes,
                 total_score, expired_count, cpu_stats):
        super().__init__()
        self.cpu_id = cpu_id
        self.ready_queue = ready_queue
        self.ready_lock = ready_lock
        self.executed_processes = executed_processes
        self.total_score = total_score
        self.expired_count = expired_count
        self.running = True
        self.start_time = time.time()
        self.current_proc = None
        self.cpu_stats = cpu_stats
        self.thread_id = None

    def run(self):
        while self.running:
            self.thread_id = threading.get_ident()
            
            # Get next process with proper locking
            with self.ready_lock:
                if not self.current_proc and self.ready_queue:
                    self.current_proc = self.ready_queue.pop(0)
                    
            if self.current_proc:
                self.cpu_stats[self.cpu_id]["proc_id"] = self.current_proc.pid
                self.cpu_stats[self.cpu_id]["proc_val"] = self.current_proc.value

                now_global = time.time() - self.start_time

                # Check deadline
                if (now_global - self.current_proc.arrival_time) > self.current_proc.end_deadline:
                    self._expire_current()
                    continue

                if self.current_proc.start_exec_time is None:
                    self.current_proc.start_exec_time = now_global

                # Get time slice based on algorithm
                if hasattr(self.current_proc, 'time_quantum'):
                    # For Weighted RR
                    time_slice = min(self.current_proc.time_quantum, 
                                   self.current_proc.remaining_time)
                else:
                    # Default time slice
                    time_slice = min(0.05, self.current_proc.remaining_time)

                # Execute time slice
                time.sleep(time_slice)
                if not self.current_proc:
                    continue
                self.current_proc.remaining_time -= time_slice
                now_global = time.time() - self.start_time
                self.current_proc.finish_exec_time = now_global

                if self.current_proc.remaining_time <= 0:
                    # Process completed
                    if (now_global - self.current_proc.arrival_time) > self.current_proc.end_deadline:
                        self._expire_current()
                    else:
                        self._complete_current()
                else:
                    # Process needs more time
                    with self.ready_lock:
                        # For preemptive algorithms, check if should preempt
                        if self.ready_queue:
                            should_preempt = False
                            next_proc = self.ready_queue[0]
                            
                            if next_proc.remaining_time < self.current_proc.remaining_time:
                                # SRTF preemption
                                should_preempt = True
                            elif next_proc.value > self.current_proc.value:
                                # Value-based preemption
                                should_preempt = True
                                
                            if should_preempt:
                                self.ready_queue.append(self.current_proc)
                                self.current_proc = None
                                self.cpu_stats[self.cpu_id]["proc_id"] = None
                                self.cpu_stats[self.cpu_id]["proc_val"] = 0.0
                            else:
                                # For RR, move to back of queue
                                self.ready_queue.append(self.current_proc)
                                self.current_proc = None
                                self.cpu_stats[self.cpu_id]["proc_id"] = None
                                self.cpu_stats[self.cpu_id]["proc_val"] = 0.0

            else:
                self.cpu_stats[self.cpu_id]["proc_id"] = None  
                self.cpu_stats[self.cpu_id]["proc_val"] = 0.0
                time.sleep(0.01)
    def _expire_current(self):
        log_process_execution(self.current_proc, self.cpu_id, "EXPIRED")
        with self.expired_count['lock']:
            self.expired_count['value'] += 1
        self.cpu_stats[self.cpu_id]["proc_id"] = None
        self.cpu_stats[self.cpu_id]["proc_val"] = 0.0
        self.current_proc = None
        time.sleep(0.01)

    def _complete_current(self):
        log_process_execution(self.current_proc, self.cpu_id, "COMPLETED")
        with self.executed_processes['lock']:
            self.executed_processes['list'].append(self.current_proc)
        with self.total_score['lock']:
            self.total_score['value'] += self.current_proc.value
        self.cpu_stats[self.cpu_id]["proc_id"] = None
        self.cpu_stats[self.cpu_id]["proc_val"] = 0.0
        self.current_proc = None
        time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.current_proc is not None:
            log_process_execution(self.current_proc, self.cpu_id, "FAILED")
            # Increment the global failed_tasks counter for the process in progress
            with failed_tasks['lock']:
                failed_tasks['value'] += 1
            self.current_proc = None


def main():

    system_state = {
        'process_generator': None,
        'scheduler': None,
        'cpus': [],
        'controller': None,
        'running': False,
        'start_time': None 
    }

    failed_tasks = {'value': 0, 'lock': threading.Lock()}
    # Default settings
    default_num_cpus = 2
    default_count = 50
    default_run_duration = 30
    default_layers = 1
    # Provide the user with the new algorithms
    layer_algos = ("RR", "weightedRR", "SRTF", "FCFS", "RateMonotonic")

    incoming_queue = deque()
    incoming_lock = threading.Lock() 
    ready_queue = []
    ready_lock = threading.Lock()

    input_queue = deque()
    input_lock = threading.Lock()
    ready_queue = []
    ready_lock = threading.Lock()

    expired_count = {'value': 0, 'lock': threading.Lock()}
    executed_processes = {'list': [], 'lock': threading.Lock()}
    total_score = {'value': 0.0, 'lock': threading.Lock()}

    # Global references
    process_generator = None
    scheduler = None
    cpus = []
    cpu_stats = {}
    usage_history = {}
    time_history = []
    start_time = None

    root = tk.Tk()
    root.title("CPU Usage - Multi-Layer Scheduling")

    # Top frame for user inputs
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    tk.Label(top_frame, text="CPUs:").pack(side=tk.LEFT, padx=5)
    cpu_entry = tk.Entry(top_frame, width=5)
    cpu_entry.insert(tk.END, str(default_num_cpus))
    cpu_entry.pack(side=tk.LEFT)

    tk.Label(top_frame, text="Processes:").pack(side=tk.LEFT, padx=5)
    proc_entry = tk.Entry(top_frame, width=5)
    proc_entry.insert(tk.END, str(default_count))
    proc_entry.pack(side=tk.LEFT)

    tk.Label(top_frame, text="Run Time(s):").pack(side=tk.LEFT, padx=5)
    time_entry = tk.Entry(top_frame, width=5)
    time_entry.insert(tk.END, str(default_run_duration))
    time_entry.pack(side=tk.LEFT)

    layer_control_frame = tk.Frame(top_frame)
    layer_control_frame.pack(side=tk.LEFT, padx=5)
    
    tk.Label(layer_control_frame, text="Layers:").pack(side=tk.LEFT)
    layer_value = tk.IntVar(value=default_layers)
    layer_label = tk.Label(layer_control_frame, textvariable=layer_value)
    layer_label.pack(side=tk.LEFT, padx=5)
    
    def increment_layers():
        current = layer_value.get()
        if current < 3:
            layer_value.set(current + 1)
            update_layer_controls()
            
    def decrement_layers():
        current = layer_value.get()
        if current > 1:
            layer_value.set(current - 1)
            update_layer_controls()
    
    tk.Button(layer_control_frame, text="-", command=decrement_layers).pack(side=tk.LEFT)
    tk.Button(layer_control_frame, text="+", command=increment_layers).pack(side=tk.LEFT)

    

    # Scheduling algo selectors for each layer
    layer_frame = tk.Frame(top_frame)
    layer_frame.pack(side=tk.LEFT)
    
    algo_vars = []
    def update_layer_controls(*args):
        for widget in layer_frame.winfo_children():
            widget.destroy()
            
        num_layers = layer_value.get()
        algo_vars.clear()
        
        for i in range(num_layers):
            tk.Label(layer_frame, text=f"Layer {i+1}:").pack(side=tk.LEFT)
            var = tk.StringVar(root)
            var.set(layer_algos[0])
            algo_vars.append(var)
            tk.OptionMenu(layer_frame, var, *layer_algos).pack(side=tk.LEFT, padx=5)
            
    update_layer_controls() # Initial setup

    # Start/Stop buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    history_button = tk.Button(btn_frame, text="Show History Logs", command=show_history_logs)
    history_button.pack(side=tk.LEFT, padx=10)

    fig = plt.Figure(figsize=(8, 6), dpi=100)
    ax_bar = fig.add_subplot(211)
    ax_line = fig.add_subplot(212)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    report_frame = tk.Frame(root)
    report_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    thread_info = tk.Text(report_frame, height=10)
    thread_info.pack(fill=tk.X)
    thread_info.running = False

    def update_report():
        thread_info.delete('1.0', tk.END)
        if 'controller' in globals():
            # Safe thread ID access
            controller_id = getattr(controller, 'thread_id', 'Not started')
            scheduler_id = getattr(scheduler, 'thread_id', 'Not started')
            
            thread_info.insert(tk.END, f"Main Controller ID: {controller_id}\n")
            thread_info.insert(tk.END, f"Scheduler ID: {scheduler_id}\n")
            thread_info.insert(tk.END, "CPU Thread IDs:\n")
            for cpu in cpus:
                cpu_id = getattr(cpu, 'thread_id', 'Not started')
                thread_info.insert(tk.END, f"CPU {cpu.cpu_id}: {cpu_id}\n")
        
        total_processes = len(executed_processes['list']) + expired_count['value'] + failed_tasks['value']
        
        thread_info.insert(tk.END, f"\nTotal Score: {total_score['value']:.2f}\n")
        thread_info.insert(tk.END, f"Total Processes: {total_processes}\n")
        thread_info.insert(tk.END, f"Completed: {len(executed_processes['list'])}\n")  
        thread_info.insert(tk.END, f"Missed Deadlines: {expired_count['value']}\n")
        thread_info.insert(tk.END, f"Failed/Unfinished: {failed_tasks['value']}\n")
        
        if thread_info.running:
            root.after(500, update_report)

    def start_system():
        nonlocal process_generator, scheduler, cpus, cpu_stats
        nonlocal usage_history, time_history, start_time
        global controller

        try:
            # Overwrite process history file at the start of each run
            with open(PROCESS_LOG_FILE, 'w') as f:
                # Initialize with the CSV header
                f.write("Timestamp,ProcessID,Event,CPU,ArrivalTime,ExecTime,StartTime,EndTime\n")

            # Stop any running system
            stop_system()

            system_state['start_time'] = time.time()
            
            # Get configuration
            num_cpus = int(cpu_entry.get())
            for i in range(num_cpus):
                usage_history[i] = [0.0]
            count = int(proc_entry.get())
            run_duration = float(time_entry.get())
            layers = layer_value.get()
            chosen_algos = [algo_vars[i].get() for i in range(layers)]

            # Clear stats
            cpu_stats.clear()
            usage_history.clear()
            time_history.clear()

            # Initialize CPUs
            for i in range(num_cpus):
                cpu_stats[i] = {"proc_id": None, "proc_val": 0.0}
                usage_history[i] = []
            
            time_history.append(0.0)
            for i in range(num_cpus):
                usage_history[i].append(0.0)

            # Reset data structures
            with input_lock:
                input_queue.clear()
            with ready_lock:
                ready_queue.clear()
            expired_count['value'] = 0
            executed_processes['list'].clear()
            total_score['value'] = 0.0

            # Create threads
            system_state['process_generator'] = ProcessGenerator(
                incoming_queue, incoming_lock, count=count, duration=run_duration)
            
            system_state['scheduler'] = Scheduler(
                incoming_queue, incoming_lock, ready_queue, ready_lock,
                expired_count, layers=layers, algorithms=chosen_algos)
            
            system_state['cpus'] = []
            for i in range(num_cpus):
                cpu_thread = CPU(
                    cpu_id=i,
                    ready_queue=ready_queue,
                    ready_lock=ready_lock, 
                    executed_processes=executed_processes,
                    total_score=total_score,
                    expired_count=expired_count,
                    cpu_stats=cpu_stats
                )
                system_state['cpus'].append(cpu_thread)

            # Create and start controller
            system_state['controller'] = MainController(
                system_state['cpus'],
                system_state['scheduler'],
                system_state['process_generator']
            )
            system_state['controller'].start()
            
            system_state['running'] = True
            thread_info.running = True
            update_report()
            update_plots(run_duration)
            
        except Exception as e:
            print(f"Error starting system: {e}")
            stop_system()


    def stop_system():
        thread_info.running = False
        try:
            # Count remaining tasks before stopping threads
            with ready_lock:
                # Log remaining tasks in ready_queue as FAILED
                for p in ready_queue:
                    log_process_execution(p, -1, "FAILED")
                remaining = len(ready_queue)
                ready_queue.clear()

            with incoming_lock:
                # Log remaining tasks in incoming_queue as FAILED
                for p in incoming_queue:
                    log_process_execution(p, -1, "FAILED")
                remaining += len(incoming_queue)
                incoming_queue.clear()

            with failed_tasks['lock']:
                failed_tasks['value'] = remaining

            # Stop threads safely
            if system_state['controller']:
                system_state['controller'].stop()
                system_state['controller'].join(timeout=1.0)
            if system_state['process_generator']:
                system_state['process_generator'].stop()
                system_state['process_generator'].join(timeout=1.0)
            if system_state['scheduler']:
                system_state['scheduler'].stop() 
                system_state['scheduler'].join(timeout=1.0)
            for cpu in system_state['cpus']:
                cpu.stop()
                cpu.join(timeout=1.0)

        except Exception as e:
            print(f"Error stopping system: {e}")
        finally:
            system_state['controller'] = None
            system_state['process_generator'] = None
            system_state['scheduler'] = None
            system_state['cpus'].clear()
            system_state['running'] = False
            
    def update_plots(run_duration):
        if not system_state['running'] or system_state['start_time'] is None:
            return
        
        now = time.time() - system_state['start_time']
        cpu_ids = list(cpu_stats.keys())
        proc_vals = [cpu_stats[cid]["proc_val"] for cid in cpu_ids]

        time_history.append(now)
        for cid in cpu_ids:
            usage_history[cid].append(cpu_stats[cid]["proc_val"])

        # Bar chart
        ax_bar.clear()
        labels = []
        for cid in cpu_ids:
            pid = cpu_stats[cid]["proc_id"]
            labels.append(f"CPU{cid}:PID{pid}" if pid else f"CPU{cid}:(idle)")
        ax_bar.bar(labels, proc_vals, color='green', alpha=0.6)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Current Value (score)")
        ax_bar.set_title("Real-Time CPU Use")

        # Line chart
        ax_line.clear()
        for cid in cpu_ids:
            ax_line.plot(time_history, usage_history[cid], label=f"CPU{cid}")
        ax_line.set_ylim(0, 1)
        ax_line.set_xlabel("Time (sec)")
        ax_line.set_ylabel("Value (score)")
        ax_line.set_title("CPU Usage Over Time")
        ax_line.legend(loc="upper right")

        fig.tight_layout()
        canvas.draw()

        if now < run_duration and system_state['running']:
            root.after(100, lambda: update_plots(run_duration))
        else:
            # Collect final stats
            final_list = executed_processes['list']
            total_generated = (
                system_state['process_generator'].generated
                if system_state['process_generator'] else 0
            )
            total_expired = expired_count['value']
            final_score = total_score['value']

            waiting_times = []
            response_times = []
            for p in final_list:
                if p.start_exec_time is not None:
                    waiting_times.append(p.start_exec_time - p.arrival_time)
                if p.finish_exec_time is not None:
                    response_times.append(p.finish_exec_time - p.arrival_time)

            stats = {
                'total_generated': total_generated,
                'expired': total_expired,
                'failed': failed_tasks['value'],
                'duration': run_duration,
                'layers': layer_value.get(),
                'algorithms': [v.get() for v in algo_vars],
                'total_score': final_score,
                'hit_rate': (
                    len(final_list) / float(total_generated)
                    if total_generated else 0.0
                ),
                'avg_waiting': (
                    sum(waiting_times)/len(waiting_times)
                    if waiting_times else 0.0
                ),
                'avg_response': (
                    sum(response_times)/len(response_times)
                    if response_times else 0.0
                )
            }

            log_final_results(system_state, final_list, stats)

            stop_system()

    start_button = tk.Button(btn_frame, text="Start", command=start_system)
    start_button.pack(side=tk.LEFT, padx=10)

    stop_button = tk.Button(btn_frame, text="Stop", command=stop_system)
    stop_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()