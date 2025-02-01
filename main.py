import time
import random
import threading
import tkinter as tk
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # Ensure we use a Tk-compatible backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        return

    if algo.upper() == "RR":
        # Round Robin - Rotate queue to give each process fair time slice
        if len(queue) > 1:
            queue.append(queue.pop(0))

    elif algo.upper() == "WEIGHTEDRR":
        # Weighted Round Robin - Higher value processes get more CPU time
        # Sort by value and remaining time as secondary factor
        queue.sort(key=lambda p: (-p.value, p.remaining_time))
        
        # Calculate time quantum based on process value
        for proc in queue:
            proc.time_quantum = 0.1 * (1 + proc.value)  # 0.1-0.2s based on value

    elif algo.upper() == "SRTF":
        # Shortest Remaining Time First
        queue.sort(key=lambda p: (
            p.remaining_time,  # Primary: shortest remaining time
            p.end_deadline,    # Secondary: earliest deadline
            -p.value          # Tertiary: highest value
        ))

    elif algo.upper() == "FCFS":
        # First Come First Served - Sort by arrival time
        queue.sort(key=lambda p: (
            p.arrival_time,    # Primary: arrival time
            p.end_deadline,    # Secondary: deadline
            -p.value          # Tertiary: value
        ))

    elif algo.upper() == "RATEMONOTONIC":
        # Rate Monotonic - Higher priority to shorter deadlines
        queue.sort(key=lambda p: (
            p.end_deadline - p.arrival_time,  # Primary: period/deadline
            p.remaining_time,                 # Secondary: remaining time
            -p.value                         # Tertiary: value
        ))

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

    def run(self):
        self.thread_id = threading.get_ident()
        while self.running:
            time.sleep(0.1)
            now = time.time() - self.start_time
            to_ready = []
            to_remove = []
            
            # Get processes from incoming queue
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
                            
            # Update expired count
            for proc in to_remove:
                with self.expired_count['lock']:
                    self.expired_count['value'] += 1

            # Update ready queue with mutex lock
            with self.ready_lock:
                # Merge existing and new processes
                combined = list(self.ready_queue) + to_ready
                self.ready_queue.clear()
                
                # Apply scheduling layers
                for algo in self.algorithms:
                    reorder_queue(algo, combined)
                    
                # Final sort by score
                combined.sort(key=lambda x: (-x.score, x.end_deadline))
                
                # Keep top 20 processes
                self.ready_queue.extend(combined[:20])

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
        with self.expired_count['lock']:
            self.expired_count['value'] += 1
        self.cpu_stats[self.cpu_id]["proc_id"] = None
        self.cpu_stats[self.cpu_id]["proc_val"] = 0.0
        self.current_proc = None
        time.sleep(0.01)

    def _complete_current(self):
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


def main():

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
        
        thread_info.insert(tk.END, f"\nTotal Score: {total_score['value']:.2f}\n")
        thread_info.insert(tk.END, f"Processes Done: {len(executed_processes['list'])}\n")  
        thread_info.insert(tk.END, f"Missed Deadlines: {expired_count['value']}\n")
        thread_info.insert(tk.END, f"Failed Tasks: {failed_tasks['value']}\n")
        
        if thread_info.running:
            root.after(500, update_report)

    def start_system():
        nonlocal process_generator, scheduler, cpus, cpu_stats
        nonlocal usage_history, time_history, start_time
        global controller

        try:
            # Stop existing threads if any
            stop_system()
            
            num_cpus = int(cpu_entry.get())
            count = int(proc_entry.get())
            run_duration = float(time_entry.get())
            layers = layer_value.get()
            chosen_algos = [algo_vars[i].get() for i in range(layers)]

            # Clear stats
            cpu_stats.clear()
            usage_history.clear()
            time_history.clear()

            for i in range(num_cpus):
                cpu_stats[i] = {"proc_id": None, "proc_val": 0.0}
                usage_history[i] = []

            # Reset data structures
            with input_lock:
                input_queue.clear()
            with ready_lock:
                ready_queue.clear()
            expired_count['value'] = 0
            executed_processes['list'].clear()
            total_score['value'] = 0.0

            # Create new threads
            process_generator = ProcessGenerator(incoming_queue, incoming_lock, 
                                            count=count, duration=run_duration)
            scheduler = Scheduler(incoming_queue, incoming_lock,
                                ready_queue, ready_lock,
                                expired_count, layers=layers, 
                                algorithms=chosen_algos)
            cpus = []
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
                cpus.append(cpu_thread)

            # Create and start controller
            controller = MainController(cpus, scheduler, process_generator)
            controller.start()
            
            start_time = time.time()
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
                remaining = len(ready_queue)
            with incoming_lock:
                remaining += len(incoming_queue)
            with failed_tasks['lock']:
                failed_tasks['value'] = remaining

            # Stop threads
            if 'controller' in globals():
                controller.stop()
                controller.join(timeout=1.0)
                
            # Stop other threads
            if process_generator:
                process_generator.stop()
                process_generator.join(timeout=1.0)
            if scheduler:
                scheduler.stop()
                scheduler.join(timeout=1.0)
            for c in cpus:
                c.stop()
                c.join(timeout=1.0)

        except Exception as e:
            print(f"Error stopping system: {e}")
        finally:
            # Force reset states
            if 'controller' in globals():
                del controller
            process_generator = None
            scheduler = None
            cpus.clear()

    def update_plots(run_duration):
        now = time.time() - start_time
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

        if now < run_duration:
            root.after(500, lambda: update_plots(run_duration))
        else:
            stop_system()
            final_list = executed_processes['list']
            total_generated = process_generator.generated
            total_expired = expired_count['value']
            final_score = total_score['value']

            waiting_times = []
            response_times = []
            for p in final_list:
                w = (p.start_exec_time - p.arrival_time) if p.start_exec_time else 0.0
                r = (p.finish_exec_time - p.arrival_time) if p.finish_exec_time else 0.0
                waiting_times.append(w)
                response_times.append(r)

            avg_waiting = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
            avg_response = sum(response_times) / len(response_times) if response_times else 0.0
            hit_rate = len(final_list) / float(total_generated) if total_generated else 0.0

            print("-------- Final Results --------")
            print(f"Total Processes Generated: {total_generated}")
            print(f"Successfully Executed: {len(final_list)}")
            print(f"Missed Deadlines: {total_expired}")
            print(f"Failed Tasks: {failed_tasks['value']}")
            print(f"Total Tasks: {total_generated}")
            print(f"Hit Rate: {hit_rate:.2f}")
            print(f"Total Score: {final_score:.2f}")
            print(f"Average Waiting Time: {avg_waiting:.2f}")
            print(f"Average Response Time: {avg_response:.2f}")

    start_button = tk.Button(btn_frame, text="Start", command=start_system)
    start_button.pack(side=tk.LEFT, padx=10)

    stop_button = tk.Button(btn_frame, text="Stop", command=stop_system)
    stop_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()