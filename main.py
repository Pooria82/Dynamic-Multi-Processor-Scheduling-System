import time
import random
import threading
import tkinter as tk
from collections import deque

import matplotlib
matplotlib.use('TkAgg')  # Ensure we use a Tk-compatible backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

###############################################################################
#                                Data Classes                                 #
###############################################################################

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

###############################################################################
#                                Threads                                      #
###############################################################################

class ProcessGenerator(threading.Thread):
    def __init__(self, input_queue, input_lock, count=50, duration=30):
        super().__init__()
        self.input_queue = input_queue
        self.input_lock = input_lock
        self.running = True
        self.count = count
        self.duration = duration
        self.start_time = time.time()
        self.generated = 0

    def run(self):
        while self.running and self.generated < self.count and (time.time() - self.start_time) < self.duration:
            current_time = time.time() - self.start_time
            exec_time = random.uniform(0.5, 3.0)
            start_dl = random.uniform(1.0, 5.0)
            end_dl = start_dl + exec_time + random.uniform(1.0, 2.0)
            value = random.uniform(0.0, 1.0)
            proc = Process(current_time, exec_time, start_dl, end_dl, value)
            with self.input_lock:
                self.input_queue.append(proc)
            self.generated += 1
            time.sleep(random.uniform(0.05, 0.2))

    def stop(self):
        self.running = False

class Scheduler(threading.Thread):
    """
    Multi-layer scheduling with up to 3 layers (selected by user).
    Each layer can use a chosen algorithm: FIFO, SJF, or Priority.
    The ready_queue capacity is limited to 20.
    """
    def __init__(self, input_queue, input_lock, ready_queue, ready_lock, expired_count,
                 layers=1, algorithms=("FIFO",)):
        super().__init__()
        self.input_queue = input_queue
        self.input_lock = input_lock
        self.ready_queue = ready_queue
        self.ready_lock = ready_lock
        self.expired_count = expired_count
        self.running = True
        self.start_time = time.time()

        self.layers = layers
        self.algorithms = algorithms

    def run(self):
        while self.running:
            time.sleep(0.1)
            now = time.time() - self.start_time
            to_ready = []
            to_remove = []
            with self.input_lock:
                for proc in self.input_queue:
                    # Check deadlines
                    if (now - proc.arrival_time) > proc.start_deadline:
                        to_remove.append(proc)
                    elif (now - proc.arrival_time) > proc.end_deadline:
                        to_remove.append(proc)
                for proc in self.input_queue:
                    if proc not in to_remove:
                        to_ready.append(proc)
                for proc in to_remove:
                    if proc in self.input_queue:
                        self.input_queue.remove(proc)
                        with self.expired_count['lock']:
                            self.expired_count['value'] += 1
                self.input_queue.clear()

            # Merge new procs with ready queue
            with self.ready_lock:
                combined = list(self.ready_queue) + to_ready
                self.ready_queue.clear()

                # Run each layer scheduling
                for algo in self.algorithms:
                    if algo.upper() == "FIFO":
                        combined.sort(key=lambda x: x.arrival_time)
                    elif algo.upper() == "SJF":
                        combined.sort(key=lambda x: x.exec_time)
                    else:  # Priority-based
                        combined.sort(key=lambda x: (-x.value, x.end_deadline))

                # Truncate to max 20
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

    def run(self):
        while self.running:
            with self.ready_lock:
                if not self.current_proc and self.ready_queue:
                    self.current_proc = self.ready_queue.pop(0)

            if self.current_proc:
                self.cpu_stats[self.cpu_id]["proc_id"] = self.current_proc.pid
                self.cpu_stats[self.cpu_id]["proc_val"] = self.current_proc.value

                now_global = time.time() - self.start_time
                # Check end-deadline
                if (now_global - self.current_proc.arrival_time) > self.current_proc.end_deadline:
                    self._expire_current()
                    continue

                if self.current_proc.start_exec_time is None:
                    self.current_proc.start_exec_time = now_global

                # Time slice
                time_slice = min(0.05, self.current_proc.remaining_time)
                time.sleep(time_slice)
                self.current_proc.remaining_time -= time_slice
                now_global = time.time() - self.start_time
                self.current_proc.finish_exec_time = now_global

                if self.current_proc.remaining_time <= 0:
                    # Check again deadline upon finish
                    if (now_global - self.current_proc.arrival_time) > self.current_proc.end_deadline:
                        self._expire_current()
                    else:
                        self._complete_current()
                else:
                    # Preempt if higher-value process arrived
                    with self.ready_lock:
                        higher_in_queue = any(p.value > self.current_proc.value for p in self.ready_queue)
                        if higher_in_queue:
                            self.ready_queue.append(self.current_proc)
                            self.ready_queue.sort(key=lambda x: (-x.value, x.end_deadline))
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

###############################################################################
#                                Main                                         #
###############################################################################

def main():
    # Default settings
    default_num_cpus = 2
    default_count = 50
    default_run_duration = 30
    default_layers = 1
    layer_algos = ("FIFO", "SJF", "Priority")

    ###########################################################################
    # Build main structures
    ###########################################################################
    input_queue = deque()
    input_lock = threading.Lock()
    ready_queue = []
    ready_lock = threading.Lock()

    expired_count = {'value': 0, 'lock': threading.Lock()}
    executed_processes = {'list': [], 'lock': threading.Lock()}
    total_score = {'value': 0.0, 'lock': threading.Lock()}

    # Global references to threads
    process_generator = None
    scheduler = None
    cpus = []
    cpu_stats = {}
    usage_history = {}
    time_history = []
    start_time = None

    ###########################################################################
    # Tkinter GUI
    ###########################################################################
    root = tk.Tk()
    root.title("CPU Usage - Multi-Layer Scheduling")

    # Top frame for user inputs
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    # Entry: number of CPUs
    tk.Label(top_frame, text="CPUs:").pack(side=tk.LEFT, padx=5)
    cpu_entry = tk.Entry(top_frame, width=5)
    cpu_entry.insert(tk.END, str(default_num_cpus))
    cpu_entry.pack(side=tk.LEFT)

    # Entry: number of Processes
    tk.Label(top_frame, text="Processes:").pack(side=tk.LEFT, padx=5)
    proc_entry = tk.Entry(top_frame, width=5)
    proc_entry.insert(tk.END, str(default_count))
    proc_entry.pack(side=tk.LEFT)

    # Entry: run duration
    tk.Label(top_frame, text="Run Time(s):").pack(side=tk.LEFT, padx=5)
    time_entry = tk.Entry(top_frame, width=5)
    time_entry.insert(tk.END, str(default_run_duration))
    time_entry.pack(side=tk.LEFT)

    # Entry: layers
    tk.Label(top_frame, text="Layers(1-3):").pack(side=tk.LEFT, padx=5)
    layers_entry = tk.Entry(top_frame, width=5)
    layers_entry.insert(tk.END, str(default_layers))
    layers_entry.pack(side=tk.LEFT)

    # Scheduling algo selectors for each layer
    algo_vars = []
    for i in range(3):
        var = tk.StringVar(root)
        var.set(layer_algos[i])  # default setting
        algo_vars.append(var)
        tk.OptionMenu(top_frame, var, *layer_algos).pack(side=tk.LEFT, padx=5)

    # Start / Stop buttons frame
    btn_frame = tk.Frame(root)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    # Figure for plotting
    fig = plt.Figure(figsize=(8, 6), dpi=100)
    ax_bar = fig.add_subplot(211)
    ax_line = fig.add_subplot(212)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_system():
        nonlocal process_generator, scheduler, cpus, cpu_stats, usage_history
        nonlocal time_history, start_time

        # Read GUI settings
        num_cpus = int(cpu_entry.get())
        count = int(proc_entry.get())
        run_duration = float(time_entry.get())
        layers = int(layers_entry.get())
        chosen_algos = [algo_vars[i].get() for i in range(layers)]

        # Clear old references
        cpu_stats.clear()
        usage_history.clear()
        time_history.clear()

        # Initialize stats
        for i in range(num_cpus):
            cpu_stats[i] = {"proc_id": None, "proc_val": 0.0}
            usage_history[i] = []

        # Setup objects
        # Clear queues
        with input_lock:
            input_queue.clear()
        with ready_lock:
            ready_queue.clear()
        expired_count['value'] = 0
        executed_processes['list'].clear()
        total_score['value'] = 0.0

        # Create threads based on input
        process_generator = ProcessGenerator(input_queue, input_lock, count=count, duration=run_duration)
        scheduler = Scheduler(input_queue, input_lock, ready_queue, ready_lock,
                              expired_count, layers=layers, algorithms=chosen_algos)

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

        # Start them
        process_generator.start()
        scheduler.start()
        for c in cpus:
            c.start()

        start_time = time.time()
        update_plots(run_duration)

    def stop_system():
        # Stop everything
        if process_generator:
            process_generator.stop()
            process_generator.join()
        if scheduler:
            scheduler.stop()
            scheduler.join()
        for c in cpus:
            c.stop()
        for c in cpus:
            c.join()

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
            print(f"Hit Rate: {hit_rate:.2f}")
            print(f"Total Score: {final_score:.2f}")
            print(f"Average Waiting Time: {avg_waiting:.2f}")
            print(f"Average Response Time: {avg_response:.2f}")

    # Buttons to run/stop
    start_button = tk.Button(btn_frame, text="Start", command=start_system)
    start_button.pack(side=tk.LEFT, padx=10)

    stop_button = tk.Button(btn_frame, text="Stop", command=stop_system)
    stop_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()