import json
import os
import time
from collections import defaultdict

import psutil
import torch


_current_phase = "init"
_phase_stack = []


def _new_metric():
    return {
        "submit_s": 0.0,
        "wait_s": 0.0,
        "exposed_s": 0.0,
        "read_bytes": 0,
        "write_bytes": 0,
        "ops": 0,
    }


_metrics = defaultdict(_new_metric)
_phase_metrics = defaultdict(lambda: defaultdict(_new_metric))
_step_records = []
_run_start = time.perf_counter()
_worker_update_s = 0.0
_worker_update_count = 0


def set_phase(phase):
    global _current_phase
    _current_phase = str(phase)


class phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _phase_stack.append(_current_phase)
        set_phase(self.name)

    def __exit__(self, exc_type, exc, tb):
        previous = _phase_stack.pop() if _phase_stack else "unknown"
        set_phase(previous)


def _tensor_bytes(tensors):
    total = 0
    for tensor in tensors or []:
        if torch.is_tensor(tensor):
            total += int(tensor.numel()) * int(tensor.element_size())
    return total


def record_io(category, direction, seconds, tensors=None, bytes_count=None, exposed=False):
    key = f"{category}.{direction}"
    bytes_value = int(bytes_count if bytes_count is not None else _tensor_bytes(tensors))
    metric = _metrics[key]
    phase_metric = _phase_metrics[_current_phase][key]
    seconds = float(seconds)
    metric["ops"] += 1
    phase_metric["ops"] += 1
    if direction.startswith("read"):
        metric["read_bytes"] += bytes_value
        phase_metric["read_bytes"] += bytes_value
    elif direction.startswith("write"):
        metric["write_bytes"] += bytes_value
        phase_metric["write_bytes"] += bytes_value
    if exposed:
        metric["wait_s"] += seconds
        metric["exposed_s"] += seconds
        phase_metric["wait_s"] += seconds
        phase_metric["exposed_s"] += seconds
    else:
        metric["submit_s"] += seconds
        phase_metric["submit_s"] += seconds


def record_wait(category, seconds, ops=1):
    record_io(category, "wait", seconds, bytes_count=0, exposed=True)
    _metrics[f"{category}.wait"]["ops"] += max(0, int(ops) - 1)
    _phase_metrics[_current_phase][f"{category}.wait"]["ops"] += max(0, int(ops) - 1)


def record_worker_update(seconds):
    global _worker_update_s, _worker_update_count
    _worker_update_s += float(seconds)
    _worker_update_count += 1


def memory_snapshot():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    result = {
        "dram_rss_gib": rss / (1024**3),
        "hbm_allocated_gib": 0.0,
        "hbm_reserved_gib": 0.0,
        "hbm_peak_allocated_gib": 0.0,
        "hbm_peak_reserved_gib": 0.0,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "hbm_allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                "hbm_reserved_gib": torch.cuda.memory_reserved() / (1024**3),
                "hbm_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "hbm_peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
        )
    return result


def reset_cuda_peaks():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def snapshot_phase_totals():
    totals = {}
    for phase_name, category_metrics in _phase_metrics.items():
        submit = sum(metric["submit_s"] for metric in category_metrics.values())
        wait = sum(metric["wait_s"] for metric in category_metrics.values())
        exposed = sum(metric["exposed_s"] for metric in category_metrics.values())
        read_bytes = sum(metric["read_bytes"] for metric in category_metrics.values())
        write_bytes = sum(metric["write_bytes"] for metric in category_metrics.values())
        totals[phase_name] = {
            "io_submit_s": submit,
            "io_wait_s": wait,
            "io_exposed_s": exposed,
            "io_read_gib": read_bytes / (1024**3),
            "io_write_gib": write_bytes / (1024**3),
        }
    return totals


def record_step(step, forward_s, backward_s, update_s, total_s):
    phase_totals = snapshot_phase_totals()
    memory = memory_snapshot()
    record = {
        "step": int(step),
        "forward_s": float(forward_s),
        "backward_s": float(backward_s),
        "update_s": float(update_s),
        "total_s": float(total_s),
        "phases": phase_totals,
        "memory": memory,
    }
    _step_records.append(record)
    forward_io = phase_totals.get("forward", {})
    backward_io = phase_totals.get("backward", {})
    update_io = phase_totals.get("update", {})
    print(
        "RatelStepMetrics "
        f"step={step} forward_s={forward_s:.6f} backward_s={backward_s:.6f} "
        f"update_s={update_s:.6f} total_s={total_s:.6f} "
        f"forward_io_exposed_s={forward_io.get('io_exposed_s', 0.0):.6f} "
        f"backward_io_exposed_s={backward_io.get('io_exposed_s', 0.0):.6f} "
        f"update_io_exposed_s={update_io.get('io_exposed_s', 0.0):.6f} "
        f"hbm_peak_allocated_gib={memory['hbm_peak_allocated_gib']:.3f} "
        f"hbm_peak_reserved_gib={memory['hbm_peak_reserved_gib']:.3f} "
        f"dram_rss_gib={memory['dram_rss_gib']:.3f}",
        flush=True,
    )


def summary(last_n=5):
    phase_totals = snapshot_phase_totals()
    return {
        "elapsed_s": time.perf_counter() - _run_start,
        "last_n": int(last_n),
        "phase_totals": phase_totals,
        "category_totals": dict(_metrics),
        "steps": _step_records,
        "worker_update_s": _worker_update_s,
        "worker_update_count": _worker_update_count,
        "memory": memory_snapshot(),
    }


def print_summary(last_n=5, detail=False):
    data = summary(last_n=last_n)
    if detail:
        print("RatelSummaryJSON " + json.dumps(data, sort_keys=True), flush=True)
    for phase_name in ("forward", "backward", "update"):
        values = data["phase_totals"].get(phase_name, {})
        print(
            "RatelPhaseIO "
            f"phase={phase_name} "
            f"submit_s={values.get('io_submit_s', 0.0):.6f} "
            f"wait_s={values.get('io_wait_s', 0.0):.6f} "
            f"exposed_s={values.get('io_exposed_s', 0.0):.6f} "
            f"read_gib={values.get('io_read_gib', 0.0):.3f} "
            f"write_gib={values.get('io_write_gib', 0.0):.3f}",
            flush=True,
        )
    memory = data["memory"]
    print(
        "RatelMemoryPeak "
        f"hbm_allocated_gib={memory['hbm_allocated_gib']:.3f} "
        f"hbm_reserved_gib={memory['hbm_reserved_gib']:.3f} "
        f"hbm_peak_allocated_gib={memory['hbm_peak_allocated_gib']:.3f} "
        f"hbm_peak_reserved_gib={memory['hbm_peak_reserved_gib']:.3f} "
        f"dram_rss_gib={memory['dram_rss_gib']:.3f}",
        flush=True,
    )
