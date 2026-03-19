#!/usr/bin/env python3
"""
Launch the NIXL EP elastic test with each rank as a fully independent process,
mimicking the Ray worker process model used by vLLM.

Unlike elastic.py which uses torch.multiprocessing.spawn (all workers forked
from one parent), this script launches each worker via subprocess.Popen so
each process has an independent Python interpreter, CUDA context, and UCX
initialization -- matching how Ray workers behave.

Usage:
    python elastic_independent.py \
        --disable-ll-nvlink \
        --num-tokens 256 --hidden-dim 2048 \
        --num-experts-per-rank 16 --num-topk 6 \
        --plan double_expansion.json \
        --num-processes 8
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import torch.multiprocessing


def run_server():
    import store_group
    import rank_server
    _store = store_group.create_master_store(port=9999)
    rank_server.start_server(port=10000)


def main():
    parser = argparse.ArgumentParser(
        description="Elastic EP Test — independent process launcher"
    )
    parser.add_argument("--plan", type=str, default="plan.json")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-experts-per-rank", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--tcp-server", type=str, default=None)
    parser.add_argument("--kineto", action="store_true")
    parser.add_argument("--disable-ll-nvlink", action="store_true")
    parser.add_argument("--combined-mode", action="store_true",
                        help="Use combined SEND+RECV instead of split SEND_ONLY")
    args = parser.parse_args()

    server_addr = args.tcp_server or "127.0.0.1"

    if not args.tcp_server:
        print("Starting TCPStore and rank server locally", flush=True)
        server_process = torch.multiprocessing.Process(
            target=run_server, daemon=True
        )
        server_process.start()
        time.sleep(1.0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "elastic_independent_worker.py")

    worker_args = [
        "--plan", args.plan,
        "--num-tokens", str(args.num_tokens),
        "--num-experts-per-rank", str(args.num_experts_per_rank),
        "--hidden-dim", str(args.hidden_dim),
        "--num-topk", str(args.num_topk),
        "--tcp-server", server_addr,
    ]
    if args.kineto:
        worker_args.append("--kineto")
    if args.disable_ll_nvlink:
        worker_args.append("--disable-ll-nvlink")
    if args.combined_mode:
        worker_args.append("--combined-mode")

    processes = []
    for i in range(args.num_processes):
        cmd = [sys.executable, worker_script, "--torch-rank", str(i)] + worker_args
        print(f"Launching worker {i}: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ.copy(),
        )
        processes.append(proc)
        time.sleep(0.1)

    for proc in processes:
        proc.wait()

    print("All workers finished.", flush=True)


if __name__ == "__main__":
    main()
