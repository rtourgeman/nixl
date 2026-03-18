#!/usr/bin/env python3
"""
Single-rank worker for the independent-process elastic EP test.
Launched by elastic_independent.py via subprocess.Popen.

This is functionally identical to the worker() function in elastic.py,
but runs as a fully independent Python process.
"""

import argparse
import os
import signal
import sys
from functools import partial

import nixl_ep
import rank_server
import store_group
import torch
from plan import Plan

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import bench, calc_diff, hash_tensor, per_token_cast_back  # noqa: E402

TCP_STORE_PORT = 9999
RANK_SERVER_PORT = 10000


def handle_sigterm(signum, frame, buffer, plan, rank_client):
    print(f"SIGTERM received for PID {os.getpid()}", flush=True)
    if plan is not None:
        rank_client.release_rank(user_context=plan.get_phase())
    else:
        rank_client.release_rank()
    if buffer is not None and buffer.runtime is not None:
        buffer.destroy()
        del buffer
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-rank", type=int, required=True)
    parser.add_argument("--plan", type=str, required=True)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-experts-per-rank", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--tcp-server", type=str, required=True)
    parser.add_argument("--kineto", action="store_true")
    parser.add_argument("--disable-ll-nvlink", action="store_true")
    args = parser.parse_args()

    server_addr = args.tcp_server
    rank_client = rank_server.RankClient(server_addr, RANK_SERVER_PORT)
    local_rank, global_rank, last_active_phase = rank_client.get_rank()
    plan = Plan(
        args.plan,
        global_rank,
        start_phase=last_active_phase if last_active_phase is not None else 0,
    )
    if plan.current_phase == -1:
        print(
            f"[PID {os.getpid()}] Process {args.torch_rank} -> "
            f"no plan phases for rank {global_rank}, exiting",
            flush=True,
        )
        return

    max_num_ranks = plan.get_max_rank() + 1
    print(
        f"[PID {os.getpid()}] Process {args.torch_rank} -> "
        f"global_rank={global_rank}, local_rank={local_rank}",
        flush=True,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    tcp_store = store_group.create_client_store(
        master_addr=server_addr,
        port=TCP_STORE_PORT,
    )

    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens,
        args.hidden_dim,
        max_num_ranks,
        args.num_experts_per_rank * max_num_ranks,
    )
    if local_rank == 0:
        print(f"Allocating buffer size: {num_rdma_bytes / 1e6} MB ...", flush=True)

    buffer = nixl_ep.Buffer(
        rank=global_rank,
        disable_ll_nvlink=True,
        explicitly_destroy=True,
        tcp_store_group=tcp_store,
    )
    buffer.update_memory_buffers(
        num_ranks=max_num_ranks,
        num_experts_per_rank=args.num_experts_per_rank,
        num_rdma_bytes=num_rdma_bytes,
    )
    signal.signal(
        signal.SIGTERM,
        partial(handle_sigterm, buffer=buffer, plan=plan, rank_client=rank_client),
    )
    remote_ranks = set()
    mask_status = torch.zeros((max_num_ranks,), dtype=torch.int32, device="cuda")

    # Import test_main from elastic.py
    from elastic import test_main

    while True:
        print(
            f"[PID {os.getpid()}] rank={global_rank} -> "
            f"start phase {plan.get_phase()}",
            flush=True,
        )

        added_ranks = plan.get_new_ranks()
        cleanly_removed = plan.get_removed_ranks()

        if global_rank in cleanly_removed:
            print(f"[PID {os.getpid()}] rank={global_rank} -> removed, exiting", flush=True)
            rank_client.release_rank(user_context=plan.get_phase())
            break

        if len(added_ranks) > 0:
            print(
                f"[PID {os.getpid()}] rank={global_rank} -> "
                f"adding connections to {added_ranks}",
                flush=True,
            )
            buffer.connect_ranks(added_ranks)
            remote_ranks.update(added_ranks)

        if len(cleanly_removed) > 0:
            print(
                f"[PID {os.getpid()}] rank={global_rank} -> "
                f"removing connections to {cleanly_removed}",
                flush=True,
            )
            buffer.disconnect_ranks(cleanly_removed)
            remote_ranks.difference_update(cleanly_removed)
            import time
            time.sleep(5)

        active_ranks_list = plan.get_active_ranks()
        current_num_ranks = max(active_ranks_list) + 1
        current_num_experts = args.num_experts_per_rank * current_num_ranks

        num_stress_iters = 50
        for stress_iter in range(num_stress_iters):
            test_main(
                args.num_tokens,
                args.hidden_dim,
                current_num_experts,
                args.num_topk,
                global_rank,
                current_num_ranks,
                max_num_ranks,
                buffer,
                kineto=args.kineto,
            )
        print(
            f"[PID {os.getpid()}] rank={global_rank} -> "
            f"completed {num_stress_iters} stress iterations for phase {plan.get_phase()}",
            flush=True,
        )

        buffer.query_mask_buffer(mask_status)
        newly_failed_ranks = set()
        for r in range(current_num_ranks):
            if mask_status[r].item() != 0 and r in remote_ranks:
                newly_failed_ranks.add(r)

        if len(newly_failed_ranks) > 0:
            print(
                f"[PID {os.getpid()}] rank={global_rank} -> "
                f"detected failures: {newly_failed_ranks}",
                flush=True,
            )
            remote_ranks.difference_update(newly_failed_ranks)
            buffer.disconnect_ranks(list(newly_failed_ranks))
            import time
            time.sleep(5)

        print(
            f"[PID {os.getpid()}] rank={global_rank} -> "
            f"end phase {plan.get_phase()}",
            flush=True,
        )

        if not plan.next():
            break

    buffer.destroy()
    print(f"[PID {os.getpid()}] rank={global_rank} -> done", flush=True)


if __name__ == "__main__":
    main()
