#!/usr/bin/env python3
"""
Standalone reproducer for NIXL-EP RDMA bug with elastic reconnect.

Reproduces a silent RDMA delivery failure (NEVER_RECEIVED) when:
  1. RDMA transport is used (disable_ll_nvlink=True, no NVLink/P2P)
  2. Split SEND_ONLY dispatch/combine mode (return_recv_hook=True)
  3. Ranks run as independent processes (subprocess.Popen, not mp.spawn)
  4. Memory views are rebuilt via a second connect_ranks() call

The bug does NOT reproduce when any of the above conditions is removed:
  - NVLink transport: always works
  - Combined SEND+RECV mode (return_recv_hook=False): always works
  - torch.multiprocessing.spawn processes: always works
  - First connect_ranks (no prior views): always works

Launch via elastic_independent.py:

  RDMA (reproduces the bug):
    python3 elastic_independent.py \
        --disable-ll-nvlink \
        --num-tokens 256 --hidden-dim 2048 \
        --num-experts-per-rank 16 --num-topk 6 \
        --plan double_expansion.json --num-processes 8

  NVLink (should pass cleanly):
    python3 elastic_independent.py \
        --num-tokens 256 --hidden-dim 2048 \
        --num-experts-per-rank 16 --num-topk 6 \
        --plan double_expansion.json --num-processes 8
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


def split_send_recv_stress_test(
    buffer, num_tokens, hidden, num_experts, num_topk,
    rank, num_ranks, max_num_ranks, num_iters=50,
):
    """Run dispatch→combine using split SEND_ONLY mode (return_recv_hook=True).

    This exercises the same execution pattern as vLLM's MoE all2all:
      1. Dispatch SEND_ONLY kernel → deferred recv hook
      2. Expert computation (simulated)
      3. Combine SEND_ONLY kernel → deferred recv hook

    After num_iters iterations, checks the mask buffer. If any ranks
    are masked, RDMA delivery failed silently (the bug).

    Returns True if all ranks are healthy, False if failures detected.
    """
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    ).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(nixl_ep.topk_idx_t)
    topk_weights = torch.randn(
        (num_tokens, num_topk), dtype=torch.float32, device="cuda"
    ).abs()

    for _ in range(num_iters):
        # Dispatch: SEND_ONLY
        packed_recv_x, _, handle, _, dispatch_hook = buffer.dispatch(
            x, topk_idx, num_tokens, num_experts,
            use_fp8=False, async_finish=False, return_recv_hook=True,
        )
        dispatch_hook()

        # Simulated expert computation
        expert_out = (
            packed_recv_x.clone()
            if isinstance(packed_recv_x, torch.Tensor)
            else packed_recv_x[0].clone()
        )

        # Combine: SEND_ONLY
        out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        _, _, combine_hook = buffer.combine(
            expert_out, topk_idx, topk_weights, handle,
            async_finish=False, zero_copy=False,
            return_recv_hook=True, out=out,
        )
        combine_hook()

    # Check for RDMA failures (auto-masked ranks)
    mask = torch.zeros((max_num_ranks,), dtype=torch.int32, device="cuda")
    buffer.query_mask_buffer(mask)
    torch.cuda.synchronize()
    masked = [r for r in range(num_ranks) if mask[r].item() != 0]
    if masked:
        print(
            f"[PID {os.getpid()}] rank={rank} FAIL: "
            f"masked ranks (RDMA delivery failure): {masked}",
            flush=True,
        )
    return len(masked) == 0


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
            f"[PID {os.getpid()}] rank {global_rank}: "
            f"no phases found, exiting",
            flush=True,
        )
        return

    max_num_ranks = plan.get_max_rank() + 1
    print(
        f"[PID {os.getpid()}] rank={global_rank}, local_rank={local_rank}",
        flush=True,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    tcp_store = store_group.create_client_store(
        master_addr=server_addr, port=TCP_STORE_PORT,
    )

    num_rdma_bytes = nixl_ep.Buffer.get_rdma_size_hint(
        args.num_tokens, args.hidden_dim,
        max_num_ranks, args.num_experts_per_rank * max_num_ranks,
    )

    buffer = nixl_ep.Buffer(
        rank=global_rank,
        disable_ll_nvlink=args.disable_ll_nvlink,
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

    while True:
        phase = plan.get_phase()
        added_ranks = plan.get_new_ranks()
        cleanly_removed = plan.get_removed_ranks()

        if global_rank in cleanly_removed:
            rank_client.release_rank(user_context=phase)
            break

        if added_ranks:
            print(f"[PID {os.getpid()}] rank={global_rank} phase {phase}: "
                  f"connecting {added_ranks}", flush=True)
            buffer.connect_ranks(added_ranks)
            remote_ranks.update(added_ranks)

        if cleanly_removed:
            buffer.disconnect_ranks(cleanly_removed)
            remote_ranks.difference_update(cleanly_removed)
            import time
            time.sleep(5)

        active_ranks = plan.get_active_ranks()
        num_ranks = max(active_ranks) + 1
        num_experts = args.num_experts_per_rank * num_ranks

        ok = split_send_recv_stress_test(
            buffer, args.num_tokens, args.hidden_dim,
            num_experts, args.num_topk,
            global_rank, num_ranks, max_num_ranks,
            num_iters=50,
        )
        print(f"[PID {os.getpid()}] rank={global_rank} phase {phase}: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

        # Clean up any failed ranks
        buffer.query_mask_buffer(mask_status)
        failed = {r for r in range(num_ranks)
                  if mask_status[r].item() != 0 and r in remote_ranks}
        if failed:
            remote_ranks.difference_update(failed)
            buffer.disconnect_ranks(list(failed))
            import time
            time.sleep(5)

        if not plan.next():
            break

    buffer.destroy()
    print(f"[PID {os.getpid()}] rank={global_rank} -> done", flush=True)


if __name__ == "__main__":
    main()
