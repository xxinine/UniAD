#!/usr/bin/env python3
"""Minimal multi-node NCCL sanity check.

Launched via torchrun across all nodes/GPUs. Each rank initializes the NCCL
process group, performs an all-reduce, and prints the result. If every rank
prints the expected world_size sum, cross-node NCCL works.
"""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    t = torch.ones(1, device=f"cuda:{local_rank}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    print(f"[rank {rank}/{world} local{local_rank} {os.uname().nodename}] "
          f"all_reduce -> {int(t.item())} (expected {world})", flush=True)

    dist.barrier()
    if rank == 0:
        ok = int(t.item()) == world
        print(f"NCCL MULTI-NODE TEST: {'PASS' if ok else 'FAIL'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
