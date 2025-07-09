import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)

    def forward(self, x):
        out = self.linear(x)

        # All-gather only if distributed
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Create tensor list to gather into
            gathered = [torch.zeros_like(out, dtype=torch.float32) for _ in range(world_size)]
            # Clone/detach/convert to normal tensor to avoid inplace error
            safe_out = out.clone().detach().to(dtype=torch.float32)
            dist.all_gather(gathered, safe_out)
            # Merge the gathered tensors
            out = torch.cat(gathered, dim=0).to(dtype=out.dtype)

        return out

def setup():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="gloo")

    if rank != 0:
        # Silence prints from non-zero ranks
        builtins_print = print
        def no_print(*args, **kwargs): pass
        globals()["print"] = no_print

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

if __name__ == "__main__":
    setup()

    start = time.time()
    model = ToyModel()
    x = torch.randn(2, 128)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Time: {time.time() - start:.3f} sec")

    if dist.is_initialized():
        dist.destroy_process_group()