import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model  # Only needed if using safetensors

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    device: str = "cpu"
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    print(f"prmpt tokens {prompt_tokens}")
    print(f"max_new_tokens {max_new_tokens}")
    print(f"eos_id {eos_id}")
    print(f"temperature {temperature}")
    print(f"device {device}")
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=device)
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        print(f"logits {logits}")
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    model_weights: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 4,
    temperature: float = 0.2,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available()
    device = f"cuda:{local_rank}" if use_cuda else "cpu"

    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")

    global print
    if rank != 0:
        print = lambda *_, **__: None

    if use_cuda:
        torch.cuda.set_device(local_rank)
    else:
        torch.set_default_device("cpu")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(2)

    with open(config) as f:
        args = ModelArgs()

    print(args)
    model = Transformer(args).to(device)
    load_model(model, model_weights)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # warm-up generate
    _ = tokenizer.encode("DeepSeek")
    _ = generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1., device)

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]

            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            messages.append({"role": "user", "content": prompt})
            prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            input_ids = tokenizer.encode(prompt_text)
            completion_tokens = generate(model, [input_ids], max_new_tokens, tokenizer.eos_token_id or -1, temperature, device)
            print(f"completion_tokens {completion_tokens}")
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size
        prompt_tokens = [
            tokenizer.encode(f"user: {prompt}\nassistant:")
            for prompt in prompts
        ]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id or -1, temperature, device)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    ckpt_path = "gpt2"  # Any available Hugging Face model like gpt2, distilgpt2, etc.
    model_weights = "/Users/shashi.b/Desktop/Synergy2/LLM/llm_scratch/checkpoints/model_epoch_50.safetensors"
    config = "/Users/shashi.b/Desktop/Synergy2/LLM/DeepSeek-V3/inference/configs/config_16B.json"  # Still needed for your own model
    input_file = "input.txt"
    interactive = True
    max_new_tokens = 5
    temperature = 0.7

    main(
        ckpt_path,
        model_weights,
        config,
        input_file,
        interactive,
        max_new_tokens,
        temperature
    )