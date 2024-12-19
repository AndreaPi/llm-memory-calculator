# llm-memory-calculator
A *very* simple calculator to estimate the GPU memory usage of a 🤗 [Transformers](https://github.com/huggingface/transformers) model, according to its configuration file. The script estimates memory requirements respectively for training, finetuning or inference.

> ⚠️ **nearly untested**: use at your own risk. PRs welcome!


Basic usage:

> python llm_memory_calculator.py config.json --mode train --batch_size 32 --seq_length 512

Features:

- Supports training, finetuning, and inference calculations
- Accounts for different precisions (`float32`, `float16`, `bfloat16`, `int8`, `fp8`)
- Includes memory optimizations like gradient checkpointing
- Supports LoRA for finetuning estimates
- Considers KV cache for inference

Advanced usage:
```
# Training with gradient checkpointing
python llm_memory_calculator.py config.json --mode train --batch_size 32 --seq_length 512 --gradient_checkpointing --dtype float16

# Finetuning with LoRA
python llm_memory_calculator.py config.json --mode finetune --batch_size 16 --seq_length 512 --lora_rank 8 --dtype float16

# Inference with KV cache
python llm_memory_calculator.py config.json --mode inference --batch_size 1 --seq_length 1024 --kv_cache --dtype float16
```

**NOTE**: the calculator just returns an estimate. The actual memory usage may vary depending on implementation details of the framework, memory fragmentation, system overhead, specific optimizations used, etc.