import json
import argparse
from typing import Dict, Optional
import math

class LLMMemoryCalculator:
    def __init__(self, config: Dict):
        """Initialize calculator with model configuration."""
        self.config = config
        self.dtype_sizes = {
            'float32': 4,
            'float16': 2,
            'bfloat16': 2,
            'int8': 1,
            'fp8': 1
        }
        
    def get_param_count(self) -> int:
        """Calculate total number of parameters in the model."""
        hidden_size = self.config.get('hidden_size', self.config.get('d_model'))
        num_layers = self.config.get('num_hidden_layers', self.config.get('n_layer'))
        vocab_size = self.config['vocab_size']
        num_attention_heads = self.config.get('num_attention_heads', self.config.get('n_head'))
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size  # Input embeddings
        position_embedding_params = self.config.get('max_position_embeddings', 0) * hidden_size
        
        # Attention parameters per layer
        attn_params_per_layer = 4 * hidden_size * hidden_size  # Q, K, V, and output projections
        
        # FFN parameters per layer
        ffn_hidden_size = self.config.get('intermediate_size', 4 * hidden_size)
        ffn_params_per_layer = 2 * hidden_size * ffn_hidden_size  # Two linear transformations
        
        # Layer norm parameters
        layer_norm_params = 4 * hidden_size * num_layers  # Two layer norms per layer
        
        # Total parameters
        total_params = (
            embedding_params +
            position_embedding_params +
            num_layers * (attn_params_per_layer + ffn_params_per_layer) +
            layer_norm_params
        )
        
        return total_params
    
    def estimate_training_memory(self, batch_size: int, seq_length: int, dtype: str = 'float32',
                               gradient_checkpointing: bool = False) -> Dict[str, float]:
        """Estimate memory requirements for training."""
        param_count = self.get_param_count()
        dtype_size = self.dtype_sizes[dtype]
        
        # Model parameters memory
        params_memory = param_count * dtype_size
        
        # Optimizer states (Adam has 2 additional states per parameter)
        optimizer_memory = param_count * dtype_size * 2
        
        # Gradients memory
        gradients_memory = param_count * dtype_size
        
        # Activations memory
        if gradient_checkpointing:
            activation_factor = 1.5  # Approximate reduction factor
        else:
            activation_factor = 3
        
        activations_memory = (
            batch_size * seq_length * self.config['hidden_size'] * 
            self.config['num_hidden_layers'] * dtype_size * activation_factor
        )
        
        # Total memory
        total_memory = params_memory + optimizer_memory + gradients_memory + activations_memory
        
        return {
            'parameters_memory_gb': params_memory / (1024**3),
            'optimizer_memory_gb': optimizer_memory / (1024**3),
            'gradients_memory_gb': gradients_memory / (1024**3),
            'activations_memory_gb': activations_memory / (1024**3),
            'total_memory_gb': total_memory / (1024**3)
        }
    
    def estimate_finetuning_memory(self, batch_size: int, seq_length: int, 
                                 dtype: str = 'float32', lora_rank: Optional[int] = None) -> Dict[str, float]:
        """Estimate memory requirements for finetuning, optionally with LoRA."""
        if lora_rank is None:
            # Traditional finetuning - similar to training but potentially smaller batch size
            return self.estimate_training_memory(batch_size, seq_length, dtype)
        
        # LoRA memory estimation
        param_count = self.get_param_count()
        dtype_size = self.dtype_sizes[dtype]
        hidden_size = self.config['hidden_size']
        
        # LoRA parameters (rank * (input_dim + output_dim) for each weight matrix)
        lora_params = lora_rank * (2 * hidden_size) * self.config['num_hidden_layers'] * 2
        
        # Base model parameters (frozen, so no optimizer states needed)
        base_model_memory = param_count * dtype_size
        
        # LoRA parameters memory including optimizer states
        lora_memory = lora_params * dtype_size * 3  # Parameters + 2 optimizer states
        
        # Activations memory (similar to inference but with gradients for LoRA params)
        activations_memory = (
            batch_size * seq_length * hidden_size * 
            self.config['num_hidden_layers'] * dtype_size
        )
        
        total_memory = base_model_memory + lora_memory + activations_memory
        
        return {
            'base_model_memory_gb': base_model_memory / (1024**3),
            'lora_memory_gb': lora_memory / (1024**3),
            'activations_memory_gb': activations_memory / (1024**3),
            'total_memory_gb': total_memory / (1024**3)
        }
    
    def estimate_inference_memory(self, batch_size: int, seq_length: int, 
                                dtype: str = 'float32', kv_cache: bool = True) -> Dict[str, float]:
        """Estimate memory requirements for inference."""
        param_count = self.get_param_count()
        dtype_size = self.dtype_sizes[dtype]
        
        # Model parameters memory
        params_memory = param_count * dtype_size
        
        # Activations memory
        activations_memory = (
            batch_size * seq_length * self.config['hidden_size'] * dtype_size
        )
        
        # KV cache memory if enabled
        kv_cache_memory = 0
        if kv_cache:
            kv_cache_memory = (
                2 * batch_size * seq_length * self.config['hidden_size'] * 
                self.config['num_hidden_layers'] * dtype_size
            )
        
        total_memory = params_memory + activations_memory + kv_cache_memory
        
        return {
            'parameters_memory_gb': params_memory / (1024**3),
            'activations_memory_gb': activations_memory / (1024**3),
            'kv_cache_memory_gb': kv_cache_memory / (1024**3),
            'total_memory_gb': total_memory / (1024**3)
        }

def main():
    parser = argparse.ArgumentParser(description='Calculate LLM memory requirements')
    parser.add_argument('config_file', type=str, help='Path to model config.json file')
    parser.add_argument('--mode', choices=['train', 'finetune', 'inference'], required=True,
                       help='Operation mode')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq_length', type=int, required=True, help='Sequence length')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16', 'int8', 'fp8'],
                       default='float32', help='Data type')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing for training')
    parser.add_argument('--lora_rank', type=int, help='LoRA rank for finetuning')
    parser.add_argument('--kv_cache', action='store_true', help='Enable KV cache for inference')
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    calculator = LLMMemoryCalculator(config)
    
    if args.mode == 'train':
        results = calculator.estimate_training_memory(
            args.batch_size, args.seq_length, args.dtype, args.gradient_checkpointing
        )
    elif args.mode == 'finetune':
        results = calculator.estimate_finetuning_memory(
            args.batch_size, args.seq_length, args.dtype, args.lora_rank
        )
    else:  # inference
        results = calculator.estimate_inference_memory(
            args.batch_size, args.seq_length, args.dtype, args.kv_cache
        )
    
    print("\nEstimated memory requirements:")
    for key, value in results.items():
        print(f"{key}: {value:.2f} GB")

if __name__ == "__main__":
    main()
