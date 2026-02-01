from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

def estimate_model_size(hf_repo_id: str) -> int:
    """
    Estimate model size in bytes assuming bf16 dtype.
    
    Uses HuggingFace Accelerate to load the model onto meta device
    (no actual memory allocation) and calculates the size.
    
    Args:
        hf_repo_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        
    Returns:
        Size in bytes for bf16 weights
    """
    config = AutoConfig.from_pretrained(hf_repo_id, trust_remote_code=True)
    
    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    
    # Calculate size: each parameter in bf16 is 2 bytes
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * 2  # bf16 = 2 bytes per element
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * 2
    
    return param_size + buffer_size