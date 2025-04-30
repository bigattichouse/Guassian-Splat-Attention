# hsa/utils/model_loader.py
import os
import torch
import json
import logging
from typing import Dict, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_checkpoint(model_dir: str) -> str:
    """
    Find the latest checkpoint file in the model directory.
    
    Args:
        model_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint
    """
    # First look for best_model.pt
    best_model_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        return best_model_path
    
    # Otherwise, find checkpoint with highest epoch
    checkpoints = []
    for filename in os.listdir(model_dir):
        if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
            try:
                epoch = int(filename.split("_")[-1].split(".")[0])
                checkpoints.append((epoch, os.path.join(model_dir, filename)))
            except ValueError:
                continue
    
    if checkpoints:
        # Return checkpoint with highest epoch
        return sorted(checkpoints, key=lambda x: x[0], reverse=True)[0][1]
    
    # If no checkpoints found, return empty string
    return ""

def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a model configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded configuration dictionary or None if file not found
    """
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None

def load_hf_model_and_tokenizer(model_dir: str, device: str = "cpu") -> Tuple[Any, Any]:
    """
    Load a model and tokenizer using Hugging Face Transformers.
    
    Args:
        model_dir: Directory containing the model
        device: Device to load the model to
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Try to load the model
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        logger.info(f"Loaded model from {model_dir} using Hugging Face Transformers")
        
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info(f"Loaded tokenizer from {model_dir}")
        
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except ImportError:
        logger.warning("Hugging Face Transformers not available")
        raise
    except Exception as e:
        logger.error(f"Failed to load model using Hugging Face Transformers: {e}")
        raise

def create_minimal_tokenizer() -> Any:
    """
    Create a minimal tokenizer for use when Hugging Face is not available.
    
    Returns:
        A minimal tokenizer object
    """
    class MinimalTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 50256
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
        
        def encode(self, text, return_tensors=None):
            # Very simple encoding, just for demonstration
            tokens = [ord(c) % 50000 for c in text]
            if return_tensors == "pt":
                import torch
                return {"input_ids": torch.tensor([tokens])}
            return tokens
        
        def decode(self, token_ids, skip_special_tokens=False):
            # Simple decoding, just for demonstration
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            return "".join([chr(t % 128) for t in token_ids])
            
        def __call__(self, text, return_tensors=None):
            return self.encode(text, return_tensors=return_tensors)
    
    logger.info("Created minimal tokenizer (Hugging Face not available)")
    return MinimalTokenizer()
