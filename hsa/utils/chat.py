# hsa/utils/chat.py
import torch
import readline  # For better input handling
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatModel:
    """
    Generic chat interface for language models.
    
    This class handles the interaction with the model for chat.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: Optional[int] = None
    ):
        """
        Initialize the chat model.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run the model on ('cpu' or 'cuda')
            max_new_tokens: Maximum number of tokens to generate in response
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            seed: Random seed for reproducibility
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            import numpy as np
            np.random.seed(seed)
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response to the user's prompt.
        
        Args:
            prompt: The input text from the user
            
        Returns:
            Generated response text
        """
        try:
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response (remove the input prompt)
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"[Error generating response: {e}]"
    
    def chat(self):
        """Start an interactive chat session with the model."""
        print("\n" + "="*50)
        print("Starting chat session. Type 'exit' or 'quit' to end the chat.")
        print("="*50 + "\n")
        
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ")
            except EOFError:
                print("\nExiting chat.")
                break
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nEnding chat session. Goodbye!")
                break
                
            # Generate response
            print("\nGenerating response...")
            response = self.generate_response(user_input)
            print(f"\nModel: {response}")


class HSAChatModel(ChatModel):
    """
    Chat interface for HSA models.
    
    Extends the generic ChatModel with HSA-specific functionality.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        hsa: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the HSA chat model.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            hsa: Optional HSA instance
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(model, tokenizer, **kwargs)
        self.hsa = hsa
    
    def adapt_attention(self, tokens: torch.Tensor) -> None:
        """
        Adapt HSA attention based on input tokens.
        
        Args:
            tokens: Input token IDs
        """
        if self.hsa is None:
            logger.warning("HSA is not available, cannot adapt attention")
            return
        
        try:
            # Get token embeddings
            with torch.no_grad():
                token_embeds = self.model.transformer.word_embeddings(tokens).cpu().numpy()
            
            # Adapt HSA
            self.hsa.adapt(token_embeds)
            logger.info("Adapted HSA attention")
        except Exception as e:
            logger.error(f"Error adapting HSA attention: {e}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response with HSA adaptation.
        
        Args:
            prompt: The input text from the user
            
        Returns:
            Generated response text
        """
        try:
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Adapt attention if HSA is available
            if self.hsa is not None:
                self.adapt_attention(inputs.input_ids[0])
            
            # Generate response (using parent method)
            return super().generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating response with HSA: {e}")
            return f"[Error generating response: {e}]"
