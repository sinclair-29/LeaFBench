import torch
from benchmark.model_interface import ModelInterface

class InstructModel(ModelInterface):
    """
    Base model class that inherits from ModelInterface.
    This class can be used to implement common functionality for all models.
    """
    def __init__(self, config, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)

    def render_prompts(self, prompts, tokenizer):
        """Apply the same chat formatting used by generation and fingerprinting."""
        system_prompt = self.params.get('system_prompt', None)
        rendered_prompts = []

        for prompt in prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    if system_prompt is not None:
                        formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                    else:
                        formatted_prompt = f"User: {prompt}\n\nAssistant:"
            elif system_prompt is not None:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\n\nAssistant:"

            # Chat templates such as Llama-2's already start with a BOS token.
            # The tokenizer adds that token again when the rendered text is encoded.
            if tokenizer.bos_token and formatted_prompt.startswith(tokenizer.bos_token):
                formatted_prompt = formatted_prompt[len(tokenizer.bos_token):]
            rendered_prompts.append(formatted_prompt)

        return rendered_prompts
    
    def generate(self, prompts, **kwargs):
        """
        Generate text for given prompts using apply_chat_template.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional generation parameters
        
        Returns:
            list: List of generated text strings
        """
        model, tokenizer = self.load_model()
        
        # Default generation parameters
        generation_params = {
            'max_new_tokens': self.params.get('max_new_tokens', 512),
            'temperature': self.params.get('temperature', 0.7),
            'do_sample': self.params.get('do_sample', True),
            'top_p': self.params.get('top_p', 0.9),
            'top_k': self.params.get('top_k', 50),
            'pad_token_id': tokenizer.pad_token_id,
        }

        rendered_prompts = self.render_prompts(prompts, tokenizer)
        
        # Tokenize input prompts
        inputs = tokenizer(
            rendered_prompts,
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=self.params.get('max_input_length', 512),
            padding_side='left'
        )

        # Move inputs to the same device as model
        if self.accelerator is not None:
            # When using accelerator, it handles device placement
            device = self.accelerator.device
        else:
            device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        # Special handling for Gemma-2 models to avoid cache device mismatch
        model_name_lower = model.__class__.__name__.lower()
        config_name_lower = getattr(model.config, 'model_family', '').lower()
        if "gemma" in model_name_lower or "gemma" in config_name_lower:
            # For Gemma models, disable cache to avoid device mismatch issues
            generation_params['use_cache'] = False
        
        # with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params
        )
        
        # Decode generated text
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_logits(self, prompts, **kwargs):
        """
        Generate logits for given prompts.
        
        Args:
            prompts (list): List of input prompt strings
            **kwargs: Additional parameters
        
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, sequence_length, vocab_size)
        """
        # if self.model is None:
        model, tokenizer = self.load_model()
        
        # Tokenize input prompts
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=kwargs.get('max_input_length', 512)
        )
        
        # Move inputs to the same device as model
        if self.accelerator is not None:
            # When using accelerator, it handles device placement
            device = self.accelerator.device
        else:
            device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get logits from model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Apply temperature if specified
        temperature = kwargs.get('temperature', 1.0)
        if temperature != 1.0:
            logits = logits / temperature
        
        return logits
