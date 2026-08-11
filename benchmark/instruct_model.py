from benchmark.base_models import BaseModel

class InstructModel(BaseModel):
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
    
    # Generation and logits are inherited from BaseModel so that raw and
    # instruction-tuned models share one validated generation-parameter path.
