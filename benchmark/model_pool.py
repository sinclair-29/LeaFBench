import logging
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from peft import AutoPeftModelForCausalLM
from gptqmodel.nn_modules.qlinear.torch import BaseQuantLinear, TorchQuantLinear
from gptqmodel import BACKEND, GPTQModel
from accelerate import disk_offload, dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import gc


class TokenEmbeddingOverrideError(RuntimeError):
    """Raised when a temporary token embedding cannot be applied safely."""


class ModelPool:
    def __init__(self, accelerator=None, max_loaded_models=1, offload_path=None, fingerprint_type="black-box", fingerprint_method=None):
        # self.models = {}  # {model_name: model_instance}
        self.tokenizers = OrderedDict()  # {model_name: tokenizer_instance}
        self.model_paths = OrderedDict()  # {model_name: model_path}
        self.tokenizer_paths = OrderedDict()  # adapters may not bundle a tokenizer
        self.accelerator = accelerator
        self.current_loaded_models = OrderedDict()  # {model_name: model_instance}
        self.max_loaded_models = max_loaded_models
        self.offload_path = offload_path
        if self.offload_path:
            os.makedirs(self.offload_path, exist_ok=True)
        self.fingerprint_type = fingerprint_type
        self.fingerprint_method = fingerprint_method
        self.backend = BACKEND("torch")  # Set backend for gptqmodel
        self.token_embedding_overrides = {}
        self.token_embedding_override_states = {}

    def register_model(self, model_name, model_path, tokenizer_path=None):
        """
        Register the model path, but do not load the model.
        """
        self.model_paths[model_name] = model_path
        self.tokenizer_paths[model_name] = tokenizer_path or model_path
        # with init_empty_weights():
        # self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_path) if model_path else None
        # tokenizer = AutoTokenizer.from_pretrained(model_path) if model_path else None
        # Set pad token if it doesn't exist
        # if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token

        # self.tokenizers[model_name] = tokenizer

    def get_tokenizer(self, model_name):
        """
        Get the tokenizer for the specified model, load it on demand and cache it.
        """
        tokenizer_path = self.tokenizer_paths[model_name]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path else None
        override = self.token_embedding_overrides.get(model_name)
        if override is not None:
            added = tokenizer.add_tokens([override["token"]])
            token_ids = tokenizer.encode(override["token"], add_special_tokens=False)
            if added != 1 or token_ids != [override["token_id"]]:
                raise TokenEmbeddingOverrideError(
                    f"Model {model_name} cannot represent copyright token "
                    f"{override['token']!r} as one newly added token."
                )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def register_token_embedding_override(self, model_name, token, embedding):
        """Register a temporary token embedding used by the next generation call."""
        if model_name not in self.model_paths:
            raise TokenEmbeddingOverrideError(f"Model {model_name} is not registered.")
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 1:
            raise TokenEmbeddingOverrideError("Token embedding override must be a 1D tensor.")

        if model_name in self.token_embedding_overrides:
            self.clear_token_embedding_override(model_name)

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_paths[model_name])
            added = tokenizer.add_tokens([token])
            token_ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception as exc:
            raise TokenEmbeddingOverrideError(
                f"Could not prepare copyright token for model {model_name}: {exc}"
            ) from exc
        if added != 1 or len(token_ids) != 1:
            raise TokenEmbeddingOverrideError(
                f"Model {model_name} cannot represent copyright token {token!r} "
                "as one newly added token."
            )

        self.token_embedding_overrides[model_name] = {
            "token": token,
            "token_id": token_ids[0],
            "vocab_size": len(tokenizer),
            "embedding": embedding.detach().cpu(),
        }

    def _apply_token_embedding_override(self, model_name, model):
        override = self.token_embedding_overrides.get(model_name)
        if override is None or model_name in self.token_embedding_override_states:
            return

        if not hasattr(model, "get_input_embeddings") or not hasattr(model, "resize_token_embeddings"):
            raise TokenEmbeddingOverrideError(
                f"Model {model_name} does not expose a resizable input embedding layer."
            )

        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            raise TokenEmbeddingOverrideError(
                f"Model {model_name} does not expose input embedding weights."
            )

        original_vocab_size = input_embeddings.weight.shape[0]
        token_id = override["token_id"]
        original_row = None
        if token_id < original_vocab_size:
            original_row = input_embeddings.weight[token_id].detach().cpu().clone()

        try:
            if token_id >= original_vocab_size:
                model.resize_token_embeddings(override["vocab_size"])
                input_embeddings = model.get_input_embeddings()

            if token_id >= input_embeddings.weight.shape[0]:
                raise TokenEmbeddingOverrideError(
                    f"Model {model_name} did not resize to token id {token_id}."
                )
            if input_embeddings.weight.shape[1] != override["embedding"].numel():
                raise TokenEmbeddingOverrideError(
                    f"Model {model_name} embedding width {input_embeddings.weight.shape[1]} "
                    f"does not match candidate width {override['embedding'].numel()}."
                )

            with torch.no_grad():
                input_embeddings.weight[token_id].copy_(
                    override["embedding"].to(
                        device=input_embeddings.weight.device,
                        dtype=input_embeddings.weight.dtype,
                    )
                )
        except Exception as exc:
            if model.get_input_embeddings().weight.shape[0] != original_vocab_size:
                model.resize_token_embeddings(original_vocab_size)
            if isinstance(exc, TokenEmbeddingOverrideError):
                raise
            raise TokenEmbeddingOverrideError(
                f"Could not apply copyright embedding to model {model_name}: {exc}"
            ) from exc

        self.token_embedding_override_states[model_name] = {
            "original_vocab_size": original_vocab_size,
            "original_row": original_row,
            "token_id": token_id,
        }

    def clear_token_embedding_override(self, model_name):
        """Restore the original in-memory model after a temporary override."""
        state = self.token_embedding_override_states.pop(model_name, None)
        model = self.current_loaded_models.get(model_name)

        if state is not None and model is not None:
            try:
                current_vocab_size = model.get_input_embeddings().weight.shape[0]
                if current_vocab_size != state["original_vocab_size"]:
                    model.resize_token_embeddings(state["original_vocab_size"])
                elif state["original_row"] is not None:
                    input_embeddings = model.get_input_embeddings()
                    with torch.no_grad():
                        input_embeddings.weight[state["token_id"]].copy_(
                            state["original_row"].to(
                                device=input_embeddings.weight.device,
                                dtype=input_embeddings.weight.dtype,
                            )
                        )
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to restore temporary embedding for %s; unloading the model.",
                    model_name,
                    exc_info=True,
                )
                self._completely_unload_model(model, model_name, logger)
                del self.current_loaded_models[model_name]

        self.token_embedding_overrides.pop(model_name, None)

    def get_model(self, model_name, type=None):
        """
        Get the model object, load it to the specified device (default GPU) on demand.
        Uses accelerator for proper multi-GPU management if available.
        """
        logger = logging.getLogger(__name__)
        if self.model_paths[model_name] is None:
            raise ValueError(f"Model {model_name} not registered.")
        else:
            if model_name not in self.current_loaded_models.keys():
                if len(self.current_loaded_models) >= self.max_loaded_models:
                    # Unload the least recently used model
                    oldest_model_name = next(iter(self.current_loaded_models))
                    logger.info(f"Unloading {oldest_model_name} to make room for {model_name}")
                    
                    # Get the model to be unloaded
                    model_to_unload = self.current_loaded_models[oldest_model_name]
                    
                    # Completely unload the model from memory
                    self._completely_unload_model(model_to_unload, oldest_model_name, logger)
                    
                    # Remove from the loaded models dictionary
                    del self.current_loaded_models[oldest_model_name]
                    self.token_embedding_override_states.pop(oldest_model_name, None)
                    
                    logger.info(f"Successfully unloaded {oldest_model_name} and freed all memory")
        
                # Load the new model
                if type == "adapter":
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        self.model_paths[model_name], 
                        device_map="balanced", 
                        torch_dtype=torch.float16
                    )
                    model = model.merge_and_unload()
                    for param in model.parameters():
                        param.requires_grad = True
                elif type == "quantization":
                    # Only dequantize if needed for white-box fingerprinting
                    if self.fingerprint_type == "white-box":
                        model = GPTQModel.load(
                            self.model_paths[model_name], 
                            device_map={"": "cpu"},  # Load to single device first
                            torch_dtype=torch.float16,
                            backend=self.backend
                        )
                        model = dequantize_model(model, torch.float16)
                        no_split_module_classes = model._no_split_modules
                        max_memory = get_balanced_memory(
                            model,
                            max_memory=None,
                            no_split_module_classes=no_split_module_classes,
                            dtype=torch.float16
                        )

                        device_map = infer_auto_device_map(
                            model,
                            max_memory=max_memory,
                            no_split_module_classes=no_split_module_classes,
                            dtype=torch.float16
                        )
                        model = dispatch_model(model, device_map=device_map)
                    else:
                        model = GPTQModel.load(
                            self.model_paths[model_name], 
                            device_map="auto", 
                            torch_dtype=torch.float16,
                            backend=self.backend
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_paths[model_name], 
                        device_map="balanced", 
                        torch_dtype=torch.float16
                    )
                self.current_loaded_models[model_name] = model
                logger.info(f"{len(self.current_loaded_models)} models loaded in the pool")
                
            else:
                # Move to end (most recently used)
                model = self.current_loaded_models.pop(model_name)
                self.current_loaded_models[model_name] = model
                
            model = self.current_loaded_models[model_name]
            self._apply_token_embedding_override(model_name, model)
            return model

    def list_models(self):
        """
        List all registered models.
        """
        return list(self.model_paths.keys())
    
    def _completely_unload_model(self, model, model_name, logger):
        """
        Completely unload a model from memory, including GPU and CPU memory.
        """
        
        try:
            # Step 1: Clear all hooks that might keep references
            if hasattr(model, '_forward_hooks'):
                model._forward_hooks.clear()
            if hasattr(model, '_backward_hooks'):
                model._backward_hooks.clear()
            if hasattr(model, '_forward_pre_hooks'):
                model._forward_pre_hooks.clear()
            
            # Step 2: Handle models with device_map (distributed across devices)
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Model {model_name} has device_map, performing distributed cleanup")
                
                # For models with device_map, we need to clear each device
                for module_name, device in model.hf_device_map.items():
                    try:
                        module = model
                        for attr in module_name.split('.'):
                            if attr:
                                module = getattr(module, attr)
                        
                        # Move module to CPU and clear its data
                        if hasattr(module, 'weight') and module.weight is not None:
                            module.weight.data = module.weight.data.cpu()
                            del module.weight
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data.cpu()
                            del module.bias
                    except Exception as e:
                        logger.warning(f"Error clearing module {module_name}: {e}")
                
                # Clear the device map
                if hasattr(model, 'hf_device_map'):
                    del model.hf_device_map
            
            # Step 3: Move all parameters and buffers to CPU and delete them
            params_to_delete = []
            buffers_to_delete = []
            
            for name, param in model.named_parameters():
                if param.device.type == 'cuda':
                    param.data = param.data.cpu()
                params_to_delete.append((name, param))
            
            for name, buffer in model.named_buffers():
                if buffer.device.type == 'cuda':
                    buffer.data = buffer.data.cpu()
                buffers_to_delete.append((name, buffer))
            
            # Clear parameter and buffer references
            for name, param in params_to_delete:
                try:
                    delattr(model, name.split('.')[-1]) if '.' not in name else None
                except:
                    pass
                del param
                
            for name, buffer in buffers_to_delete:
                try:
                    delattr(model, name.split('.')[-1]) if '.' not in name else None
                except:
                    pass
                del buffer
            
            # Step 4: Clear model's internal state
            if hasattr(model, 'config'):
                del model.config
            if hasattr(model, 'generation_config'):
                del model.generation_config
            
            # Step 5: Force model to CPU (if not already done)
            try:
                model.cpu()
            except:
                pass
            
            # Step 6: Multiple rounds of cleanup
            for i in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"Model {model_name} completely unloaded from memory")
            
        except Exception as e:
            logger.error(f"Error during complete model unloading: {e}")
            # Fallback: still try basic cleanup
            try:
                model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

def dequantize_model(model: PreTrainedModel, dtype: torch.dtype):
    modules_to_replace = []
    # First, collect all quantized modules that need to be replaced
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear):
            if not isinstance(module, TorchQuantLinear):
                 raise ValueError(
                    "Only models using TorchQuantLinear are supported for dequantization."
                    "Please load the model with backend='torch'."
                )
            modules_to_replace.append((name, module))

    for name, module in modules_to_replace:
        # Create a new, non-quantized nn.Linear module
        # Since the model is on CPU, the new module is also created on CPU
        device = torch.device("cpu")
        has_bias = module.bias is not None and module.bias.numel() > 0

        new_module = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            device=device,
            dtype=dtype
        )

        # Dequantize the weights and assign them to the new module
        # The .T is used depending on the output shape of `dequantize_weight`.
        # Assume `dequantize_weight` outputs (in, out), after transpose it becomes (out, in), matching nn.Linear.weight shape.
        dequantized_weight = module.dequantize_weight().T
        new_module.weight = nn.Parameter(dequantized_weight.detach())

        if has_bias:
            new_module.bias = nn.Parameter(module.bias.detach())

        # Use a more robust way to locate and replace the submodule in the parent module
        parent_name, module_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, module_name, new_module)

    # Clean up quantization information in the model config
    if hasattr(model.config, 'quantization_config'):
        del model.config.quantization_config
        model.config.is_quantized = False # Add a flag

    return model.model
