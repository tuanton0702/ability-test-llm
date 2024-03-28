import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import logging


        


# def PredictLLM(prompt):
#     # The model that you want to train from the Hugging Face hub
#     model_name = "NousResearch/Llama-2-7b-chat-hf"

#     # Fine-tuned model name
#     new_model = "./model-checkpoint/llama-2-7b-chat-guanaco"
#     # Load the entire model on the GPU 0
#     # device_map = {"": }
#     # Reload model in FP16 and merge it with LoRA weights
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         low_cpu_mem_usage=True,
#         return_dict=True,
#         torch_dtype=torch.float16,
#         # device_map=device_map,
#     )
#     print("LOADED BASED MODEL")
#     model = PeftModel.from_pretrained(base_model, new_model)
#     print("DONE COMBINE MODEL")

#     model = model.merge_and_unload()
#     print("INIT MODEL")
#     # Reload tokenizer to save it
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
#     print("DONE RELOAD TOKENIZER")

#     # # Ignore warnings
#     # logging.set_verbosity(logging.CRITICAL)

#     # Run text generation pipeline with our next model
#     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=36)
#     result = pipe(f"<s>[INST] {prompt} [/INST]")
#     return result[0]['generated_text']
def PredictLLM(request, model, tokenizer):
    """
    Generates text based on the provided prompt using the specified model and tokenizer.

    Args:
        request: Request object containing the prompt.
        model: Pretrained model for text generation.
        tokenizer: Tokenizer for processing text inputs.

    Returns:
        str: Generated text based on the prompt.
    """
    prompt = request.prompt
    # Run text generation pipeline with our next model
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=36)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']
