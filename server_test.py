import asyncio
import logging
from grpc import aio
import proto.llama_pb2
import proto.llama_pb2_grpc
from concurrent import futures
import finetune_llm
import predict_llm
import csv
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def byte_reader(request):
    # print(request.decode('utf-8'))
    lst_data = request.csv_data.decode('utf-8').split('<s>')[1:]
    train_df = pd.DataFrame(lst_data, columns =['text'])
    return train_df

# class ModelInit:
    # def __init__(self):
    #     self.model = None
    #     self.tokenizer = None
    #     self.init_model()

    def init_model(self):
        # The model that you want to train from the Hugging Face hub
        model_name = "NousResearch/Llama-2-7b-chat-hf"
        # Fine-tuned model name
        new_model = "./model-checkpoint/llama-2-7b-chat-guanaco"
        
        # Load the entire model on the GPU 0
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, new_model)
        self.model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

        logging.info("Model initialized successfully.")
class ModelInit:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.init_model()

    def init_model(self):
        model_name = "NousResearch/Llama-2-7b-chat-hf"
        new_model = "./model-checkpoint/llama-2-7b-chat-guanaco"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16
        )
        print("done load based")
        model = PeftModel.from_pretrained(base_model, new_model)
        self.model = model.merge_and_unload()
        print("done merged")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        print('Done load tokenizer')
        print("Model initialized successfully.")
        # logging.info("Model initialized successfully.")
class LlamaService(proto.llama_pb2_grpc.LlamaServiceServicer):
    def __init__(self, model_init):
        self.model_init = model_init
    async def FineTune(self, request, context):
        train_df = byte_reader(request)
        result = finetune_llm.FineTuneLLm(train_df)
        return proto.llama_pb2.FineTuneResponse(status=result["status"], logs=result['logs'])

    async def Predict(self, request, context):
        # Implement prediction logic here
        result = predict_llm.PredictLLM(request, self.model_init.model, self.model_init.tokenizer)
        # responses = "tesssssssssst"
        return proto.llama_pb2.PredictionResponse(response=result)

async def serve():
    server = aio.server()
    listen_addr = "[::]:50051"
    model_init = ModelInit()
    llama_service = LlamaService(model_init)  # Passing model_init to LlamaService
    proto.llama_pb2_grpc.add_LlamaServiceServicer_to_server(llama_service, server)
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    print("gRPC server started")
    asyncio.run(serve())