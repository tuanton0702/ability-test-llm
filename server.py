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
import json
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def byte_reader(request):
    """
    Reads and processes the CSV data from the request.

    Args:
        request: Request object containing the CSV data.

    Returns:
        pd.DataFrame: Processed DataFrame containing the CSV data.
    """
    lst_data = request.csv_data.decode('utf-8').split('<s>')[1:]
    train_df = pd.DataFrame(lst_data, columns =['text'])
    return train_df

def make_datset(lst_data):
    """
    Creates a dataset from the provided list of data.

    Args:
        lst_data: List of data to create the dataset from.

    Returns:
        str: Timestamp string representation of the dataset creation.
    """
    timestamp = datetime.now().timestamp()
    dt_object = datetime.fromtimestamp(timestamp)
    timestamp_string_representation = dt_object.strftime("%Y-%m-%d-%H-%M%S")
    json_df = {
        "text" : []
    }
    for data in lst_data:
        data_str = f"<s>[INST] {data['instruction']} [/INST] {data['answer']} </s>"
        json_df["text"].append(data_str)
    data_df = pd.DataFrame(json_df)
    data_df.to_csv(f"./CreatedDataset/output_{timestamp_string_representation}.csv")
    return timestamp_string_representation

class ModelInit:
    def __init__(self):
        """
        Initializes the model and tokenizer for the Llama service.

        Args:
            None
        """
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

class LlamaService(proto.llama_pb2_grpc.LlamaServiceServicer):
    """
    Implements gRPC service methods for uploading multiple JSON files, fine-tuning a model, and making predictions.

    Args:
        model_init: ModelInit object for initializing the model and tokenizer.

    Returns:
        proto.llama_pb2.UploadResponse or proto.llama_pb2.FineTuneResponse or proto.llama_pb2.PredictionResponse: Response based on the gRPC service method called.
    """
    def __init__(self, model_init):
        self.model_init = model_init
    async def UploadMultipleJsonFiles(self, request, context):
        lst_json_data = []
        for byte in request.files:
            json_data = json.loads(byte.decode('utf-8'))
            lst_json_data.append(json_data)
        timestamp_string_representation = make_datset(lst_json_data)
        return proto.llama_pb2.UploadResponse(success=True, message=f"Files uploaded successfully and saved to ./CreatedDataset/output_{timestamp_string_representation}.csv")
    
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
    """
    Starts the gRPC server and handles incoming requests.

    Args:
        None
    """
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