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
from datetime import datetime

def byte_reader(request):
    lst_data = request.csv_data.decode('utf-8').split('<s>')[1:]
    train_df = pd.DataFrame(lst_data, columns =['text'])
    return train_df

def make_datset(lst_data):
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
    def __init__(self) -> None:
        pass

class LlamaService(proto.llama_pb2_grpc.LlamaServiceServicer):

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
        result = predict_llm.PredictLLM(request)
        # responses = "tesssssssssst"
        return proto.llama_pb2.PredictionResponse(response=result)

async def serve():
    server = aio.server()
    listen_addr = "[::]:50051"

    proto.llama_pb2_grpc.add_LlamaServiceServicer_to_server(LlamaService(), server)
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    print("gRPC server started")
    asyncio.run(serve())