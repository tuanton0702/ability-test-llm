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


def byte_reader(request):
    # print(request.decode('utf-8'))
    lst_data = request.csv_data.decode('utf-8').split('<s>')[1:]
    train_df = pd.DataFrame(lst_data, columns =['text'])
    return train_df

class ModelInit:
    def __init__(self) -> None:
        pass

class LlamaService(proto.llama_pb2_grpc.LlamaServiceServicer):
    
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