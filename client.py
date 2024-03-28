import typing as t

import grpc
from fastapi import Depends, FastAPI, status, UploadFile, File
from fastapi.responses import JSONResponse
from google.protobuf.json_format import MessageToDict

import proto.llama_pb2
import proto.llama_pb2_grpc
import uvicorn

app = FastAPI()

async def grpc_channel():
    channel = grpc.aio.insecure_channel("localhost:50051")
    client = proto.llama_pb2_grpc.LlamaServiceStub(channel)
    return client

@app.post('/create-dataset')
async def CreateDataset(files: t.List[UploadFile] = File(...),
                        client: t.Any = Depends(grpc_channel)) -> JSONResponse:
    lst_byte = []
    for file in files:
        content = await file.read()
        lst_byte.append(content)
    result = await client.UploadMultipleJsonFiles(proto.llama_pb2.MultipleJsonFiles(files = lst_byte))
    return JSONResponse(MessageToDict(result))
@app.post('/fine-tune')
async def Finetune(file: UploadFile = File(...),
                  client: t.Any = Depends(grpc_channel)) -> JSONResponse:
    data = await file.read()

    result = await client.FineTune(proto.llama_pb2.FineTuneRequest(csv_data=data))
    return JSONResponse(MessageToDict(result))

@app.post('/predict')
async def Predict(prompt: str,
                  client: t.Any = Depends(grpc_channel)) -> JSONResponse:
    predict = await client.Predict(proto.llama_pb2.PredictionRequest(prompt=prompt))
    return JSONResponse(MessageToDict(predict))

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)