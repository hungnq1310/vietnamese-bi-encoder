import sys
import os
from typing import List, Any
import time
from functools import partial
import numpy as np

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from underthesea import word_tokenize

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from trism import TritonModel
from utils import client

# Parse environment variables
#
model_name    = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:8000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")


grpc = protocol.lower() == "grpc"


# ----------------------------------------------------------
# Create triton model.
model = TritonModel(
  model=model_name,                 # Model name.
  version=model_version,            # Model version.
  url=url,                          # Triton Server URL.
  grpc=grpc                         # Use gRPC or Http.
)

# View metadata.
for inp in model.inputs:
  print(f"name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}\n")
for out in model.outputs:
  print(f"name: {out.name}, shape: {out.shape}, datatype: {out.dtype}\n")


# ----------------------------------------------------------
class ListStr(BaseModel):
    texts: List[str]

############
# FastAPI
############


app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/viencoder")
async def viencoder(textRequest: ListStr) -> JSONResponse:

    # Word-segment the input texts
    texts = textRequest.texts
    text_responses = await preprocessing(texts)
    print(text_responses)
    text_obj = np.array(text_responses, dtype="object")

    # -------------------INFERENCE--------------------
    try:
        start_time = time.time()
        outputs = model.run(data = [text_obj])
        end_time = time.time()
        print("Process time: ", end_time - start_time)
        return JSONResponse(outputs.tolist())
    except Exception as e:
        return JSONResponse(content={"Error": "Inference failed with error: " + str(e)})

    # ----------------------------------------------------------------

@app.post("/word-segment")
async def preprocessing(texts: List[str]) -> List[str]:
    return [word_tokenize(sentence, format="text") for sentence in texts]


###################
# Helper functions
###################

def requestGenerator(text_obj, input_name, output_name, dtype):
    # define protocol
    if protocol.lower() == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input and output data
    inputs = [client.InferInput(input_name, text_obj.shape, dtype)]
    inputs[0].set_data_from_numpy(text_obj)
    outputs = [client.InferRequestedOutput(output_name)]
    return inputs, outputs