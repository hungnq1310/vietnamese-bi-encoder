# Vietnamese BiEncdoer

This repostory is used for deploying model [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) by using Triton Inference Server and FastAPI. 

## Prepare
### Export model
Currently model is exported at path `/models/viencoder/1` using optimum of huggingface:
```
optimum-cli export onnx --model "bkai-foundation-models/vietnamese-bi-encoder" models/viencoder/1
```

## Usage
### With Docker Compose
```
docker-compose up
```
This command will start two images: 
- tritonserver: using file `Dockerfile.triton` to build more required packages as well as deloying docker, currently image version is `24.07` . 
- viencoder: using `Dockerfile` to create image for client. After being initialized, we can access API through `http://127.0.0.1:7999/docs`

> [!WARNING]  
> All of image are require GPU device, all of configuration can be viewed on file `docker-compose.yaml`

## Explain
FastAPI will deploy two main API:
1. preprocessing (`/word-segment`) for normalize texts - using `underthesea`
2. embedding (`/embed`) for creating meaning vector of texts








 




