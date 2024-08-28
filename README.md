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
docker pull heronq02/viencoder:cuda11.8-final
docker pull nvcr.io/nvidia/tritonserver:24.07-py3
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

## Benchmark

```
$ perf_analyzer -m ensemble_model --shape TEXT:8 --concurrency-range 1:4 --collect-metrics
> Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 52.436 infer/sec, latency 19055 usec
Concurrency: 2, throughput: 56.491 infer/sec, latency 35376 usec
Concurrency: 3, throughput: 53.4915 infer/sec, latency 56020 usec
Concurrency: 4, throughput: 55.5469 infer/sec, latency 71970 usec
```
Currently can support ~ 55 inferences / sec with per inference is batch size 8 of strings. GPU memory usage is 4.242 GB





 




