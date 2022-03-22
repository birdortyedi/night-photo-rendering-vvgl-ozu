# VVGL OZU - Night Photography Rendering Challenge @ NTIRE 2022, CVPR Workshops

Environment:
* NVIDIA RTX A4000 24GB
* CUDA 11
* PyTorch 1.8.0
* Python 3.6

Inference time:
* GPU: ~4 mins.
* CPU: ~20 mins.

To build the docker image:

```
docker build -t vvgl-ozu .
```

To run the solution submitted to the challenge:

```
docker run -v $(pwd)/data:/data vvgl-ozu ./run.sh
```
