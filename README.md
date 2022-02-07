# (DL)² - A Deep Learning Data Loading Analysis Sandbox

## Requirements
(DL)² is developed on Ubuntu using NVIDIA GPUs and the following software:
* Python 3.8
* CUDA 10.2 and CUDA 11.1 depending on the GPU
* PyTorch 1.9

PyTorch 1.9 is necessary due to suboptimal performance on NVIDIA Ampere GPUs of
previous releases (see [[1]](https://github.com/pytorch/pytorch/issues/60434)).
For other libraries used in this project, please see the respective 
requirements files for CUDA 10 and CUDA 11.

**Note:** The requirements files only cover requirements to execute experiments
and gather log data. To run the analysis notebooks, a common Jupyter setup
with Numpy/Pandas/Matplotlib is required. To generate plots in the same format
as they appear in the paper, a LaTeX installation is necessary as well. To 
analyze image quality with the `analyze_image_dataset` script, `magick` must be
present in the `PATH`.

## Setup
Clone the repository to your machine and export its location as an
environment variable.
```shell
git clone https://github.com/lbhm/dl2.git
export DL2_HOME=$(pwd)/dl2
```

We recommend that you either directly copy or symlink your experiment data in a
`data/` directory at the top-level of this repo.
```shell
ln -s <path_to_your_data> $DL2_HOME/data
```

You can run the experiments directly on your system or in Docker a container.

#### Docker
Switch to the `docker` directory and build the docker image.
```shell
cd docker
make build
```
The docker container uses CUDA 11.1.

#### Native
We recommend creating a virtual environment and installing the required 
packages into that environment.

```shell
virtualenv venv
source venv/bin/activate
pip install -r requirements_cuda11.txt
```

## Execution
#### Docker
We provide a convenience script called `run_docker.sh` that starts a docker 
container with some sensible runtime parameters set. Start the container by 
running
```shell
run_docker.sh
```
This will start a bash shell in which you can run experiments such as
```shell
python dl2/main.py -d <path_to_data> -a resnet50 -l dali-cpu -w 8 -b 256 -e 50 \
    -p 100 --label-smoothing 0.1 --lr 0.256 --lr-policy cosine --mom 0.875 \
    --wd 3.0517578125e-05 --amp --static-loss-scale 128 --memory-format nhwc -n docker-test
```
Alternatively, you can append your command to the `run_docker.sh` script:
```shell
run_docker.sh python <command>
```

#### Native Execution
To execute code directly on your system, run the `dl2/main.py` script. For 
example:
```shell
python dl2/main.py -d <path_to_data> -a resnet50 -l dali-cpu -w 8 -b 256 -e 50 \
    -p 100 --label-smoothing 0.1 --lr 0.256 --lr-policy cosine --mom 0.875 \
    --wd 3.0517578125e-05 --amp --static-loss-scale 128 --memory-format nhwc -n native-test
```

To run an experiment in a distributed setup, prepend your command with the 
`torch.distributed.run` module. For example:
```shell
python -m torch.distributed.run --nproc_per_node 4 dl2/main.py <args>
```

## Reproducibility
To reproduce the results from our paper, execute the instructions provided in
`scripts/command_list.sh`. The commands assume that there are two data folders,
`data-ssd` and `data-hdd`, that link to disks of the respective type. The file
is not designed to be executed in a fully automatic way so please read the 
comments. For example, we cannot provide a copy of the datasets we use. To 
acquire a copy of ImageNet and Places365, please see the download instrcutions
at [[2]](https://image-net.org/download.php) and
[[3]](http://places2.csail.mit.edu/download.html).

**Warning:** Executing all the commands will take a _very_ long time.

## Citation
TBD

## References
- [1] https://github.com/pytorch/pytorch/issues/60434
- [2] https://image-net.org/download.php
- [3] http://places2.csail.mit.edu/download.html