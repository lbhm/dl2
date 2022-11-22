# (DL)² - A Deep Learning Data Loading Analysis Sandbox

This repository contains the source code, experiment logs, and result analyses
for our ICDE 2023 paper **"The Art of Losing to Win: Using Lossy Image Compression
to Improve Data Loading in Deep Learning Pipelines"**. A preprint of the paper is
available at [[1]](https://lbeh.me/pdf/The_Art_of_Losing_to_Win.pdf).

(DL)² is an experiment sandbox that we built for our research.

## Requirements

(DL)² is developed on Ubuntu using NVIDIA GPUs and the following software:

* Python 3.8
* CUDA 10.2 and CUDA 11.1, depending on the GPU
* PyTorch 1.9

PyTorch 1.9 is necessary due to suboptimal performance on NVIDIA Ampere GPUs of
previous releases (see [[2]](https://github.com/pytorch/pytorch/issues/60434)).
For other libraries used in this project, please see the respective
requirements files for CUDA 10 and CUDA 11.

**Note:** The requirements files only cover requirements to execute experiments
and gather log data. To run the analysis notebooks, a common Jupyter setup
with `numpy`, `pandas`, and `matplotlib` is required. To generate plots in the
same format as they appear in the paper, a LaTeX installation is necessary as
well. To analyze image quality with the `analyze_image_dataset` script, `magick`
must be installed and accessible on your `$PATH`.

## Setup

Clone the repository to your machine and export its location as an
environment variable.

```shell
git clone https://github.com/lbhm/dl2.git
export DL2_HOME=$(pwd)/dl2
```

We recommend that you either directly copy or symlink your benchmark datasets
in a `data/` directory at the top-level of this repo.

```shell
ln -s <path_to_your_data> $DL2_HOME/data
```

As we compare different storage types in the paper, we created multiple `data-x/`
directories with *x* referring to the storage type.

## Installation

You can run the experiments in a Docker container or directly on your system.

### Docker Installation

Switch to the `docker/` directory and build the docker image.

```shell
cd docker
make build
```

The docker container uses CUDA 11.1.

### Native Installation

We recommend creating a virtual environment and installing the required
packages into that environment.

```shell
virtualenv venv
source venv/bin/activate
pip install -r requirements_cuda11.txt
```

## Execution

The main user interface of (DL)² is the `dl2/main.py` script. To get an overview
of possible parameters and their usage, run

```shell
python dl2/main.py -h
```

Other scripts and helper tools all have a CLI documentation available via `-h`.
 
### Docker Execution

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

Alternatively, you can directly pass your command to the `run_docker.sh` script:

```shell
run_docker.sh python <command>
```

### Native Execution

To execute code directly on your system, run the `dl2/main.py` script. For
example:

```shell
python dl2/main.py -d <path_to_data> -a resnet50 -l dali-cpu -w 8 -b 256 -e 50 \
    -p 100 --label-smoothing 0.1 --lr 0.256 --lr-policy cosine --mom 0.875 \
    --wd 3.0517578125e-05 --amp --static-loss-scale 128 --memory-format nhwc -n native-test
```

To run an experiment with multple GPUs or in a distributed setup, prepend your
command with the `torch.distributed.run` module. For example:

```shell
python -m torch.distributed.run --nproc_per_node 4 dl2/main.py <args>
```

## Logs, Notebooks, and Plots

Experiment logs for all results that we report in the paper are in `logs/`. The
logs are organized by the correponding hypothesis that we invetigate in the paper.
The experiment names, such as `inet-alex-ssd-raw-pytorch`, encode important parameters
of the respective experiment. The full list of parameters is  always logged in
the first line of each `experiment_report.json` file. In addition to the experiment
logs, `logs/misc/` contains some additional summary plots about the datasets we
used.

**Note:** Some of the experiment folders in `h5/` (learned compression) are empty
since the experiments did not finish within their time limit as we describe in
the paper.

The `notebooks/` directory contains the Jupyter notebooks that we used for analyzing
the experiment results and creating the plots in our paper. All files in `plots/`
can be recreated by running the respective notebooks.

**Note:** A `matplotlib`-compatible TeX installation is required to recreate the
plots as they appear in our paper. Alternatively, with the `DEBUG` flag set to
`True`, the plots can be recreated without TeX though with a different layout.

## Reproducibility

To reproduce the results from our paper, execute the instructions provided in
`scripts/command_list.sh`. The commands assume a server infrastructure as we describe
in our paper and refer to three data folders (`data-ssd`, `data-hdd`, and `data-sas`)
that link to disks of the respective type.

**Note:** The command list was not designed to be executed in a fully automatic
way so please read the comments. For example, we cannot provide a copy of the datasets
that we use. To acquire a copy of ImageNet and Places365, please see the download
instrcutions at [[3]](https://image-net.org/download.php) and
[[4]](http://places2.csail.mit.edu/download.html).

**Warning:** Executing all the commands will take a *very* long time.

## Citation

```bibtex
@inproceedings{behme_art_2023,
    title = {The Art of Losing to Win: Using Lossy Image Compression to Improve Data Loading in Deep Learning Pipelines},
    author = {Lennart Behme and Saravanan Thirumuruganathan and Alireza Rezaei Mahdiraji and Jorge{-}Arnulfo Quian\'{e}{-}Ruiz and Volker Markl},
    booktitle = {39th {IEEE} International Conference on Data Engineering},
    eventtitle =  {ICDE '23},
    year = {2023},
    address = {Anaheim, California}
}
```

## References

* [1] <https://lbeh.me/pdf/The_Art_of_Losing_to_Win.pdf>
* [2] <https://github.com/pytorch/pytorch/issues/60434>
* [3] <https://image-net.org/download.php>
* [4] <http://places2.csail.mit.edu/download.html>
  
