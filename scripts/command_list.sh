#!/bin/bash

##########
## PREP ##
##########

cd "$DL2_HOME" || return
source venv/bin/activate

# Dataset creation
# Since ImageNet has a lot of different JPEG compression qualities, we check for each image if the compression target
# is lower than the current compression quality before we recompress (only for JPEG targets).
# Afterwards, copy the dataset variants to `data-hdd` or `data-sas` as needed
python scripts/analyze_image_dataset.py -s data-ssd/imagenet/raw -o data-ssd/imagenet/stats.csv -a quality

# Utility command for unmounting unused magick user filesystems if the above command throws warnings
cat /etc/mtab | grep magick | awk '{ print $2 }' | xargs -l fusermount -uz

# ImageNet
python dl2/single_encoder.py pillow -f JPEG --quality 85 --quality-list data-ssd/imagenet/stats.csv data-ssd/imagenet/raw/ data-ssd/imagenet/jpeg-85
python dl2/single_encoder.py pillow -f JPEG --quality 75 --quality-list data-ssd/imagenet/stats.csv data-ssd/imagenet/raw/ data-ssd/imagenet/jpeg-75
python dl2/single_encoder.py pillow -f JPEG --quality 50 --quality-list data-ssd/imagenet/stats.csv data-ssd/imagenet/raw/ data-ssd/imagenet/jpeg-50
python dl2/single_encoder.py pillow -f JPEG --quality 25 --quality-list data-ssd/imagenet/stats.csv data-ssd/imagenet/raw/ data-ssd/imagenet/jpeg-25
python dl2/single_encoder.py pillow -f JPEG --quality 10 --quality-list data-ssd/imagenet/stats.csv data-ssd/imagenet/raw/ data-ssd/imagenet/jpeg-10
python dl2/single_encoder.py pillow -f WebP --quality 85 data-ssd/imagenet/raw/ data-ssd/imagenet/webp-85
python dl2/single_encoder.py pillow -f WebP --quality 75 data-ssd/imagenet/raw/ data-ssd/imagenet/webp-75
python dl2/single_encoder.py pillow -f WebP --quality 50 data-ssd/imagenet/raw/ data-ssd/imagenet/webp-50
python dl2/single_encoder.py pillow -f WebP --quality 25 data-ssd/imagenet/raw/ data-ssd/imagenet/webp-25
python dl2/single_encoder.py pillow -f WebP --quality 10 data-ssd/imagenet/raw/ data-ssd/imagenet/webp-10

# Subsets of ImageNet for the compression vs subsampling trade-off analysis
python scripts/create_data_subset.py data-ssd/imagenet/raw data-ssd/imagenet/raw-50/ -s 0.5
python scripts/create_data_subset.py data-ssd/imagenet/raw data-ssd/imagenet/raw-40/ -s 0.4
python scripts/create_data_subset.py data-ssd/imagenet/raw data-ssd/imagenet/raw-30/ -s 0.3
python scripts/create_data_subset.py data-ssd/imagenet/raw data-ssd/imagenet/raw-20/ -s 0.2
python scripts/create_data_subset.py data-ssd/imagenet/raw data-ssd/imagenet/raw-10/ -s 0.1

# Places365
# Creating an image quality list is not necessary as the compression quality of Places365 images always is 75
python dl2/single_encoder.py pillow -f JPEG --quality 50 data-ssd/places365/raw/ data-ssd/places365/jpeg-50
python dl2/single_encoder.py pillow -f JPEG --quality 25 data-ssd/places365/raw/ data-ssd/places365/jpeg-25
python dl2/single_encoder.py pillow -f JPEG --quality 10 data-ssd/places365/raw/ data-ssd/places365/jpeg-10
python dl2/single_encoder.py pillow -f JPEG --quality 5  data-ssd/places365/raw/ data-ssd/places365/jpeg-05
python dl2/single_encoder.py pillow -f JPEG --quality 1  data-ssd/places365/raw/ data-ssd/places365/jpeg-01
python dl2/single_encoder.py pillow -f WebP --quality 50 data-ssd/places365/raw/ data-ssd/places365/webp-50
python dl2/single_encoder.py pillow -f WebP --quality 25 data-ssd/places365/raw/ data-ssd/places365/webp-25
python dl2/single_encoder.py pillow -f WebP --quality 10 data-ssd/places365/raw/ data-ssd/places365/webp-10
python dl2/single_encoder.py pillow -f WebP --quality 5  data-ssd/places365/raw/ data-ssd/places365/webp-05
python dl2/single_encoder.py pillow -f WebP --quality 1  data-ssd/places365/raw/ data-ssd/places365/webp-01

# Adjust the following block according to your hardware setup and resource control configuration (you might want to add it to your .bashrc)
#####
sudo apt install -y cgroup-tools
sudo cgcreate -a "$(id -u)" -t "$(id -u)" -g memory:dl2
echo 68719476736 | tee /sys/fs/cgroup/memory/dl2/memory.limit_in_bytes

function clear-cache () {
    echo 3 | sudo tee /proc/sys/vm/drop_caches
}
function memlim () {
    cgexec -g memory:dl2 "$@"
}
function ptdist () {
    python -m torch.distributed.run --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 "$@"
}
function memlim-ptdist () {
    cgexec -g memory:dl2 python -m torch.distributed.run --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 "$@"
}

export -f clear-cache
export -f memlim
export -f ptdist
export -f memlim-ptdist
#####

set -x

##########
### H1 ###
##########

## H1a ##
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 4 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w4;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 8 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w8;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w16;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 24 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w24;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 32 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w32;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-mem-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r50-raw-ssd-w40;
clear-cache;

clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 4 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w4;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 8 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w8;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 16 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w16;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 24 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w24;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 32 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w32;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-mem-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n r18-raw-ssd-w40;
clear-cache;

clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 4 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w4;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 8 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w8;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 16 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w16;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 24 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w24;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 32 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w32;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-mem-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-raw-ssd-w40;
clear-cache;

## Addition: ResNet50 and AlexNet with synthetic data loader
python dl2/main.py -d no/real/dataset -a resnet50 -l synthetic -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --synth-train-samples 1281167 --synth-val-samples 50000 --workspace logs/h1 -n r50-raw-synth;
python dl2/main.py -d no/real/dataset -a alexnet -l synthetic -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --synth-train-samples 1281167 --synth-val-samples 50000 --workspace logs/h1 -n alex-raw-synth;

## Addition: JPEG vs WebP CPU resource comparison
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 4 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w4;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 8 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w8;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 16 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w16;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 24 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w24;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 32 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w32;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-jpeg85-mem-w40;
clear-cache;

clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 4 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w4;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 8 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w8;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 16 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w16;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 24 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w24;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 32 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w32;
clear-cache;
python dl2/main.py -d data-ssd/imagenet/webp-85/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h1 -n alex-webp85-mem-w40;
clear-cache;

## H1b ##
clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-hdd/imagenet/raw -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-raw-hdd-dalic;
clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-ssd/imagenet/raw -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-raw-ssd-dalic;
clear-cache;
python dl2/dataloaders/standalone.py -d data-ssd/imagenet/raw -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-raw-mem-dalic;
clear-cache;

clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-hdd/imagenet/jpeg-75 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg75-hdd-dalic;
clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-ssd/imagenet/jpeg-75 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg75-ssd-dalic;
clear-cache;
python dl2/dataloaders/standalone.py -d data-ssd/imagenet/jpeg-75 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg75-mem-dalic;
clear-cache;

clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-hdd/imagenet/jpeg-10 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg10-hdd-dalic;
clear-cache;
memlim python dl2/dataloaders/standalone.py -d data-ssd/imagenet/jpeg-10 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg10-ssd-dalic;
clear-cache;
python dl2/dataloaders/standalone.py -d data-ssd/imagenet/jpeg-10 -b 256 -e 6 -w 40 --dali-cpu --workspace logs/h1 --experiment-name standalone-jpeg10-mem-dalic;
clear-cache;

##########
### H2 ###
##########

## H2a ##
# Imagenet
ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-raw-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-jpeg85-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-jpeg75-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-jpeg50-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-jpeg25-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-jpeg10-mem;
ptdist dl2/main.py -d data-sas/imagenet/webp-85/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-webp85-mem;
ptdist dl2/main.py -d data-sas/imagenet/webp-75/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-webp75-mem;
ptdist dl2/main.py -d data-sas/imagenet/webp-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-webp50-mem;
ptdist dl2/main.py -d data-sas/imagenet/webp-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-webp25-mem;
ptdist dl2/main.py -d data-sas/imagenet/webp-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r50-webp10-mem;

ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-raw-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-jpeg85-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-jpeg75-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-jpeg50-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-jpeg25-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-r18-jpeg10-mem;

ptdist dl2/main.py -d data-sas/imagenet/raw/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-raw-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-jpeg85-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-jpeg75-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-jpeg50-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-jpeg25-mem;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n acc-inet-alex-jpeg10-mem;

# Places365
ptdist dl2/main.py -d data-sas/places365/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-raw-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-jpeg50-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-jpeg25-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-jpeg10-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-05/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-jpeg05-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-01/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-jpeg01-mem;
ptdist dl2/main.py -d data-sas/places365/webp-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-webp50-mem;
ptdist dl2/main.py -d data-sas/places365/webp-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-webp25-mem;
ptdist dl2/main.py -d data-sas/places365/webp-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-webp10-mem;
ptdist dl2/main.py -d data-sas/places365/webp-05/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-webp05-mem;
ptdist dl2/main.py -d data-sas/places365/webp-01/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r50-webp01-mem;

ptdist dl2/main.py -d data-sas/places365/raw/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-raw-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-jpeg50-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-25/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-jpeg25-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-10/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-jpeg10-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-05/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-jpeg05-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-01/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-r18-jpeg01-mem;

ptdist dl2/main.py -d data-sas/places365/raw/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-raw-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-jpeg50-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-25/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-jpeg25-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-10/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-jpeg10-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-05/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-jpeg05-mem;
ptdist dl2/main.py -d data-sas/places365/jpeg-01/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n acc-p365-alex-jpeg01-mem;

## H2b ##
# R50 HDD
ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 5 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-raw-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 13 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg85-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 28 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg75-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 28 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg50-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 28 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg25-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 28 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg10-hdd-10h;

ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 10 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-raw-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 27 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg85-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 57 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg75-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 57 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg50-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 57 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg25-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 57 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg10-hdd-20h;

ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 34 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-raw-hdd-62h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 86 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg85-hdd-62h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 179 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --mixup 0.2 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg75-hdd-62h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 179 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --mixup 0.2 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg50-hdd-62h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 180 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --mixup 0.2 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg25-hdd-62h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 180 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --mixup 0.2 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r50-jpeg10-hdd-62h;

# R18 HDD
ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 5 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-raw-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 13 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg85-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 79 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg75-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 80 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg50-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 80 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg25-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 80 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg10-hdd-10h;

ptdist dl2/main.py -d data-sas/imagenet/raw/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 10 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-raw-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 27 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg85-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 159 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg75-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 160 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg50-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 160 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg25-hdd-20h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 160 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-r18-jpeg10-hdd-20h;

# Alex HDD
ptdist dl2/main.py -d data-sas/imagenet/raw/ -a alexnet -l dali-cpu -w 80 -b 512 -e 5 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-raw-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 80 -b 512 -e 13 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg85-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a alexnet -l dali-cpu -w 80 -b 512 -e 168 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg75-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 180 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg50-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a alexnet -l dali-cpu -w 80 -b 512 -e 187 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg25-hdd-10h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a alexnet -l dali-cpu -w 80 -b 512 -e 194 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg10-hdd-10h;

# Alex SSD
ptdist dl2/main.py -d data-sas/imagenet/raw/ -a alexnet -l dali-cpu -w 80 -b 512 -e 34 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-raw-ssd-5h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 80 -b 512 -e 67 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg85-ssd-5h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-75/ -a alexnet -l dali-cpu -w 80 -b 512 -e 84 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg75-ssd-5h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 89 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg50-ssd-5h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-25/ -a alexnet -l dali-cpu -w 80 -b 512 -e 94 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg25-ssd-5h;
ptdist dl2/main.py -d data-sas/imagenet/jpeg-10/ -a alexnet -l dali-cpu -w 80 -b 512 -e 96 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n time-inet-alex-jpeg10-ssd-5h;

## H2c ##
# ImageNet
ptdist dl2/main.py -d data-sas/imagenet/raw-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r50-raw50-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-40/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r50-raw40-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-30/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r50-raw30-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-20/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r50-raw20-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-10/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r50-raw10-mem;

ptdist dl2/main.py -d data-sas/imagenet/raw-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r18-raw50-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-40/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r18-raw40-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-30/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r18-raw30-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-20/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r18-raw20-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-10/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-r18-raw10-mem;

ptdist dl2/main.py -d data-sas/imagenet/raw-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-alex-raw50-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-40/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-alex-raw40-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-30/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-alex-raw30-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-20/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-alex-raw20-mem;
ptdist dl2/main.py -d data-sas/imagenet/raw-10/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --workspace logs/h2 -n space-inet-alex-raw10-mem;

# Places365
ptdist dl2/main.py -d data-sas/places365/raw-50/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r50-raw50-mem;
ptdist dl2/main.py -d data-sas/places365/raw-40/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r50-raw40-mem;
ptdist dl2/main.py -d data-sas/places365/raw-30/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r50-raw30-mem;
ptdist dl2/main.py -d data-sas/places365/raw-20/ -a resnet50 -l dali-cpu -w 16 -b 256 -e 90 --optimizer-batch-size 768 --lr 0.768 --wd 6.103515625e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r50-raw20-mem;

ptdist dl2/main.py -d data-sas/places365/raw-50/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r18-raw50-mem;
ptdist dl2/main.py -d data-sas/places365/raw-40/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r18-raw40-mem;
ptdist dl2/main.py -d data-sas/places365/raw-30/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r18-raw30-mem;
ptdist dl2/main.py -d data-sas/places365/raw-20/ -a resnet18 -l dali-cpu -w 64 -b 512 -e 90 --optimizer-batch-size 1536 --lr 1.536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-r18-raw20-mem;

ptdist dl2/main.py -d data-sas/places365/raw-50/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-alex-raw50-mem;
ptdist dl2/main.py -d data-sas/places365/raw-40/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-alex-raw40-mem;
ptdist dl2/main.py -d data-sas/places365/raw-30/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-alex-raw30-mem;
ptdist dl2/main.py -d data-sas/places365/raw-20/ -a alexnet -l dali-cpu -w 80 -b 512 -e 90 --optimizer-batch-size 1536 --lr 0.1536 --warmup 8 --wd 3.0517578125e-05 --amp --static-loss-scale 128 --data-mean 116.7,112.5,104.0 --data-std 61.0,60.2,62.6 --num-classes 365 --workspace logs/h2 -n space-p365-alex-raw20-mem;

##########
### H3 ###
##########

python dl2/regimen.py --stages 45,45 --datasets data-sas/imagenet/jpeg-10,data-sas/imagenet/raw --experiment-name inet-r50-10:45_raw:45-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 45,45 --datasets data-sas/imagenet/jpeg-25,data-sas/imagenet/raw --experiment-name inet-r50-25:45_raw:45-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 45,45 --datasets data-sas/imagenet/jpeg-50,data-sas/imagenet/raw --experiment-name inet-r50-50:45_raw:45-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 45,45 --datasets data-sas/imagenet/jpeg-75,data-sas/imagenet/raw --experiment-name inet-r50-75:45_raw:45-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;

python dl2/regimen.py --stages 80,10 --datasets data-sas/imagenet/jpeg-10,data-sas/imagenet/raw --experiment-name inet-r50-10:80_raw:10-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 80,10 --datasets data-sas/imagenet/jpeg-25,data-sas/imagenet/raw --experiment-name inet-r50-25:80_raw:10-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 80,10 --datasets data-sas/imagenet/jpeg-50,data-sas/imagenet/raw --experiment-name inet-r50-50:80_raw:10-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;
python dl2/regimen.py --stages 80,10 --datasets data-sas/imagenet/jpeg-75,data-sas/imagenet/raw --experiment-name inet-r50-75:80_raw:10-mem --workspace logs/h3 --training-script "-m torch.distributed.run" --script-args --nproc_per_node 3 --standalone --log_dir logs/distributed -r 1:1,2:1,3:1 dl2/main.py -a resnet50 -l dali-cpu -w 16 -b 256 --optimizer-batch-size 768 --lr 0.768 --wd 3.0517578125e-05 --amp --static-loss-scale 64;

##########
### H4 ###
##########

clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet50 -l pytorch -w 10 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r50-hdd-raw-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet50 -l pytorch -w 10 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r50-hdd-raw-minio --dataset MinIOFolder --cache-size 60666413056;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet50 -l pytorch -w 10 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r50-hdd-jpeg85-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet50 -l pytorch -w 10 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r50-hdd-jpeg85-minio --dataset MinIOFolder --cache-size 60666413056;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet18 -l pytorch -w 32 -b 512 -e 6 --lr 0.512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r18-hdd-raw-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet18 -l pytorch -w 32 -b 512 -e 6 --lr 0.512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r18-hdd-raw-minio --dataset MinIOFolder --cache-size 48855252992;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet18 -l pytorch -w 32 -b 512 -e 6 --lr 0.512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r18-hdd-jpeg85-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet18 -l pytorch -w 32 -b 512 -e 6 --lr 0.512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-r18-hdd-jpeg85-minio --dataset MinIOFolder --cache-size 48855252992;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-hdd-raw-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-hdd-raw-minio --dataset MinIOFolder --cache-size 43486543872;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-hdd-jpeg85-pytorch;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-hdd-jpeg85-minio --dataset MinIOFolder --cache-size 43486543872;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-ssd-raw-pytorch;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-ssd-raw-minio --dataset MinIOFolder --cache-size 43486543872;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-ssd-jpeg85-pytorch;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l pytorch -w 40 -b 512 -e 6 --lr 0.0512 --warmup 8 --wd 3.0517578125e-05  --amp --static-loss-scale 128 --no-checkpoints --skip-validation --no-persistent-workers --workspace logs/h4 -n inet-alex-ssd-jpeg85-minio --dataset MinIOFolder --cache-size 43486543872;
clear-cache;

###########################
### Learned Compression ###
###########################

## H5a / H5b ##
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/jpeg-85 -f JPEG --quality 85 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-85;
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/jpeg-50 -f JPEG --quality 50 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-50;
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/jpeg-10 -f JPEG --quality 10 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-10;
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/webp-85 -f WebP --quality 85 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-85;
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/webp-50 -f WebP --quality 50 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-50;
python dl2/single_encoder.py pillow data-sas/imagenet-5/raw data-sas/imagenet-5/webp-10 -f WebP --quality 10 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-10;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-1 -a bmshj2018-factorized -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-1;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-1 -a bmshj2018-hyperprior -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-1;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-1         -a mbt2018-mean         -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-1;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-1              -a mbt2018              -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-1;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-1     -a cheng2020-anchor     -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-1;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-1       -a cheng2020-attn       -m mse -q 1 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-1;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-4 -a bmshj2018-factorized -m mse -q 4 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-4;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-4 -a bmshj2018-hyperprior -m mse -q 4 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-4;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-4         -a mbt2018-mean         -m mse -q 4 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-4;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-4              -a mbt2018              -m mse -q 4 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-4;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-3     -a cheng2020-anchor     -m mse -q 3 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-3;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-3       -a cheng2020-attn       -m mse -q 3 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-3;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-8 -a bmshj2018-factorized -m mse -q 8 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-8;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-8 -a bmshj2018-hyperprior -m mse -q 8 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-8;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-8         -a mbt2018-mean         -m mse -q 8 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-8;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-8              -a mbt2018              -m mse -q 8 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-8;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-6     -a cheng2020-anchor     -m mse -q 6 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-6;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-6       -a cheng2020-attn       -m mse -q 6 -w 32 -b 512 --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-6;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-1 -a bmshj2018-factorized -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-1 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-1 -a bmshj2018-hyperprior -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-1 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-1         -a mbt2018-mean         -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-1 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-1              -a mbt2018              -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-1 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-1     -a cheng2020-anchor     -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-1 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-1       -a cheng2020-attn       -m mse -q 1 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-1 --report-file cuda_run.json;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-4 -a bmshj2018-factorized -m mse -q 4 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-4 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-4 -a bmshj2018-hyperprior -m mse -q 4 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-4 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-4         -a mbt2018-mean         -m mse -q 4 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-4 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-4              -a mbt2018              -m mse -q 4 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-4 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-3     -a cheng2020-anchor     -m mse -q 3 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-3 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-3       -a cheng2020-attn       -m mse -q 3 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-3 --report-file cuda_run.json;

python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-8 -a bmshj2018-factorized -m mse -q 8 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_factorized-mse-8 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_hyperprior-mse-8 -a bmshj2018-hyperprior -m mse -q 8 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-bmshj2018_hyperprior-mse-8 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018_mean-mse-8         -a mbt2018-mean         -m mse -q 8 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018_mean-mse-8 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/mbt2018-mse-8              -a mbt2018              -m mse -q 8 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-mbt2018-mse-8 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_anchor-mse-6     -a cheng2020-anchor     -m mse -q 6 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_anchor-mse-6 --report-file cuda_run.json;
python dl2/batch_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/cheng2020_attn-mse-6       -a cheng2020-attn       -m mse -q 6 -w 32 -b 512 --device cuda --workspace logs/h5 --experiment-name enc-dl-cheng2020_attn-mse-6 --report-file cuda_run.json;

python dl2/batch_encoder.py lossyless  data-sas/imagenet-5/raw data-sas/imagenet-5/lossyless-0.1              --beta 0.1 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.1;
python dl2/batch_encoder.py lossyless  data-sas/imagenet-5/raw data-sas/imagenet-5/lossyless-0.05             --beta 0.05 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.05;
python dl2/batch_encoder.py lossyless  data-sas/imagenet-5/raw data-sas/imagenet-5/lossyless-0.01             --beta 0.01 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.01;

## Addition: comparison of batched and single image decoding to quantify the effect of batched preprocessing on the dataset size
python dl2/single_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-1-mp  -a bmshj2018-factorized -m mse -q 1 -w 1 --workspace logs/h5 --experiment-name enc-mp-bmshj2018_factorized-mse-1;
python dl2/single_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-4-mp  -a bmshj2018-factorized -m mse -q 4 -w 1 --workspace logs/h5 --experiment-name enc-mp-bmshj2018_factorized-mse-4;
python dl2/single_encoder.py compressai data-sas/imagenet-5/raw data-sas/imagenet-5/bmshj2018_factorized-mse-8-mp  -a bmshj2018-factorized -m mse -q 8 -w 1 --workspace logs/h5 --experiment-name enc-mp-bmshj2018_factorized-mse-8;

## H5c ##
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/jpeg-85                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-85;
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/jpeg-50                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-50;
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/jpeg-10                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-10;
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/webp-85                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-85;
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/webp-50                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-50;
timeout --preserve-status 2h python dl2/single_decoder.py data-sas/imagenet-5/webp-10                   -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-10;

timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_factorized-mse-1 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_factorized-mse-1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_hyperprior-mse-1 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_hyperprior-mse-1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018_mean-mse-1         -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018_mean-mse-1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018-mse-1              -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018-mse-1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_anchor-mse-1     -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_anchor-mse-1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_attn-mse-1       -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_attn-mse-1;

timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_factorized-mse-4 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_factorized-mse-4;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_hyperprior-mse-4 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_hyperprior-mse-4;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018_mean-mse-4         -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018_mean-mse-4;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018-mse-4              -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018-mse-4;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_anchor-mse-3     -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_anchor-mse-3;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_attn-mse-3       -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_attn-mse-3;

timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_factorized-mse-8 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_factorized-mse-8;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/bmshj2018_hyperprior-mse-8 -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-bmshj2018_hyperprior-mse-8;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018_mean-mse-8         -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018_mean-mse-8;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/mbt2018-mse-8              -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-mbt2018-mse-8;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_anchor-mse-6     -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_anchor-mse-6;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/cheng2020_attn-mse-6       -l compressai -w 8 --workspace logs/h5 --experiment-name dec-dl-cheng2020_attn-mse-6;

timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/lossyless-0.1              -l lossyless --beta 0.1 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.1;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/lossyless-0.05             -l lossyless --beta 0.05 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.05;
timeout --preserve-status 2h python dl2/batch_decoder.py data-sas/imagenet-5/lossyless-0.01             -l lossyless --beta 0.01 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.01;

## Addition: lossyless vs conventional codecs on the entire dataset
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/jpeg-85 -f JPEG --quality 85 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-85 --report-file full_dataset.json;
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/jpeg-50 -f JPEG --quality 50 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-50 --report-file full_dataset.json;
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/jpeg-10 -f JPEG --quality 10 -w 256 --workspace logs/h5 --experiment-name enc-mp-jpeg-10 --report-file full_dataset.json;
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/webp-85 -f WebP --quality 85 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-85 --report-file full_dataset.json;
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/webp-50 -f WebP --quality 50 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-50 --report-file full_dataset.json;
python dl2/single_encoder.py pillow data-sas/imagenet/raw data-sas/tmp/webp-10 -f WebP --quality 10 -w 256 --workspace logs/h5 --experiment-name enc-mp-webp-10 --report-file full_dataset.json;

python dl2/single_decoder.py data-sas/tmp/jpeg-85 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-85 --report-file full_dataset.json;
python dl2/single_decoder.py data-sas/tmp/jpeg-50 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-50 --report-file full_dataset.json;
python dl2/single_decoder.py data-sas/tmp/jpeg-10 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-jpeg-10 --report-file full_dataset.json;
python dl2/single_decoder.py data-sas/tmp/webp-85 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-85 --report-file full_dataset.json;
python dl2/single_decoder.py data-sas/tmp/webp-50 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-50 --report-file full_dataset.json;
python dl2/single_decoder.py data-sas/tmp/webp-10 -l pillow -w 256 --workspace logs/h5 --experiment-name dec-mp-webp-10 --report-file full_dataset.json;

python dl2/batch_encoder.py lossyless data-sas/imagenet/raw data-sas/tmp/lossyless-0.1  --beta 0.1 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.1 --report-file full_dataset.json;
python dl2/batch_encoder.py lossyless data-sas/imagenet/raw data-sas/tmp/lossyless-0.05 --beta 0.05 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.05 --report-file full_dataset.json;
python dl2/batch_encoder.py lossyless data-sas/imagenet/raw data-sas/tmp/lossyless-0.01 --beta 0.01 -w 16 -b 1024 --device cuda --workspace logs/h5 --experiment-name enc-dl-lossyless-0.01 --report-file full_dataset.json;

python dl2/batch_decoder.py data-sas/tmp/lossyless-0.1  -l lossyless --beta 0.1 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.1 --report-file full_dataset.json;
python dl2/batch_decoder.py data-sas/tmp/lossyless-0.05 -l lossyless --beta 0.05 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.05 --report-file full_dataset.json;
python dl2/batch_decoder.py data-sas/tmp/lossyless-0.01 -l lossyless --beta 0.01 -w 8 --workspace logs/h5 --experiment-name dec-dl-lossyless-0.01 --report-file full_dataset.json;

##########
## Misc ##
##########

## Memory-limit experiment ##
function thrashing-test () {
    echo "$3" | tee /sys/fs/cgroup/memory/dl2/memory.limit_in_bytes
    echo "Memory limit set to $(cat /sys/fs/cgroup/memory/dl2/memory.limit_in_bytes) bytes."
    experiment="$1";
    device="$2";
    mkdir -p logs/misc/"$experiment";

    clear-cache;
    iostat -cdk -p "$device" -o JSON -y -t 1 > logs/misc/"$experiment"/iostat.json &
    iostatPID=$!;
    sleep 2;
    set -x;
    memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a resnet50 -l "$4" -w 16 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --no-checkpoints --workspace logs/misc -n "$experiment";
    set +x;
    kill -2 $iostatPID;
    cat /proc/meminfo > logs/misc/"$experiment"/meminfo.txt
}

thrashing-test "io-16G-pytorch" "sda4" "17179869184" "pytorch"
thrashing-test "io-16G-dali" "sda4" "17179869184" "dali-cpu"
thrashing-test "io-32G-pytorch" "sda4" "34359738368" "pytorch"
thrashing-test "io-32G-dali" "sda4" "34359738368" "dali-cpu"
thrashing-test "io-48G-pytorch" "sda4" "51539607552" "pytorch"
thrashing-test "io-48G-dali" "sda4" "51539607552" "dali-cpu"
thrashing-test "io-64G-pytorch" "sda4" "68719476736" "pytorch"
thrashing-test "io-64G-dali" "sda4" "68719476736" "dali-cpu"
thrashing-test "io-80G-pytorch" "sda4" "85899345920" "pytorch"
thrashing-test "io-80G-dali" "sda4" "85899345920" "dali-cpu"
thrashing-test "io-96G-pytorch" "sda4" "103079215104" "pytorch"
thrashing-test "io-96G-dali" "sda4" "103079215104" "dali-cpu"
thrashing-test "io-112G-pytorch" "sda4" "120259084288" "pytorch"
thrashing-test "io-112G-dali" "sda4" "120259084288" "dali-cpu"
thrashing-test "io-128G-pytorch" "sda4" "137438953472" "pytorch"
thrashing-test "io-128G-dali" "sda4" "137438953472" "dali-cpu"
thrashing-test "io-144G-pytorch" "sda4" "154618822656" "pytorch"
thrashing-test "io-144G-dali" "sda4" "154618822656" "dali-cpu"
thrashing-test "io-160G-pytorch" "sda4" "171798691840" "pytorch"
thrashing-test "io-160G-dali" "sda4" "171798691840" "dali-cpu"

# Resetting the memory limit to our default
echo 68719476736 | tee /sys/fs/cgroup/memory/dl2/memory.limit_in_bytes
set -x

## Average epoch time measurements for HDD ##
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-jpeg85-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-75/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-jpeg75-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-50/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-jpeg50-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-25/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-jpeg25-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-10/ -a resnet50 -l dali-cpu -w 40 -b 256 -e 6 --lr 0.256 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r50-jpeg10-hdd-w40;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-jpeg85-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-75/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-jpeg75-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-50/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-jpeg50-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-25/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-jpeg25-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-10/ -a resnet18 -l dali-cpu -w 40 -b 512 -e 6 --lr 0.512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n r18-jpeg10-hdd-w40;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/raw/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-raw-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg85-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-75/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg75-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-50/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg50-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-25/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg25-hdd-w40;
clear-cache;
memlim python dl2/main.py -d data-hdd/imagenet/jpeg-10/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg10-hdd-w40;
clear-cache;

clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/raw/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-raw-ssd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-85/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg85-ssd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-75/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg75-ssd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-50/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg50-ssd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-25/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg25-ssd-w40;
clear-cache;
memlim python dl2/main.py -d data-ssd/imagenet/jpeg-10/ -a alexnet -l dali-cpu -w 40 -b 512 -e 6 --lr 0.0512 --wd 6.103515625e-05 --amp --static-loss-scale 128 --workspace logs/misc -n alex-jpeg10-ssd-w40;
clear-cache;
