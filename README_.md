# トヨタ自動車 誤踏み検知プロジェクト
This repository contains 2 Pytorch implementations of collision detection based on single-frame and multi-frame prediction. 

## Initialization
All the commands bellow are run form the root of the repository.

#### Build the docker image
The following command will build the docker image `collision_detection_single` with the working directory set to `/workshop` and the default command being `bash`. 
The image includes the minimum requirements to run the trained model in inference.
```
$ ./docker/build.sh
```

#### Start a docker container

`docker/docker.sh` starts a docker container with the current directory mounted under `/workshop`. The script accepts a flag `--docker` to add options when starting the container and additional commands to start inside the container.
```
$ ./docker/docker.sh
```


## Dataset
#### Single-frame Dataset
Dataset Name: ETH

Dataset structure:
data/ETH/
|-- test.csv                       <- Test items refer from original dataset
|-- test_merged.csv                <- Test items + validation items
|-- train.csv                      <- Train items refer from original dataset
|-- val.csv                        <- Validation items refer from original dataset
|-- training                       <- Train dataset
|   |-- DSCN2561                   <- Sample train case with frames
|   |   |-- 50_up_labels.txt       <- Lable file(extended 50 frames ahead) 
|   |   |-- images
|   |   |   |-- frame_00001.jpg    <- Sample frame
|   |   |   |-- frame_00002.jpg
...
|   |   -- labels.txt              <- orig lable file 
...
|-- validation                     <- Validation dataset
|   |-- DSCN2571                   <- Sample test case with frames
|   |   |-- 50_up_labels.txt       <- Lable file(extended 50 frames ahead) 
|   |   |-- images
|   |   |   |-- frame_00001.jpg    <- Sample frame
|   |   |   |-- frame_00002.jpg
...
|   |   -- labels.txt              <- orig lable file 

Notes: 
    Due to the warning distance of the raw data is too short, all cases are 50 frames ahead.

#### Multi-frame Dataset
Dataset Name: BDD Attention
Link: https://bdd-data.berkeley.edu

Dataset structure:
data/BDDA/
|-- classes.txt                       <- classes: 0,1  
|-- labels                            <- Pre-processed Annotation lables 
|   |-- test.csv                      <- Test items refer from original dataset
|   |-- train.csv                     <- Train items refer from original dataset
|   |-- train_val.csv                 <- Train items + validation items
|   |-- val.csv                       <- Validation items refer from original dataset
|   |-- train_refine.csv              <- Refined Train items, clean the wrong annoations
|   |-- val_refine.csv                <- Refined Val items, clean the wrong annoations
|-- csv                               <- Annotation files 
|   |-- BDDA_test.csv                 <- Annotation Test file
|   |-- BDDA_train.csv                <- Annotation Train file
|   |-- BDDA_val.csv                  <- Annotation Val file
|-- clips_test                        <- Cutout Test clips
|   |-- forward_collision_warning     <- Warning clips
|   |   |-- 1003_168_EwGY3y43.mp4     <- Sample video
|   |   |-- 1003_168_PDRXSKC2.mp4
...
|-- clips_train                       <- Cutout Train clips
|   |-- forward_collision_warning     <- Warning clips
|   |   |-- 1004_1_v8pX6AXx.mp4       <- Sample video
|   |   |-- 1005_2_nBaXSCeL.mp4
...
|-- clips_val                         <- Cutout Train clips
|   |-- forward_collision_warning     <- Warning clips
|   |   |-- 1737_1_OGAT4iGU.mp4       <- Sample video
|   |   |-- 1737_1_Tz9gn8OO.mp4
...


## Solution 1: Single-frame prediction
Project directory: projects/collision_detection_single
All the utility scripts are located under the directory `scripts`.

### Install the required packages
Install the requirements files. 
```
$ pip3 install -r requirements.txt
```

### Models & results(accuracy)
- Model: resnet18-224               Top-1 93.27
- Model: ssl_resnet18-224           Top-1 90.86
- Model: mobilenetv3_large_100-224  Top-1 90.80
- Model: mobilenetv3_large_075-224  Top-1 90.48
- Model: inception_resnet_v2-299    Top-1 90.29
- Model: resnet152-224              Top-1 90.16
- Model: swsl_resnet50-224          Top-1 90.04
- Model: mobilenetv3_rw-224         Top-1 89.40
- Model: swsl_resnet18-224          Top-1 88.26
- Model: seresnext26tn_32x4d-224    Top-1 87.69
- Model: tv_resnet34-224            Top-1 87.63
- Model: resnext50d_32x4d-224       Top-1 87.44
- Model: seresnext26d_32x4d         Top-1 86.29
- Model: seresnext26t_32x4d-224     Top-1 85.66
- Model: mobilenetv3_small_075-224  Top-1 83.38


### Train
#### scripts/train.sh
`scripts/train.sh` run training on input data given the pre-trained model to load. 

```
$ ./scripts/train.sh
```

The following parameters are usually customized to suit your training: 
- `data`: path to dataset
- `model`: model architecture (default: resnet18)
- `sched`: LR scheduler (default: "step")
- `opt`: Optimizer (default: "sgd")
- `aa`: Use AutoAugment policy. "v0" or "original". (default: None)
- `epochs`: number of epochs to train
- `warmup-epochs`: epochs to warmup LR
- `lr`: learning rate (default: 0.01)
- `batch-size`: input batch size for training
- `resume`: Resume full model and optimizer state from checkpoint

Example:
```
# Shell of training.
sudo CUDA_VISIBLE_DEVICES=2 python train.py \
                        /workspace/data/ETH \
                        --model resnet18 \
                        --sched cosine \
                        --opt sgd \
                        --aa original \
                        --epochs 120 \
                        --warmup-epochs 5 \
                        --lr 1e-5 \
                        --batch-size 256 \
                        --resume ./output/train/best_models/resnet18-224_model_best_93.27014207885163_41.pth.tar
```


### Validation
#### scripts/val.sh
`scripts/val.sh` run validation on test data given the trained model to load. 

```
$ ./scripts/val.sh
```

The following parameters are usually customized to suit your situation: 
- `data`: path to dataset
- `model`: model architecture (default: resnet18)
- `checkpoint`: path to trained model

Example:
```
# Shell of validation.
CUDA_VISIBLE_DEVICES=2 python validate.py \
                    /workspace/data/ETH/validation \
                    --model resnet18 \
                    --checkpoint ./output/train/best_models/resnet18-224_model_best_93.27014207885163_41.pth.tar
```


### Inference
#### scripts/infer.sh
`scripts/infer.sh` run inference on test data given the trained model to load. 

```
$ ./scripts/infer.sh
```

The following parameters are usually customized to suit your situation: 
- `video_dir`: path to the video input data
- `pred_dir`: path to prediction results
- `video_name`: output file name, in frames
- `checkpoint`: path to trained model

Example:
```
 python infer.py --video_dir demo/intersection_2.mp4 \
                 --pred_dir demo/pred_out \
                 --video_name intersection_2 \
                 --checkpoint output/train/best_models/resnet18-224_model_best_93.27014207885163_41.pth.tar
```
Inference results： demo/pred_out/pred_intersection_2_single.mp4



## Solution 2: Multi-frame prediction
Project directory: projects/collision_detection_multiframe
All the utility scripts are located under the directory `scripts`.


### Install the required packages
Install the requirements files. 
```
$ pip3 install -r requirements.txt
```


### Train
#### scripts/train.sh
`scripts/train.sh` run training on input data given the pre-trained model to load. 

```
$ ./scripts/train.sh
```

The following parameters are usually customized to suit your training:
- `video_dir`: path to dataset
- `train_dir`: path to train files
- `batch_size`: Batch Size
- `lr`: learning rate (default: 0.001)

Example:
```
sudo CUDA_VISIBLE_DEVICES=0 python train.py \
                        --video_dir /workspace/data/BDDA \
                        --train_dir labels/train_refine.csv \
                        --batch_size 8 \
                        --lr 0.001
```


### Evaluation
#### scripts/eval.sh
`scripts/eval.sh` run evaluation on test data given the trained model to load. 

```
$ ./scripts/eval.sh
```

The following parameters are usually customized to suit your evaluation:
- `video_dir`: path to dataset
- `val_dir`: path to validation files
- `checkpoint`: path to checkpoints

Example:
```
sudo CUDA_VISIBLE_DEVICES=0 python eval.py \
                        --video_dir /workspace/data/BDDA \
                        --val_dir labels/val_refine.csv \
                        --checkpoint checkpoints/best_models/spt_112_res34_model_best_91.139244_37_ig65m.pth.tar
```


### Inference
#### scripts/infer.sh
`scripts/infer.sh` run inference on test data given the trained model to load. 

```
$ ./scripts/infer.sh
```

The following parameters are usually customized to suit your inference:
- `video_dir`: path to the video file input data
- `pred_dir`: path to the prediction results
- `num_frames`: Frames number of input clip length. 8 or 16 or 32, (default: 16).
- `averaging_size`: averaging 1~5 latest clips to make video-level prediction (or smoothing). (default: 3)
- `which_data`: feed data source. "webcam", "videofile", (default: "videofile")

Example:
```
$ python infer.py --video_dir demo/intersection_2.mp4  \
                  --pred_dir demo/pred_out \
                  --num_frames 16 \
                  --averaging_size 3
```
Inference results： demo/pred_out/pred_intersection_2_multi.mp4

