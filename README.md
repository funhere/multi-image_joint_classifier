# Multiple Images Joint Classifier
This repository contains a Pytorch implementation of classifier based on multiple images joint prediction. 

![](assets/mv_framework.png)

## Supported Features
- Joint training/inference on one or multiple images each time;
- Each image is processed independently on a CNN net, then fusion;
- Models/schedulers/optimizers are factorized and can be customized;
- Supported models: Resnet/ResNeXT serials/EfficientNet/DPN.etc.,
- A simple web app based on Flask and React.

## Requirements
- Python 3.7.0
- PyTorch
- CUDA
- CUDNN

## Installation
- Install Python 3.7.0
- pip install -r requirements.txt            

## Train
```
# Train on DPN131
$ ./distributed_train.sh 1 ./data --model dpn131 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 1 -j 1 &
or
# Train on resnext50d_32x4d with pretrained model
$ ./distributed_train.sh 1 ./data --model resnext50d_32x4d --sched cosine --epochs 2500 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 32 --resume your/best/model/model_best.pth.tar -j 2 &
or
# run on train.sh
$ sh train.sh
```

## Validation & Analysis

 Run `validation_and_analysis.ipynb` Jupyter notebooks for model validation and post-processing analysis.
 
 Please change the dataset paths and pre-trained weights path. 



# Web App
## Our example:  Usuke Images Classifier
Web app based on:
- [Flask web framework](https://palletsprojects.com/p/flask/)
- [React](https://reactjs.org/)

## Requirements

The following were used for model **detection** (see [requirements.txt](requirements.txt)):    
- PyTorch:  version  1.0.0
- Python:  version 3.7

The following were used for model **deployment**:    
- Heroku
- Flask:  version 1.0

## Installation
- pip install -r requirements.txt

## Launch the server
$> python3 app.py

## Access on web 
Run on web with:
 http://0.0.0.0:8001/

## Folder Structure
```
├── data
├── data_conversion
│   ├── __init__.py
│   ├── autoaugment.py
│   ├── config.py
│   ├── constants.py
│   ├── distributed_sampler.py
│   ├── loader.py
│   ├── mixup.py
│   ├── mv_dataset.py
│   ├── random_erasing.py
│   └── transforms.py
├── distributed_train.sh
├── docker
│   ├── Dockerfile
│   └── README.md
├── loss
│   ├── __init__.py
│   └── cross_entropy.py
├── models
│   ├── __init__.py
│   ├── adaptive_avgmax_pool.py
│   ├── conv2d_helpers.py
│   ├── densenet.py
│   ├── dpn.py
│   ├── factory.py
│   ├── gen_efficientnet.py
│   ├── gluon_resnet.py
│   ├── helpers.py
│   ├── inception_resnet_v2.py
│   ├── inception_v3.py
│   ├── inception_v4.py
│   ├── median_pool.py
│   ├── nasnet.py
│   ├── pnasnet.py
│   ├── registry.py
│   ├── resnet.py
│   ├── senet.py
│   ├── test_time_pool.py
│   └── xception.py
├── notebooks
│   ├── data_analysis.ipynb
├── optim
│   ├── __init__.py
│   ├── nadam.py
│   ├── optim_factory.py
│   └── rmsprop_tf.py
├── output
├── requirements.txt
├── scheduler
│   ├── __init__.py
│   ├── cosine_lr.py
│   ├── plateau_lr.py
│   ├── scheduler.py
│   ├── scheduler_factory.py
│   ├── step_lr.py
│   └── tanh_lr.py
├── train.py
├── train.sh
├── utils
│   ├── plots.py
│   └── utils.py
├── validation_and_analysis.ipynb
└── web_app
    ├── Dockerfile
    ├── README.md
    ├── models
    │   ├── README.md
    │   ├── classes.csv
    │   └── classes.txt
    ├── notebooks
    ├── requirements.txt
    └── src
        ├── app.py
        ├── config.yaml
        └── static
            ├── css
            │   └── custom.css
            ├── index.html
            └── js
                ├── main.js
                └── main.jsx
```



 


 

 
 
 

