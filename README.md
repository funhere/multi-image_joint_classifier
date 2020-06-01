# Multiple Images Joint Classifier
This repository contains a Pytorch implementation of classifier based on multiple images joint prediction. 

![](assets/mv_framework.png)

## Supported Features
- Joint training/inference on one or multiple images each time;
- Each image runs on one backbone net and combines all the results to classify.
- Models/schedulers/optimizers are factorized and can be customized.
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


## Models & Draft results
- Model: dpn131-224                     
- Model: wide_resnet101_2-224           
- Model: resnext50d_32x4d-224          
- Model: wide_resnet50_2-224            
- Model: dpn68b-224                    
- Model: ig_resnext101_32x16d-224       
- Model: tf_efficientnet_b7-600         
- Model: tf_efficientnet_b4-380         
- Model: resnet34-224                   

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
 
## Table of Contents
```
└── web_app
    ├── Dockerfile
    ├── README.md
    ├── assets
    ├── docs
    │   ├── 1_b_gcloud.md
    │   └── 2_heroku_app.md
    ├── models                    <- Deployed models are store here.
    │   ├── README.md
    │   ├── classes.csv
    │   └── classes.txt
    ├── notebooks
    ├── requirements.txt
    └── src
        ├── app.py                 <- Main web application 
        ├── config.yaml
        └── static
            ├── css
            │   └── custom.css
            ├── index.html
            └── js
                ├── main.js
                └── main.jsx
```

## Launch the server
$ python3 app.py

## Access on web 
Run on web with:
 http://0.0.0.0:8001/




 


 

 
 
 

