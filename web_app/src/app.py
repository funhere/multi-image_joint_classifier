import sys
sys.path.append('/Users/simon.feng/Desktop/0_prj/mv/0_src/multi_images_joint_classifier')

import numpy as np
from collections import OrderedDict
from PIL import Image
import csv
from models import create_model, load_checkpoint

import yaml
import sys
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask
import requests
import torch
import json
import math

with open("config.yaml", 'r') as stream:
    APP_CONFIG = yaml.load(stream)

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
device='cpu' #['cuda','cpu']
NUM_CLASS = 10
BATCH_SIZE = 1
input_size = (3, 224, 224)
class_file = '../models/classes.csv'
path_ckpt = './output/train/'
best_model = '../models/model_best.pth.tar'
model_name='dpn131'
models = [   
    dict(model='dpn68b', checkpoint=path_ckpt+'20190820-072809-dpn68b-224'+best_model),
    dict(model='tf_efficientnet_b7', checkpoint=path_ckpt+'20190816-224614-tf_efficientnet_b7-600'+best_model),
    dict(model='tf_efficientnet_b4', checkpoint=path_ckpt+'20190816-093103-tf_efficientnet_b4-380'+best_model),
    dict(model='ig_resnext101_32x16d', checkpoint=path_ckpt+'20190816-011222-ig_resnext101_32x16d-224'+best_model),
    dict(model='dpn68b', checkpoint=path_ckpt+'20190820-072809-dpn68b-224'+best_model),
    dict(model='dpn131', checkpoint=path_ckpt+'20190820-073412-dpn131-224'+best_model),
    dict(model='wide_resnet50_2', checkpoint=path_ckpt+'20190818-044730-wide_resnet50_2-224'+best_model),
    dict(model='wide_resnet101_2', checkpoint=path_ckpt+'20190818-050005-wide_resnet101_2-224'+best_model),
    dict(model='resnext50d_32x4d', checkpoint=path_ckpt+'20190820-063613-resnext50d_32x4d-224'+best_model),
    dict(model='resnet34', checkpoint=path_ckpt+'20190815-065347-resnet34-224'+best_model),
]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

import torchvision.transforms as transforms
img_size=224
crop_pct=0.875
interpolation='bilinear'
scale_size = int(math.floor(img_size / crop_pct))
transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_checkpoint(model, checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'state_dict'
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def read_classes(class_file):
    # parse the provided class file. format: [class_name, class_id]
    try:
        with open(class_file, 'r', newline='') as file:
            classes = load_classes(csv.reader(file, delimiter=','))
            
    except ValueError as e:
        raise ValueError('invalid CSV class file: {}: {}'.format(class_file, e))
    
    return classes

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError('line {}: format should be \'class_name,class_id\''.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        #result[int(class_id)] = 'Type '+ class_name
        result[int(class_id)] = class_name
    return result

def load_model(model_name='dpn131', checkpoint='../models/model_best.pth.tar', device='cpu'):
    model = create_model(model_name, num_classes=NUM_CLASS, in_chans=3, pretrained=False)
    load_checkpoint(model, checkpoint)
    
    model = model.to(device)
    model.eval()
        
    return model


def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = Image.open(response.content).convert('RGB')
    return img

def load_image_bytes(files):
    x_views, y = [],[]
    
    for file in files:
        if file and allowed_file(file.filename): 
            print("==file:",file.filename)
            img = Image.open(BytesIO(file.read())).convert('RGB')
            if transform is not None:
                img_tf = transform(img)
                x_views.append(img_tf.unsqueeze(0))
        else:
            raise FileNotFoundError()

    return x_views

def predict(model, input, n=10):
    predictions = []
    topk_idx, topk_val = [],[]

    with torch.no_grad():
        #input, target = loader
        input = np.stack(input, axis=1)
        input = torch.from_numpy(input)
        input = input.to(device)
        output = model(input)

        output = output.softmax(1)
        topk_v, topk_i = output.topk(n, 1, True, True)
        topk_val.append(topk_v.detach().numpy())
        topk_idx.append(topk_i.detach().numpy())
        classes = read_classes(class_file)
    topk_idx = topk_idx[0][0]
    topk_val = topk_val[0][0]
    for pi, pv in zip(topk_idx, topk_val):
        if pv > 2e-5:
            predictions.append(
                {"class": str(classes[pi]), "prob": '{:.2f}%'.format(100*pv) } 
            )
    print("===predictions:",predictions)
    
    return predictions

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        imgs = load_image_url(url)
    else:        
        uploaded_files = flask.request.files.getlist("file")       
        imgs = load_image_bytes(uploaded_files)
    
    model.eval()
    res = predict(model, imgs)   
    return flask.jsonify(res)


@app.route('/api/classes', methods=['GET'])
def classes():
    classes = ['unknown','I','Ⅱ','Ⅱ-v','Ⅲ','Ⅲ-v','Ⅳ','Ⅴ','Ⅵ','Ⅶ']
    return flask.jsonify(classes)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}

model = load_model(model_name,checkpoint=best_model, device=device)

if __name__ == '__main__':
    port = os.environ.get('PORT', 8001)
    

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=False, host='0.0.0.0', port=port)
