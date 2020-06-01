from torch.utils.data.dataset import Dataset
import os, sys
from PIL import Image
import torchvision.transforms as transforms
import csv

import numpy as np

class MultiViewDataSet(Dataset):
    """CSV dataset.
    Format of files:
        class_list: [class_name, class_id]
        train_file: [img_path, (x1, y1, x2, y2,) class_name]
    """

    def __init__(self, train_file, class_list, base_path='../data/img', transform=None,  data_type='train'):
        """
        Args:
            train_file (string): CSV file with training annotations
            class_list (string): CSV file with class id and class name
            annotations (string): CSV file with class list
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        #self.transform = transforms.Compose([transforms.ToTensor()])
        self.base_path = base_path
        
        self.x = []
        self.y = []
        
        # parse the provided class file. format: [class_name, class_id]
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            #raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)
            print(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
                
        # csv with [img_path, (x1, y1, x2, y2,) class_name]
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_csv_file(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            #raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
            print(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())
 
    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        """
        try:
            return function(value)
        except ValueError as e:
            #raise_from(ValueError(fmt.format(e)), None) 
            raise(e)
            
    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        'rb': python2
        'r': python3
        """
        if sys.version_info[0] < 3: 
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')
        
    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                #raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
                print(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result
    
    def _read_csv_file(self, csv_reader, classes):
        result = {}
        inval_imgs = []
        for line, row in enumerate(csv_reader):
            #escape header line
            if line == 0:
                continue
            line += 1
            
            try:
                img_file, class_name = row[:2]
            except ValueError:
                #raise_from(ValueError('line {}: format should be \'img_file,class_name\''.format(line)), None)
                print(ValueError('line {}: format should be \'img_file,class_name\''.format(line)), None)

            # check class name is correctly present
            if (class_name) == ('') :
                continue  
            
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
                
            if img_file not in result:
                result[img_file] = []
            result[img_file].append({'class': class_name})

            ext = 'jpg' 
            img_path_1 = f"{self.base_path}/{img_file}.{ext}"
            #img_path_1 = self.base_path+"/"+img_file+".jpg"
            img_path_2 = f"{self.base_path}/{img_file} (2).{ext}"
            #img_path_2 = self.base_path+"/"+img_file+" (2)"+".jpg"
            
            #figure out non-exisit images
            pair_1_missed = False
            pair_2_missed = False
            from pathlib import Path
            if not Path(img_path_1).exists() :
                inval_imgs.append(img_file)
                pair_1_missed = True
            if not Path(img_path_2).exists() :
                inval_imgs.append(img_path_2)
                pair_2_missed = True
            if pair_1_missed and pair_2_missed :
                continue
            
            views = []
            if pair_1_missed == False :
                views.append(img_path_1)  
            if pair_2_missed == False :
                views.append(img_path_2)
            
            self.x.append(views)
            self.y.append(self.name_to_label(class_name))
        
#         print("==invalid images:")
#         for img in inval_imgs:
#             print(img)

        return result
    
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            img = self._load_image(view)

            if self.transform is not None:
                img = self.transform(img)
            views.append(img)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
    
    def _load_image(self, image_index):
        img = Image.open(image_index).convert('RGB')
        return img

    
    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations
    
    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                print(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result
    
    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)
    
    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.image_names[i][0]) for i in indices]
            else:
                return [self.image_names[i] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.image_names]
            else:
                return [x[0] for x in self.image_names]
