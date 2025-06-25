import os
import json
import numpy as np
from numpy import random as nprdm
import random
import tqdm
import multiprocessing
import argparse
import threading

random.seed(71)
nprdm.seed(71)


IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'



class RECDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 version = 'vg',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders
        self.version = version

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        
        ### ==> TODO: 实现Referring Expression Comprehension数据集
        result = []
        ### <===

        return result



class GCDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'images/VG_100K': '',
                    'images2/VG_100K_2':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Grounded Captioning数据集
        result = []
        ### <===

        return result
    
class REGDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'train2014': '',
                    'val2014':''
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Referring Expression Generation数据集
        result = []
        ### <===
        return result



class FlickrDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Flik30K-entities数据集
        result = []
        ### <===

        return result


 

class GPT4GenDataset():
    def __init__(self, filename="",
                 template_file="",
                 image_folders={
                    'flickr30k': '',
                 },
                 version='p',
                 total = None,
                 ratio = None,
                 shuffle = False,
                ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.version = version
        assert version in ['a', 'c', 'bc']

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(self,):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现GPT-4生成的数据集
        result = []
        ### <===

        return result
    


if __name__ == '__main__':

    datasets = [
        RECDataset(filename="", version='vg', ratio=1/20),
        GCDataset(filename="", ratio=1/20),
        RECDataset(filename="", image_folders={'train2014': ''}, version='coco'),
        REGDataset(filename=""),
        FlickrDataset(),
        GPT4GenDataset(version='a', template_file=r""),
        GPT4GenDataset(version='c', template_file=r""),
        GPT4GenDataset(version='bc', template_file=r""),
        GPT4GenDataset(filename='',version='bc', template_file=r""),
    ]

    ### ==> TODO: 实现用于Visual Grounding的指令微调数据集的构建
    tot = 0
    results = []
    ### <===

    # save
    with open("train_minicpmv_grounding.json", 'w') as f:
        json.dump(results, f)
    print("Total # exmaples: %d" % tot)