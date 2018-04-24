# -*- coding: utf-8 -*-
# @Time    : 2018/4/24 10:04
# @Author  : zhoujun
import argparse
import requests

URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path):
    img = open(image_path,'rb').read()
    msg = {'image':img}
    try:
        r = requests.post(URL,files=msg).json()
        if r['state']:
            print('sucess',r['predictions'])
        else:
            print('failed')
    except:
        print('failed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('-f','--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)