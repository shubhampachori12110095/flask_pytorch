import requests

URL = 'http://127.0.0.1:5000/predict'
def predict_result(image_path):
    img = open(image_path,'rb').read()
    msg = {'image':img}

    r = requests.post(URL,files=msg).json()
    if r['state']:
        print('sucess',r['result'])
    else:
        print('failed')

if __name__ == '__main__':
    image_path = 'E:/zj/mnist/mnist_img/test/0/1.jpg'
    predict_result(image_path)