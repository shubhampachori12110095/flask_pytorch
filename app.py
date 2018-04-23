import flask
import time
import cv2
import numpy as np
from model import Pytorch_model

app = flask.Flask(__name__)
model = None


@app.route('/')
def helloword():
    return 'hello'


@app.route('/predict', methods=['POST'])
def predict():
    data = {'state': False}
    if flask.request.method == 'POST':
        img = flask.request.files['image'].read()
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, flags=1)
        start = time.time()
        result = model.predict(img, is_numpy=True, topk=3)
        cost_time = time.time() - start
        data['predictions'] = list()
        for label, prob in result:
            m_predict = {'label': label, 'probability': float(prob)}
            data['predictions'].append(m_predict)
        data['state'] = True
        data['time'] = cost_time
    return flask.jsonify(data)


if __name__ == '__main__':
    model_path = 'net.pkl'
    model = Pytorch_model(model_path=model_path, img_shape=[28, 28], img_channel=3)
    app.run(debug=True)
