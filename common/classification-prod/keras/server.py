import argparse
import pickle

import flask
import mlflow.keras
import numpy as np
from flask import Response, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = None
model = None

app = flask.Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_sentence():
    global model
    global tokenizer
    
    params = flask.request.json
    if not params:
        return Response("{'error': 'Missing the sentence to classify'}", status=400, mimetype='application/json')
    sentence = params.get('sentence')
    print('sentence', sentence)
    prediction = model.predict(np.array(pad_sequences(tokenizer.texts_to_sequences([sentence]),
                                                      maxlen=1368, padding='post')))
    print('prediction', prediction)
    
    if prediction >= 0.5:
        return jsonify({
            'label': 1
        })
    else:
        return jsonify({
            'label': 0
        })


def main():
    global tokenizer
    global model

    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_path')
    args = argparser.parse_args()
    model_path = args.model_path
    
    with open('/tmp/classification_tokenizer.pkl', 'rb') as fp:
        tokenizer = pickle.load(fp)
        
    model = mlflow.keras.load_model(model_path)
    app.run(host='0.0.0.0', port=80)

if __name__ == '__main__':
    main()
