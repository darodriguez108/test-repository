import numpy as np
import pickle
from flask import Flask, request, jsonify, abort
import os
import pandas as pd

#open excel file

pre = os.path.dirname(os.path.realpath('__file__'))
fname = 'DataReadyforKNNModel.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)

#get min and max values

somin = data['Sabot_OD'].min()
somax = data['Sabot_OD'].max()

lmin = data['Lp_Mass'].min()
lmax = data['Lp_Mass'].max()

smin = data['Squeeze'].min()
smax = data['Squeeze'].max()

#unpickle model

knn_pkl = open("KNNRegressionModel.pkl","rb")

model = pickle.load(knn_pkl)



#api

app = Flask(__name__)

@app.route('/50cal', methods=['PUT'])
def make_predict():
    requestdata = request.get_json(force=True)
    predict_request = [[requestdata['muzzle'], requestdata['sabotOD'], requestdata['LPMass']]]
    predict_request = np.array(predict_request)
    x = predict_request.item(1)
    y = predict_request.item(0)
    z = predict_request.item(2)
    squeeze = x - y
    cb = (x-somin)/(somax - somin)
    eb = (z-lmin)/(lmax - lmin)
    fb = (squeeze-smin)/(smax - smin)
    predict = np.array([[cb, eb, fb]])
    y_hat = model.predict(predict)
    output = [y_hat[0]]
    return jsonify(results=output)


#    inputs = {'muzzle': requestdata['muzzle'], 
#              'sabotOD':requestdata['sabotOD'], 
#              'LPMass': requestdata['LPMass']}
#    squeeze = inputs['sabotOD'] - inputs['muzzle']
#    cb = (inputs['sabotOD']-somin)/(somax - somin)
#    eb = (inputs['LPMass']-lmin)/(lmax - lmin)
#    fb = (squeeze-smin)/(smax - smin)
    
#    predict_request = np.array([[cb, eb, fb]])
#    y = model.predict(predict_request)
#    return jsonify(y)

app.run(port=5000, debug=True)