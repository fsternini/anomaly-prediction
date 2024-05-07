import xgboost as xgb
import json
import os
import numpy as np

class Model:
    def __init__(self, conf_path):
        base_path = os.path.dirname(__file__) 
        abs_path = os.path.join(base_path, '..', conf_path)
        with open(abs_path, 'r') as file:
            self.conf = json.load(file)
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(base_path,'..',self.conf["model_path"]))
        self.threshold = self.conf["threshold"]

    def predict(self, data):
        data = np.array(data)
        if data.ndim<2:
            data = data.reshape((1,data.shape[-1]))
        dmatrix = xgb.DMatrix(data)
        predictions = self.model.predict(dmatrix)
        predictions[predictions>self.threshold]=True
        predictions[predictions<self.threshold]=False
        return predictions