from tensorflow.keras.models import model_from_json
import numpy as np 
import tensorflow as tf 
 
class FacialExpressionModel(object):
    EMOTION_LIST=['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise','Neutral']
    def __init__(self, model_jsoon_file, model_weights_file):
        with open(model_josn_file,"r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded= model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    
    def predict_emotion(self, img):
        self.preds= self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTION_LIST(np.argmax(self.preds))




