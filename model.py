from tensorflow.keras.models import model_from_json
import numpy as np 
import tensorflow as tf 
 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class FacialExpressionModel(object):
    EMOTION_LIST=['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise','Neutral']
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file,"r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model= model_from_json(loaded_model_json)
        
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    
    def predict_emotion(self, img):
        self.preds= self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTION_LIST[np.argmax(self.preds)]




