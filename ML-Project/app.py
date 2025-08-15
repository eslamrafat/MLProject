import joblib
from PIL import Image
import numpy as np
import gradio as gr


logreg = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(data):
    if data is None:
        return "No image!"
    
    img = data  
    if len(img.shape) == 3:
        img = img.mean(axis=2)  
    
    img = Image.fromarray(img.astype(np.uint8)).resize((8, 8), Image.LANCZOS)
    img_array = np.array(img).reshape(64)
    img_array = 16 - (img_array / 16)
    img_array = img_array.reshape(1, -1)
    img_scaled = scaler.transform(img_array)
    prediction = logreg.predict(img_scaled)[0]
    return str(prediction)


gr.Interface(fn=predict, inputs=gr.Sketchpad(), outputs="label").launch()