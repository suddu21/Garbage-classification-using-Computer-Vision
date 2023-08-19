import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
 
# load model
model = load_model('model520.h5')

#prediction classes
classnames = ['paper', 'cardboard', 'plastic', 'metal', 'food', 'battery', 'shoes', 'clothes', 'glass', 'medical']

#prediction function
def predict_image(img):
  img_4d=img.reshape(-1,224, 224,3)
  prediction=model.predict(img_4d)[0]
  return {classnames[i]: float(prediction[i]) for i in range(len(classnames))}

#Gradio interface
image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=3)
article="<p style='text-align: center; font-weight:bold;'>Model based on the VGG-16 CNN</p>"

gr.Interface(fn=predict_image, inputs=image,  title="Garbage Classifier VGG-16",
    description="This is a Garbage Classification Model Trained using VGG-16 architecture. Deployed to Hugging Face using Gradio.", outputs=label, article=article, enable_queue=True, interpretation='default').launch(share="True")