import gradio as gr
import numpy as np
from keras.models import load_model

model = load_model("./model")

def predict(img):
    img = img.reshape((-1, 128, 128, 3))
    prediction = model.predict(img).tolist()[0]
    # {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
    class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    return {
        class_names[0]: prediction[0], 
        class_names[1]: prediction[1],
        class_names[2]: prediction[2],
        class_names[3]: prediction[3],
    }

iface = gr.Interface(
    fn = predict, 
    inputs = gr.Image(shape=(128, 128)), 
    outputs = gr.outputs.Label(),
)

iface.launch(share=True)