import gradio as gr
import cv2
import numpy as np
from Resize_images import process_image
from HOG_LBP_seperately import extract_features
from annotate import color_overlay_grid
import pickle
from LBP_Color_fun import extract_lbp_color_features

def pipeline_gradio(img, model_path):
    """Gradio-compatible wrapper around your pipeline."""
    # Convert uploaded image (PIL) to OpenCV BGR
    img_cv = np.array(img)
    
    ##### Feature Extraction #####
    img_processed = process_image(None, img_cv)
    #print("Processed Image Shape:", img_processed.shape)
    if model_path.endswith('5_11_2025_new.pkl'):
        fd = extract_lbp_color_features(img=img_processed) # df object
        X  = fd.drop(columns=["grid_id"]).values
        #X = np.concat([X, np.zeros((64,9))], axis=1)  # Padding to match model input
    else:
        fd  = extract_features(mode="lbp", img_in=img_processed, Gaus_blur=False) # df object
        X   = fd.drop(columns=["image_id", "image_name"]).values
        
    print(fd.head())
    
    
    #print(X.shape)
    
    ##### Load Model & Predict #####
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(X)
    print("prediction shape:", prediction.shape)
    #prediction = prediction.reshape(8, 8)
    
    ##### Annotate Image #####
    annotated_img = color_overlay_grid(None, prediction, img_in=img_processed)
    #annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # convert back for display
    
    return annotated_img, str(prediction)
# --- Gradio Interface ---
model_choices = {
    "LightGBM Model": r"E:\Sem_3\DS 203\E7\Code\model\lgbm_4_11_2025.pkl",
    "LightGBM Model Tuned": r"E:\Sem_3\DS 203\E7\Code\model\lgbm_model_tuned_2_5_11_2025_new.pkl"
    # add more models here if needed
}

demo = gr.Interface(
    fn=pipeline_gradio,
    inputs=[
        gr.Image(label="Upload or Drop an Image", type="pil"),
        gr.Dropdown(list(model_choices.values()), label="Select Model", value=list(model_choices.values())[0])
    ],
    outputs=[
        gr.Image(label="Annotated Output"),
        gr.Textbox(label="Prediction (8x8 Matrix)")
    ],
    title="Image Feature + Annotation Pipeline",
    description="Upload an image, select a model, and view annotated results."
)

if __name__ == "__main__":
    demo.launch()
