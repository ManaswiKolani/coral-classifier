import streamlit as st
from util import load_trained_model, preprocess_image, predict
from io import StringIO
# Constants
MODEL_PATH = '../Models/model.h5'
CLASS_NAMES = ['bleached_corals', 'healthy_corals']
IMG_SIZE = (256, 256)


st.set_page_config(page_title="Coral Health Classifier", page_icon="🪸", layout="wide")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div {
        color: black !important;
    }

    /* Outer uploader container */
    [data-testid="stFileUploader"] {
        background-color: #ffe5d9 !important;
        border: 1px solid #ffccbc !important;
        padding: 0.5rem;
        border-radius: 12px;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Inner drag and drop area */
    [data-testid="stFileUploader"] > div > div {
        background-color: white !important;
        border: 2px dashed #cccccc !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        min-height: 60px !important;
    }

    /* Drag and drop section specifically */
    [data-testid="stFileUploader"] section {
        background-color: white !important;
        color: black !important;
    }

    /* Upload button */
    [data-testid="stFileUploader"] button {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: 1px solid #cccccc !important;
    }

    /* Text inside uploader */
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div {
        color: black !important;
    }

    </style>
""", unsafe_allow_html=True)


# Header GIF
st.markdown("""
    <div style="width: 100%; margin-bottom: 2rem;">
        <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGk1M3p6a3pxY290MjU1ODgzMXNocWYwMHl5MGszN2cyYzNpODI0eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26u47309v8SiMoiKk/giphy.gif" style="width: 100%; height: 120px; object-fit: cover; border-radius: 12px;">
    </div>
""", unsafe_allow_html=True)

# App title
st.title("🪸 Coral Health Classifier")
st.write("Upload a coral image to classify it as **bleached** or **healthy**.")

# Load model 
@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)

model = get_model()

# Upload image
uploaded_file = st.file_uploader("**Choose an image**", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('🔍 Analyzing coral image...'):
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = preprocess_image("temp.jpg", target_size=IMG_SIZE)
        label, confidence = predict(model, img, CLASS_NAMES)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    with col2:
        st.markdown("## 🧠 Prediction")
        st.markdown(
            f"<h1 style='font-size: 48px; margin-bottom: 10px;'>"
            f"{label.replace('_', ' ').capitalize()}</h1>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size: 20px; margin-top: 0px;'>"
            f"Confidence: <b>{confidence:.2%}</b></p>", unsafe_allow_html=True)
        st.progress(min(confidence, 1.0))
        
        if 'bleached' in label.lower():
            st.markdown("🔴 <span style='background-color:#ffd6d6; padding:5px 10px; border-radius:8px;'>Coral Bleaching Detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("🟢 <span style='background-color:#d4f5d0; padding:5px 10px; border-radius:8px;'>Healthy Coral</span>", unsafe_allow_html=True)
            
    


st.markdown("""
<div style='
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    padding: 2rem;
    border-radius: 12px;
    max-width: 800px;
    margin: 3rem auto;
'>

<h2 style='text-align: center;'>📘 Model Interpretation & Performance</h2>

<h3>🔍 How to Interpret Predictions</h3>
<p>
- The model classifies coral images as either <strong>bleached</strong> or <strong>healthy</strong> based on visual cues.<br>
- A confidence score near <strong>100%</strong> means the model is very certain.<br>
</p>

<h3>📈 Model Accuracy</h3>
<p>
- The model achieved a validation accuracy of <strong>81.3%</strong>.<br>
- Precision: <strong>84.4%</strong><br>
- Recall: <strong>79.2%</strong><br>
</p>


<h3>📊 Confusion Matrix</h3>
<table style='width: 100%; border-collapse: collapse; margin-top: 1rem;'>
  <tr>
    <th></th>
    <th>Predicted: Bleached</th>
    <th>Predicted: Healthy</th>
  </tr>
  <tr>
    <td><strong>Actual: Bleached</strong></td>
    <td>92</td>
    <td>12</td>
  </tr>
  <tr>
    <td><strong>Actual: Healthy</strong></td>
    <td>19</td>
    <td>69</td>
  </tr>
</table>

<p style='margin-top: 1rem;'>
✅ <strong>True Positives:</strong> 92 &nbsp;&nbsp;
✅ <strong>True Negatives:</strong> 69<br>
⚠️ <strong>False Positives:</strong> 19 &nbsp;&nbsp;
⚠️ <strong>False Negatives:</strong> 12
</p>


<h3>📁 Dataset Description</h3>
<p>
Dataset: <a href='https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification?select=healthy_corals' target='_blank'>
Healthy and Bleached Corals Image Classification</a><br><br>

- Total images: <strong>923</strong><br>
- Healthy corals: <strong>438</strong> images<br>
- Bleached corals: <strong>485</strong> images<br>
- Format: <strong>JPEG</strong>, resized to a maximum of 300px (width or height)<br>
- Source: <strong>Flickr API</strong><br><br>

This dataset supports researchers and developers in building coral health classification models. It helps monitor environmental changes and supports conservation efforts for coral reef ecosystems.
</p>
""", unsafe_allow_html=True)

with st.expander("📊 View Model Architecture"):
    architecture = []
    model.summary(print_fn=lambda x: architecture.append(x))
    st.text("\n".join(architecture))

st.markdown("</div>", unsafe_allow_html=True)


