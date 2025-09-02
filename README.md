<h1>
  <img src="https://www.thiings.co/_next/image?url=https%3A%2F%2Flftz25oez4aqbxpq.public.blob.vercel-storage.com%2Fimage-U8mhT6gzngEVbkP6EAZfbx0wT1zfMl.png&w=2048&q=75" width="150"/>
  Coral Health Classifier
</h1>


A deep learning web app that classifies coral images as **bleached** or **healthy** using a Convolutional Neural Network (CNN), deployed with **Streamlit**.

---

## About the Project

This interactive app uses:
- **Convolutional Neural Networks (CNNs)**
- **TensorFlow** and **Keras** for training and prediction
- **Adam optimizer** for adaptive learning
- **Streamlit** for the web UI
- **Python** for all back-end logic and data handling

Upload a coral photo and receive a prediction with confidence, model interpretation, and architecture overview.

---

## Dataset

**Kaggle Dataset:** [Healthy and Bleached Corals](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification?select=healthy_corals)

| Feature              | Value                         |
|----------------------|-------------------------------|
| Total Images         | 923                            |
| Healthy Coral Images | 438                            |
| Bleached Coral Images| 485                            |
| Format               | JPEG                           |
| Max Size             | 300px (width or height)        |
| Source               | Flickr API                     |

---

## About the Model

-  **Architecture:** 3 convolutional layers (ReLU + MaxPooling) → Flatten → Dense layers with dropout → Output sigmoid
-  **Loss Function:** Binary Crossentropy  
-  **Optimizer:** Adam  
-  **Regularization:** Dropout to reduce overfitting  
-  **Training:** 10 epochs with early stopping

### Validation Metrics
| Metric     | Value   |
|------------|---------|
| Accuracy   | 81.3%   |
| Precision  | 84.4%   |
| Recall     | 79.2%   |

### Confusion Matrix

|                       | Predicted: Bleached | Predicted: Healthy |
|-----------------------|---------------------|---------------------|
| **Actual: Bleached**  | 92                  | 12                  |
| **Actual: Healthy**   | 19                  | 69                  |

---

## Preview

<img src="Assets/preview.png" width="700"/>

---

## 🚀 Try the App
💻 **Run Locally:**

```bash
git clone https://github.com/yourusername/CoralClassification.git
cd CoralClassification
pip install -r requirements.txt
streamlit run app/main.py



