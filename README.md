&nbsp;&nbsp;<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWN5NnAyOHo1NTVrbzgyOXE1YTRlczJwa3h1OHE3eW1maG93MTFqeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/84tJDt3AmqvKg6MGWu/giphy.gif" width="60"> 
# 🪸 Coral Health Classifier 

A deep learning web app that classifies coral images as **bleached** or **healthy** using a Convolutional Neural Network (CNN), deployed with **Streamlit**.

---

## 🌊 About the Project

This interactive app uses:
- 🧠 **Convolutional Neural Networks (CNNs)**
- ⚙️ **TensorFlow** and **Keras** for training and prediction
- 🎛️ **Adam optimizer** for adaptive learning
- 🎨 **Streamlit** for the web UI
- 🐍 **Python** for all back-end logic and data handling

Upload a coral photo and receive a prediction with confidence, model interpretation, and architecture overview.

---

## 🌐 Background: Why Coral Bleaching Matters

Coral reefs are critical ecosystems that host thousands of marine species. But climate change, especially rising ocean temperatures, can disrupt their balance.

A temperature increase of just 2°F can cause corals to expel the algae (zooxanthellae) that give them nutrients and color — resulting in **coral bleaching**. Bleached corals are more vulnerable to disease, struggle to reproduce, and can die without recovery.

> 🧊 In rare cases, even cold-water events have caused coral bleaching, such as in Florida in 2010.

> 🔥 Between 2014–2017, ~75% of tropical coral reefs worldwide experienced severe thermal stress. In some cases, entire reef systems collapsed.

Coral reefs protect coastlines from storms, support biodiversity, and generate billions in tourism. Their loss impacts wildlife, food security, coastal safety, and economies.

---

## 🗂️ Dataset

**Kaggle Dataset:** [Healthy and Bleached Corals](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification?select=healthy_corals)

| Feature              | Value                         |
|----------------------|-------------------------------|
| Total Images         | 923                            |
| Healthy Coral Images | 438                            |
| Bleached Coral Images| 485                            |
| Format               | JPEG                           |
| Max Size             | 300px (width or height)        |
| Source               | Flickr API                     |

The dataset was curated for training classification models to detect coral bleaching in real-world conditions.

---

## 🧠 About the Model

- 📦 **Architecture:** 3 convolutional layers (ReLU + MaxPooling) → Flatten → Dense layers with dropout → Output sigmoid
- 🔧 **Loss Function:** Binary Crossentropy  
- 🧪 **Optimizer:** Adam  
- 🔄 **Regularization:** Dropout to reduce overfitting  
- 📊 **Training:** 10 epochs with early stopping

### 📈 Validation Metrics
| Metric     | Value   |
|------------|---------|
| Accuracy   | 81.3%   |
| Precision  | 84.4%   |
| Recall     | 79.2%   |

### 📊 Confusion Matrix

|                       | Predicted: Bleached | Predicted: Healthy |
|-----------------------|---------------------|---------------------|
| **Actual: Bleached**  | 92                  | 12                  |
| **Actual: Healthy**   | 19                  | 69                  |

---

## 📱 App Preview

<img src="assets/screenshot.png" width="700"/>

---

## 🚀 Try the App

🔗 **Live App:** [Streamlit Deployment Link](https://your-app-link.streamlit.app)  
💻 **Run Locally:**

```bash
git clone https://github.com/yourusername/CoralClassification.git
cd CoralClassification
pip install -r requirements.txt
streamlit run app/main.py
