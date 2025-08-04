# 🧠 Digit Classifier App

A custom-trained Convolutional Neural Network (CNN) built with TensorFlow and Streamlit to recognize real-world handwritten digits from pen-and-paper images. This app preprocesses input images, classifies digits 0-9, and displays predictions through a clean web interface.

---

## 🚀 Features

- Trained on real handwritten digit images (cropped from pen-on-paper)
- Image preprocessing including grayscale conversion, thresholding, and resizing
- Fast and accurate digit prediction using a CNN model
- Simple and user-friendly Streamlit interface for easy testing
- Supports uploading your own images for classification

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** TensorFlow/Keras (CNN model)  
- **Image Processing:** OpenCV, NumPy  
- **Dataset:** Manually cropped and labeled handwritten digit images

---

## 💻 How to Run Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/asheenafazal/digit-classifier-app.git
   cd digit-classifier-app
Create and activate a Python virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv mnist-env
# Windows
.\mnist-env\Scripts\activate
# macOS/Linux
source mnist-env/bin/activate
Install dependencies:


pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Open your browser and go to:
http://localhost:8501

Upload an image of a handwritten digit and see the prediction!

<img width="1076" height="480" alt="image" src="https://github.com/user-attachments/assets/81f40f90-1695-41d6-a3de-24018ab56f71" />


📞 Contact
LinkedIn: www.linkedin.com/in/asheena-fazal-179538267

Email: asheenafazal@gmail.com

Feel free to reach out if you want to collaborate or have questions!
