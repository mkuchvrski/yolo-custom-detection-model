# Glasses Detection Using YOLOv7

This project uses YOLOv7 to train a custom model for detecting glasses on people's faces. The application includes a Streamlit app for easy interaction with the model.

## Setup Instructions

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/mkuchvrski/yolo-custom-detection-model.git
cd yolo-custom-detection-model
```

### 2. Create Virtual Environment
Repository consists of requirements.txt file where all required libraries are listed, run:
```bash
python3 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### 3. Run streamlit app
Streamlit app is written in main_st.py file, run:
```bash
cd yolo-v7-gpu
streamlit run main_st.py
