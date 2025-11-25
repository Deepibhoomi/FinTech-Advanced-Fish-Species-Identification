# 🐟 Fish Species Classification using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📋 Overview
This project involves building a Deep Learning model to classify different species of fish from images. It utilizes **Transfer Learning** with the **MobileNetV2** architecture, a lightweight and efficient Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset.

The workflow includes data cleaning (removing ground truth masks), preprocessing, training a custom classification head, and evaluating performance using a confusion matrix and classification report.

## 📂 Dataset
The project uses the **"A Large Scale Fish Dataset"** (likely sourced from Kaggle).
* **Structure:** The dataset contains subdirectories for different fish classes.
* **Preprocessing Note:** The dataset includes "GT" (Ground Truth) folders containing masks. These are programmatically filtered out during the data loading phase to focus solely on the RGB images of the fish.
* **Classes:** The model is configured to classify **9 distinct fish species**.

## 🛠️ Tech Stack
* **Core:** Python
* **Deep Learning:** TensorFlow, Keras, MobileNetV2
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Utilities:** OS, Scikit-learn (Train-Test Split, Metrics)

## ⚙️ Model Architecture
The model uses a Transfer Learning approach:
1.  **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
    * *Input Shape:* 224x224x3
    * *Weights:* Frozen (non-trainable) to retain learned features.
    * *Pooling:* Average Pooling.
2.  **Custom Head (Classifier):**
    * Dense Layer (128 units, ReLU activation)
    * Dense Layer (128 units, ReLU activation)
    * Output Layer (9 units, Softmax activation)

## 📊 Workflow
1.  **Data Loading:** Iterates through the directory, filtering out non-image files and GT directories.
2.  **Exploratory Data Analysis (EDA):** Visualizes sample images from each class to ensure data integrity.
3.  **Data Split:** Splits data into Training (80%) and Testing (20%) sets.
4.  **Generators:** Uses `ImageDataGenerator` with `preprocess_input` specifically for MobileNetV2.
5.  **Training:** Trains the model for 5 epochs using the Adam optimizer and Categorical Crossentropy loss.
6.  **Evaluation:**
    * Calculates Test Loss and Accuracy.
    * Generates a Confusion Matrix heatmap.
    * Produces a Classification Report (Precision, Recall, F1-Score).
    * Visualizes specific prediction errors.

## 🚀 How to Run
1.  **Clone the repository** (if applicable).
2.  **Download the Dataset:** Ensure the dataset is located at:
    `/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset`
    *(Or update the `DIR` variable in the notebook to match your local path).*
3.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
    ```
4.  **Run the Notebook:** Execute the cells sequentially in Jupyter Notebook or Google Colab.

## 📈 Results
* **Validation Accuracy:** *[Insert your final validation accuracy here]*
* **Test Accuracy:** *[Insert your final test accuracy here]*

The notebook generates a heatmap of the Confusion Matrix at the end to visualize misclassifications between similar fish species.

## 🤝 Acknowledgements
* [MobileNetV2 Research Paper](https://arxiv.org/abs/1801.04381) for the efficient architecture.
* Keras Applications for providing the pre-trained weights.
