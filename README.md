---

# HateSpeechDetect-text

## **Project Overview**
In this project, I focused on benchmarking various machine learning models, deep learning architectures, and fine-tuned BERT-based models to evaluate their performance across multiple metrics. The aim was to establish a robust and efficient framework for text classification tasks, ultimately improving the overall accuracy of predictions by 25%.

### **Key Highlights**
1. **Data Collection:**
   - Scraped data from Twitter using `BeautifulSoup` (BS4), collecting tweets related to a specific domain for text classification tasks.
   - Preprocessed the scraped data to clean, tokenize, and transform it for effective model training.

2. **Machine Learning Models:**
   - Benchmarked traditional ML models, including:
     - Logistic Regression
     - Support Vector Classifier
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - AdaBoost Classifier
     - XGBoost Classifier

3. **Deep Learning Architectures:**
   - Evaluated the performance of deep learning models:
     - LSTM (Long Short-Term Memory)
     - Bi-LSTM (Bidirectional LSTM)
     - GRU (Gated Recurrent Unit)

4. **Fine-Tuned BERT-Based Models:**
   - Implemented and fine-tuned transformer-based models, including:
     - BERT
     - RoBERTa
     - ALBERT
     - DistilBERT
   - Employed **QLoRA-based fine-tuning** on the **Phi-2 model** to enhance its performance.

5. **Performance Metrics:**
   - Evaluated models on the following metrics:
     - Accuracy
     - Precision (Macro and Weighted)
     - Recall (Macro and Weighted)
     - F1-Score (Macro and Weighted)

6. **Achievements:**
   - Improved the best-performing model’s accuracy by **25%** through advanced fine-tuning and hyperparameter optimization.
   - Achieved consistent performance with BERT-based models, maintaining **91% accuracy** across multiple datasets.

7. **Best Performing Model:**
   - After rigorous benchmarking, **DistilBERT** emerged as the top-performing model, achieving a balanced performance with **91% accuracy**. 
   - The fine-tuned DistilBERT model has been pushed to the [datafreak/hatespeech-distill-bert](https://huggingface.co/datafreak/hatespeech-distill-bert) repository.

8. **FastAPI App:**
   - The FastAPI application serving the model has been **dockerized** to make it easily deployable and scalable.

## **Benchmarking Results**

| **Model**                     | **Accuracy** | **Precision (Macro)** | **Recall (Macro)** | **F1 (Macro)** | **Precision (Weighted)** | **Recall (Weighted)** | **F1 (Weighted)** |
|-------------------------------|--------------|------------------------|--------------------|----------------|--------------------------|-----------------------|--------------------|
| Logistic Regression           | 0.75         | 0.58                  | 0.67              | 0.59           | 0.86                    | 0.75                 | 0.79              |
| Support Vector Classifier     | 0.75         | 0.54                  | 0.60              | 0.55           | 0.83                    | 0.75                 | 0.78              |
| Decision Tree Classifier      | 0.72         | 0.51                  | 0.53              | 0.51           | 0.79                    | 0.72                 | 0.75              |
| Random Forest Classifier      | 0.83         | 0.59                  | 0.60              | 0.59           | 0.82                    | 0.83                 | 0.82              |
| Gradient Boosting Classifier  | 0.75         | 0.57                  | 0.63              | 0.57           | 0.84                    | 0.75                 | 0.79              |
| AdaBoost Classifier           | 0.71         | 0.54                  | 0.59              | 0.54           | 0.83                    | 0.71                 | 0.76              |
| XGBoost Classifier            | 0.81         | 0.59                  | 0.62              | 0.60           | 0.83                    | 0.81                 | 0.82              |
| LSTM                          | 0.73         | 0.57                  | 0.54              | 0.51           | 0.85                    | 0.73                 | 0.77              |
| Bi-LSTM                       | 0.75         | 0.55                  | 0.63              | 0.57           | 0.85                    | 0.75                 | 0.78              |
| GRU                           | 0.79         | 0.57                  | 0.66              | 0.60           | 0.85                    | 0.79                 | 0.81              |
| BERT                          | 0.91         | 0.76                  | 0.69              | 0.71           | 0.90                    | 0.91                 | 0.90              |
| roBERTa                       | 0.91         | 0.75                  | 0.72              | 0.74           | 0.90                    | 0.91                 | 0.90              |
| ALBERT                        | 0.91         | 0.76                  | 0.66              | 0.67           | 0.90                    | 0.91                 | 0.91              |
| DistilBERT                    | 0.91         | 0.79                  | 0.73              | 0.75           | 0.91                    | 0.91                 | 0.91              |
| Phi-2 (QLoRA Fine-Tuned)      | 0.90         | 0.75                  | 0.68              | 0.70           | 0.89                    | 0.90                 | 0.89              |

---

## **Technical Tools and Frameworks**
- **Data Collection:** BeautifulSoup (BS4), Python for scraping and preprocessing.
- **Modeling:** Scikit-learn, TensorFlow, PyTorch, and HuggingFace Transformers.
- **Fine-Tuning:** QLoRA-based approach for parameter-efficient fine-tuning of large language models like Phi-2.
- **Visualization:** Matplotlib and Seaborn for plotting evaluation metrics.

## **Results**
The project demonstrated the superior performance of BERT-based models, particularly after applying QLoRA-based fine-tuning on Phi-2, which outperformed traditional machine learning and standard deep learning models across all key metrics. Random Forest and XGBoost were the top-performing ML models, while GRU showed the best results among deep learning approaches.

---

## **How to Use the API Locally**
To run the **FastAPI app** locally and test the **DistilBERT** model for hate speech detection, follow the instructions below:

### **1. Clone the Repository**
Clone the repository containing the FastAPI application and model files.

```bash
git clone https://github.com/datafreak/hatespeech-distill-bert.git
cd hatespeech-distill-bert
```

### **2. Install Dependencies**
Ensure you have Python 3.10+ installed and set up a virtual environment (recommended).

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3. Run the Docker Container (Optional)**
To run the API using Docker, you can use the following commands. Make sure you’ve built the Docker image before.

```bash
docker build -t hatespeech-api .
docker run -p 8000:8000 hatespeech-api
```

This will start the FastAPI app and expose it on port `8000`.

### **4. Access the API**
Once the container is running, you can access the API via the following endpoint in your browser or Postman:

```
http://127.0.0.1:8000
```

To test the API, send a **POST request** to `http://127.0.0.1:8000/predict/` with the following body format:

```json
{
  "text": "Your tweet or text here"
}
```

This will return a prediction of whether the text is hate speech/offensive language or not.

---
