# 🧠 Fake Internship Detector  
*Detecting fraudulent job postings using NLP and Machine Learning*

---

## 📖 Overview  
The **Fake Internship Detector** is an NLP-based machine learning project designed to identify **fraudulent job and internship postings**. With the rise of online job scams targeting students, this project aims to make the hiring space safer by analyzing the **language and writing patterns** of job descriptions.

The model learns to differentiate between **genuine** and **fake** postings using both **textual** and **behavioral** signals extracted from real job data.

---

## 🎯 Objectives  
- Detect fraudulent job/internship postings automatically  
- Use **NLP** to analyze textual patterns in job descriptions  
- Combine linguistic and behavioral features for improved accuracy  
- Build a robust model capable of identifying scams with high reliability  

---

## 🧩 Dataset  
- Source: Open-source job posting dataset (such as Kaggle – *Real or Fake Job Posting Prediction*)  
- Total records: ~18,000  
- Target variable: `fraudulent` → 0 = Genuine, 1 = Fake  
- Notable challenge: Class imbalance (~95% genuine, 5% fake)

---

## ⚙️ Project Workflow  

### 🪜 Step 1: Data Cleaning & Preprocessing  
- Combined multiple text columns (title, description, requirements, etc.) into one field  
- Removed URLs, punctuation, numbers, and special characters using **regex**  
- Converted all text to lowercase  
- Applied **stopword removal** and **lemmatization** (using NLTK)  
- Handled missing values and irrelevant columns  

### 📊 Step 2: Exploratory Data Analysis (EDA)  
- Visualized class imbalance (95% genuine vs. 5% fake)  
- Generated **missing value heatmaps**, **word clouds**, and **text length distributions**  
- Observed that fake posts were shorter, repetitive, and used flashy terms like “money”, “urgent”, “work from home”

### ⚙️ Step 3: Feature Engineering  
Added numerical indicators to strengthen the model:  
- `num_words`  
- `num_unique_words`  
- `num_chars`  
- `avg_word_len`  
- `num_exclamations`, `num_question_marks`, `num_uppercase`  

> 🧠 *These features complement NLP features by capturing stylistic and behavioral patterns.*

### 🧠 Step 4: Text Vectorization  
- Used **TF-IDF Vectorizer** to transform text into numerical features  
- Parameters:  
  - `max_features = 5000`  
  - `ngram_range = (1, 2)` → captures both single words and bigrams  
  - `stop_words = 'english'`  

### 🔀 Step 5: Data Balancing  
- Handled severe class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**  

### 🤖 Step 6: Model Training & Evaluation  
Trained and compared multiple models:  
- **Logistic Regression**  
- **Random Forest**  
- **SVM**

**Evaluation Metrics:**  
- Accuracy  
- Precision  
- Recall  
- F1-Score (primary metric)

> 🎯 **Final Model Performance:**  
> F1-Score ≈ **80%** (balanced performance between precision & recall)

### 📈 Step 7: Visualization  
- Used **Matplotlib**, **Seaborn**, and **Plotly** for interactive and comparative insights  
- Plotted feature distributions, confusion matrices, and ROC curves  

---

## 🧰 Tech Stack  
| Category | Tools & Libraries |
|-----------|------------------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| NLP | NLTK, Scikit-learn (TF-IDF) |
| Modeling | Logistic Regression, Random Forest, SVM |
| Balancing | SMOTE |
| Environment | Jupyter Notebook / Google Colab |

---

## 🚀 Results  
- Achieved **~80% F1-Score** on test data  
- Significantly improved recall (caught more fake postings)  
- Successfully identified linguistic and behavioral patterns unique to fraudulent ads  

---

## 📦 Project Structure  
app/
  dashboard.py
model/
  random_forest_model.joblib
  tfidf_vectorizer.joblib
notebooks/
  data_preprocessing_eda_baseline.ipynb
src/
  __pycache__/
    preprocessing.cpython-312.pyc
  model.py
  preprocessing.py


  
---

## 🔍 Future Work  
- Integrate a **REST API** for job posting platforms  
- Experiment with **transformer-based NLP models** (e.g., BERT, RoBERTa)  
- Expand dataset for multilingual job postings  

---

## 🏆 Key Takeaways  
- Combined **NLP + Feature Engineering** for hybrid modeling  
- Learned to handle **imbalanced datasets** effectively  
- Improved skills in **text preprocessing**, **model tuning**, and **data visualization**  
- Created a system that can genuinely help users avoid online fraud  

---

## 📫 Connect With Me  
💼 [LinkedIn](https://www.linkedin.com/in/sameer-chauhan-363298269/)  
💻 [GitHub](https://https://github.com/SamIeer)  
📧 Sameerchauhan212204@gmail.com 

---

⭐ *If you liked this project, consider giving it a star on GitHub!*  
