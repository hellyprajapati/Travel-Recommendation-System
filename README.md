# 🗺️ Travel Recommendation System

A smart travel recommendation system that suggests destinations and hotels based on user preferences like **city**, **destination type**, and **price range**.

---

## 🚀 **Project Objective**

This project aims to help travelers find **relevant destinations** and **budget-friendly hotels** by using:
- **Content-Based Filtering** (for destinations)
- **Price-Based Filtering** (for hotels)

Built using **Python**, **Flask**, **Pandas**, **Scikit-learn**, and **web scraping**.

---

## 🔍 **Features**

✅ Recommend places to visit in a selected city & category  
✅ Recommend hotels in the selected city based on user’s price preference  
✅ Interactive web application using Flask  
✅ Visualizations for data insights (EDA)  
✅ Evaluation metrics (Precision@K, Recall@K, NDCG) to measure recommendation quality

---

## 🗂️ **Project Structure**
Travel-Recommendation-System/
│
├── data/ # CSV files for destination & hotel data
├── model/ # Jupyter Notebooks and saved ML models (.pkl)
├── templates/ # HTML files (index.html, results.html)
├── app.py # Flask application
├── README.md # Project overview


---

## 📊 **Data Sources**

- Destinations scraped from [Holidify](https://www.holidify.com/)
- Hotels scraped from [Make My Trip](https://www.makemytrip.com/)  

---

## 📊 EDA & Model Development
EDA Notebooks: Visualizations, summary stats, key insights.

Model Notebook: Training and evaluation of the recommendation model.

Metrics: Precision@K, Recall@K, F1-score, NDCG.

##💡 Future Improvements
Add recommendations for flights, trains, and packages.

Integrate user reviews and ratings for more accurate suggestions.

Implement hybrid filtering combining collaborative + content-based methods.



