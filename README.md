
# 🏡 House Price Prediction App

A web-based machine learning app built with **Streamlit** that predicts house prices based on user input. This project demonstrates an end-to-end ML workflow including data loading, preprocessing, exploratory data analysis (EDA), model training using **SVM**, evaluation, and prediction — all in one interface.

---

## 📂 Project Structure

```
├── app.py               # Streamlit web app
├── house.csv            # Dataset used for model training
├── house.ipynb          # Notebook for EDA and model experimentation
├── model.pkl            # Trained ML model
├── scaler.pkl           # Scaler for input feature normalization
```

---

## 📊 Features

- Load and display raw housing data
- Interactive **EDA visualizations**:
  - Histograms for major features
  - Correlation heatmap
- Preprocessing with **StandardScaler**
- Model training using **Linear Regression** (can be extended to SVM)
- **Model evaluation** (R² Score, MAE)
- Save and download trained model and scaler
- Interactive **Streamlit sidebar** to input new house features and predict prices in real-time

---

## 🔧 Tech Stack

| Tool            | Use                            |
|-----------------|---------------------------------|
| Python          | Core language                  |
| Pandas, NumPy   | Data manipulation              |
| Scikit-learn    | ML model, preprocessing        |
| Matplotlib, Seaborn | Visualizations            |
| Streamlit       | Web app frontend               |
| Pickle          | Model saving                   |

---

## 🎯 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-price-predictor
   cd house-price-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Example Prediction

> 📍Input:
- Bedrooms: 3  
- Bathrooms: 2  
- Sqft Living: 1800  
- Sqft Lot: 7500  
- Floors: 1  

> 💰 Output:
- Estimated House Price: **$537,000.00**

---

## 💡 Future Improvements

- Switch model to **Support Vector Regression (SVR)** for better performance
- Host app on Streamlit Cloud or Hugging Face Spaces
- Add feature importance and model selection options
- Upload dataset functionality for custom training

---

## 🙌 Author

**Muhammad Zeeshan**  
- AI Engineer Intern @ DeepVision.ai  
- Passionate about ML, Python, and full-stack development  
- 📧 mzeeshan3783901@gmail.com

---

## 📃 License

MIT License — feel free to use and contribute!
