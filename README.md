
# 🏥 Medical Insurance Cost Predictor & Analytics Dashboard

An interactive Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to estimate individual medical insurance charges and analyze underlying healthcare risk factors in real time.

---

## 🌟 Key Features

- **Real-Time Cost Prediction:** Instantly calculates estimated annual insurance premiums based on individual user parameters.
- **Log Transformation Handling:** Leverages log-transformed target variables (`np.log1p` / `np.expm1`) for stabilized machine learning predictions.
- **Interactive Visualizations (Plotly):**
  - **Cost vs. National Benchmarks:** Bar chart comparing predicted premiums against average smoker and non-smoker baselines.
  - **Age Trajectory Simulation:** Line chart showing projected premium increases from age 18 to 70 holding other health metrics constant.
- **Risk Metrics & Indicators:** Visual feedback on BMI risk levels, smoking impacts, and family size dependencies.
- **Responsive Dashboard UI:** Clean multi-tab layout with an interactive sidebar built using Streamlit.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Frontend / Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (`GradientBoostingRegressor`, `RandomForestRegressor`, `LinearRegression`)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Data Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/)

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit web application
├── projet.ipynb            # Jupyter Notebook for EDA, model training & evaluation
├── medical_insurance.csv   # Dataset used for training
├── insurance_model.pkl     # Serialized trained Machine Learning model
├── feature_columns.pkl     # Saved feature schema for One-Hot Encoded input matching
├── requirements.txt        # List of required Python dependencies
└── README.md               # Project documentation
