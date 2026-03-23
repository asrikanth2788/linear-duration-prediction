````markdown
# 🚖 NYC Taxi Trip Duration Prediction (Linear Regression)

This project trains a **Linear Regression** model to predict taxi trip durations using New York City green taxi trip data.  
It uses `scikit-learn`, `pandas`, `DictVectorizer`, and plots residuals for model evaluation.

---

## 🧰 Tools & Libraries

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib (for saving models)

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib pyarrow
````

or

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
linear-duration-prediction/
├── train.py             # Main training script
├── models/              # Saved models (.joblib)
├── notebooks/           # Optional: data exploration / visualization
├── README.md            # Project description
├── requirements.txt     # Dependencies
└── .gitignore
```

---

## 🚀 How to Run

1. Clone the repository:

```bash id="9f3kq1"
git clone git@github.com:asrikanth2788/linear-duration-prediction.git
cd linear-duration-prediction
```

2. Set up a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the training script:

```bash
python train.py --year 2024 --month 12
```

* `--year` and `--month` specify which NYC taxi dataset to use.
* The script automatically trains on that month and validates on the following month.
* Model artifacts (`DictVectorizer` + trained model) are saved in `models/` as `YYYY_MM_model.joblib`.

---

## 🔧 How It Works

1. **Read data:** Download NYC green taxi Parquet files from `https://d37ci6vzurychx.cloudfront.net`.
2. **Preprocess:**

   * Calculate trip duration in minutes
   * Filter durations between 1 and 60 minutes
   * Combine pickup & dropoff locations as `DO_PU` feature
3. **Transform:**

   * Convert features into a dictionary and vectorize using `DictVectorizer`
4. **Train Linear Regression:** Fit model to training data
5. **Validate:** Predict on next month’s data and compute RMSE
6. **Save model:** Persist `DictVectorizer` and trained model with `joblib`
7. **Optional plots:** Residuals histogram, residuals vs predicted, predicted vs actual

---

## 📊 Metrics

* **Root Mean Squared Error (RMSE)** is printed for validation month
* Use the optional plotting functions to visually inspect model performance

---

## 🧠 Notes

* Ensure internet connection to download NYC taxi Parquet files
* Virtual environment recommended to manage dependencies
* Modify `train.py` if you want to save plots or explore additional features

---

## 📌 Future Improvements

* Try **other regression models** (Random Forest, XGBoost, LightGBM)
* Add **feature engineering** (hour of day, day of week, weather data)
* Deploy as **API** using Flask or FastAPI
* Automate monthly training with **GitHub Actions / CI pipelines**

```



