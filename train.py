import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import argparse
import seaborn as sns

def read_data(year, month):
    _ =f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(_)
    df["duration"] = (df["lpep_dropoff_datetime"]-df["lpep_pickup_datetime"]).dt.total_seconds()/60
    df = df[(df["duration"]>1) & (df["duration"]<60)]
    df["DO_PU"] = df["DOLocationID"].astype(str) + "_" + df["PULocationID"].astype(str)
    return df

def transform(df, dv=None):
    features = ["DO_PU","trip_distance"]
    print(df[["DO_PU","trip_distance", "duration"]].head())
    train_dicts=df[features].to_dict(orient="records")
    if dv == None:
        dv = DictVectorizer()
        X = dv.fit_transform(train_dicts)
        y = df["duration"].values
    else:
        X = dv.transform(train_dicts)
        y = df["duration"].values
    return X, y, dv

def train_linear_model(X, y):
    model=LinearRegression()
    model.fit(X,y)
    return model

def plots(y_val, y_pred, model):
    residuals = y_val - y_pred
    
    # 1. Histogram of Residuals
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    # 2. Residuals vs Predicted
    plt.subplot(3, 1, 2)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")

    # 3. Predicted vs Actual
    plt.subplot(3, 1, 3)
    plt.scatter(y_val, y_pred, alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=2024, type=int)
    parser.add_argument("--month", default=12, type=int)
    args = parser.parse_args()
    year=args.year
    month=args.month
    df = read_data(year, month)
    X_train, y_train, dv = transform(df)
    print("X_train and y_train are ready")
    year_val = year+1 if month==12 else year
    month_val= 1 if month==12 else month+1
    df_val = read_data(year_val, month_val)
    X_val, y_val, _ = transform(df_val, dv)
    linear_model = train_linear_model(X_train, y_train)
    y_pred = linear_model.predict(X_val)
    print("rmse:", root_mean_squared_error(y_val, y_pred))
    joblib.dump((dv, linear_model), f"./models/{year}_{month:02d}_model.joblib")
    # plots(y_val, y_pred, linear_model)
    
if __name__ == "__main__":
    run()