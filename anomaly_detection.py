import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully.")
        print(f"Dataset Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

def clean_data(data):
    cleaned_data = data.drop(columns=['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID'])
    cleaned_data['TransactionDate'] = pd.to_datetime(cleaned_data['TransactionDate'])
    cleaned_data['PreviousTransactionDate'] = pd.to_datetime(cleaned_data['PreviousTransactionDate'])
    cleaned_data['TimeSinceLastTransaction'] = (
        cleaned_data['TransactionDate'] - cleaned_data['PreviousTransactionDate']
    ).dt.total_seconds()
    cleaned_data = cleaned_data.drop(columns=['TransactionDate', 'PreviousTransactionDate'])
    categorical_columns = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
    cleaned_data = pd.get_dummies(cleaned_data, columns=categorical_columns, drop_first=True)
    cleaned_data = cleaned_data.dropna()
    print("Data cleaned and features engineered successfully.")
    return cleaned_data

def train_model(data, contamination=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(data)
    print("Isolation Forest model trained.")
    return model

def detect_anomalies(model, data, original_data):
    predictions = model.predict(data)
    original_data['Anomaly'] = predictions
    print("Anomalies detected.")
    return original_data

def save_results(data, output_file='anomaly_results.csv'):
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def visualize_anomalies(data, feature1, feature2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data[feature1], 
        y=data[feature2], 
        hue=data['Anomaly'], 
        palette={1: 'blue', -1: 'red'},
        alpha=0.7
    )
    plt.title("Anomaly Detection in Transactions")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title="Anomaly")
    plt.show()

if __name__ == "__main__":
    filepath = 'bank_transactions.csv'
    raw_data = load_data(filepath)
    cleaned_data = clean_data(raw_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cleaned_data.drop(columns=['TransactionAmount']))
    model = train_model(scaled_data, contamination=0.05)
    results = detect_anomalies(model, scaled_data, cleaned_data)
    save_results(results)
    visualize_anomalies(results, feature1='TransactionAmount', feature2='AccountBalance')

