import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import mlflow
import mlflow.sklearn
from mypackage.myclass import LoadData
from mypackage.dataprep import DataPrep

# Set up MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("inpatient-app")
mlflow.autolog()  # Automatically logs parameters, metrics, and models for supported libraries

def load_data():
    """
    Function to load and preprocess the data.
    """
    load_class = LoadData()  # Assuming LoadData is already correctly implemented
    df = load_class.clean_data()  # Clean the data
    dataprep_class = DataPrep(df)  # Assuming DataPrep is the class to prepare data
    df = dataprep_class.prep()  # Prepare the data (e.g., scaling, imputing)
    return df

def processing(df):
    """
    Function to fit a Gaussian Mixture Model and log the results in MLflow.
    """
    # Fit the model
    gmm = GaussianMixture(n_components=5, n_init=10)
    gmm.fit(df)

    # Calculate density and identify anomalies
    densities = gmm.score_samples(df)
    density_threshold = np.percentile(densities, 1)
    anomalies = df[densities < density_threshold]
    anomalies = pd.DataFrame(anomalies)
    
    # Save and log the anomalies as an artifact (CSV file)
    anomalies.to_csv('inpatient_anomalies.csv', index=False)
    mlflow.log_artifact('inpatient_anomalies.csv')

    # Log the model
    mlflow.sklearn.log_model(gmm, "gmm_model")  # Logs the GMM model to MLflow

if __name__ == "__main__":
    # Start an MLflow run
    with mlflow.start_run():
        df = load_data()  # Load and prepare the data
        processing(df)  # Process data and log the model/artifacts
