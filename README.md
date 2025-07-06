# ClaimGuard: Insurance Fraud Detection System

A machine learning system for detecting anomalies in insurance claims using Gaussian Mixture Models and MLflow for experiment tracking.

## Overview

ClaimGuard is an intelligent fraud detection system designed to identify suspicious patterns in insurance claims data. The system processes inpatient medical claims data and uses unsupervised learning techniques to detect potential fraudulent activities.

## Features

- **Anomaly Detection**: Uses Gaussian Mixture Models to identify unusual claim patterns
- **MLflow Integration**: Comprehensive experiment tracking and model management
- **Data Processing Pipeline**: Automated data cleaning and preprocessing
- **Scalable Architecture**: Designed for large-scale insurance claim processing

## Project Structure

```
ClaimGuard/
├── data/
│   └── inpatient.csv          # Claims data
├── src/
│   ├── mypackage/
│   │   ├── __init__.py
│   │   ├── myclass.py         # Data loading and cleaning
│   │   └── dataprep.py        # Data preprocessing
│   └── train.py               # Model training script
├── MLproject.txt              # MLflow project configuration
├── conda.yaml                 # Environment dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MahammadRafi06/ClaimGuard.git
   cd ClaimGuard
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f conda.yaml
   conda activate mlflow-env
   ```

3. **Start MLflow tracking server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## Usage

### Training the Model

Run the training pipeline:
```bash
python src/train.py
```

### Using MLflow

1. **View experiments:**
   ```bash
   mlflow ui
   ```
   Navigate to `http://localhost:5000`

2. **Run with MLflow:**
   ```bash
   mlflow run . --experiment-name "claim-fraud-detection"
   ```

## Data Processing

The system processes insurance claim data through several stages:

1. **Data Loading**: Reads inpatient CSV data
2. **Date Processing**: Converts date columns to numeric format
3. **Feature Engineering**: Creates derived features like length of stay
4. **Encoding**: Processes categorical variables and diagnostic codes
5. **Dimensionality Reduction**: Uses PCA for feature reduction

## Model Details

- **Algorithm**: Gaussian Mixture Model with 5 components
- **Anomaly Detection**: Bottom 1% of density scores flagged as anomalies
- **Features**: Processed medical codes, patient demographics, and claim details

## Dependencies

- Python 3.10.15
- MLflow 2.17.2
- scikit-learn 1.5.2
- pandas 2.2.3
- numpy 2.1.3

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Author

**MahammadRafi**
- GitHub: [@MahammadRafi06](https://github.com/MahammadRafi06)
- Email: mrafi@uw.edu

## Acknowledgments

- MLflow for experiment tracking
- scikit-learn for machine learning algorithms
- Insurance industry data standards
