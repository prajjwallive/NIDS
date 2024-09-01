from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

app = Flask(__name__, template_folder='templates')

# Paths to models and selected features
ml_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of 'ml_flask_app' directory

# Define paths for models and selected features
dt_model_path = os.path.join(ml_dir, 'dt_model.pkl')
knn_model_path = os.path.join(ml_dir, 'knn_model.pkl')
lr_model_path = os.path.join(ml_dir, 'lr_model.pkl')
selected_features_path = os.path.join(ml_dir, 'selected_features.pkl')

# Load models and selected features
dt_model = joblib.load(dt_model_path)
knn_model = joblib.load(knn_model_path)
lr_model = joblib.load(lr_model_path)
selected_features = joblib.load(selected_features_path)

# Ensure selected_features is a list
if not isinstance(selected_features, list):
    raise TypeError("selected_features should be a list of feature names.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                # Read the uploaded CSV file into a DataFrame
                data = pd.read_csv(file)
            except Exception as e:
                return render_template('index.html', error="Error reading CSV file. Please check the file format.")

            # Filter and reorder columns based on selected_features
            try:
                data = data[selected_features]
            except KeyError as e:
                return render_template('index.html', error=f"Error: {e}. Please check if all selected features are present in the CSV file.")

            # Make predictions using all three models
            dt_predictions = dt_model.predict(data)
            knn_predictions = knn_model.predict(data)
            lr_predictions = lr_model.predict(data)

            # Add predictions as new columns to the DataFrame
            data['dt_predictions'] = dt_predictions
            data['knn_predictions'] = knn_predictions
            data['lr_predictions'] = lr_predictions

            # Calculate total anomalies based on file size
            total_records = len(data)
            anomaly_rate = 45.93239886 / 100
            true_anomalies = int(total_records * anomaly_rate)

            # Check if true labels are provided
            if 'label' in data.columns:
                true_labels = data['label']

                # Calculate classification reports for each model
                dt_report = classification_report(true_labels, dt_predictions, output_dict=True)
                knn_report = classification_report(true_labels, knn_predictions, output_dict=True)
                lr_report = classification_report(true_labels, lr_predictions, output_dict=True)

                # Extract metrics for each model and round to 4 decimal places
                def extract_metrics(report):
                    metrics = {}
                    metrics['normal_precision'] = round(report['1']['precision'], 4)
                    metrics['normal_recall'] = round(report['1']['recall'], 4)
                    metrics['normal_f1'] = round(report['1']['f1-score'], 4)
                    metrics['anomaly_precision'] = round(report['0']['precision'], 4)
                    metrics['anomaly_recall'] = round(report['0']['recall'], 4)
                    metrics['anomaly_f1'] = round(report['0']['f1-score'], 4)
                    metrics['accuracy'] = round(report['accuracy'], 4)
                    return metrics

                dt_metrics = extract_metrics(dt_report)
                knn_metrics = extract_metrics(knn_report)
                lr_metrics = extract_metrics(lr_report)

            else:
                true_labels = None
                dt_metrics = knn_metrics = lr_metrics = {}

            # Calculate counts and percentages and round to 4 decimal places
            def calculate_counts(predictions, true_anomalies):
                total = len(predictions)
                normal_count = (predictions == 1).sum()
                anomaly_count = (predictions == 0).sum()
                normal_percentage = round((normal_count / total) * 100, 4)
                anomaly_percentage = round((anomaly_count / total) * 100, 4)
                anomaly_detection_percentage = round(abs(((anomaly_count - true_anomalies) / true_anomalies) * 100), 4) if true_anomalies else None
                return normal_count, anomaly_count, normal_percentage, anomaly_percentage, anomaly_detection_percentage

            dt_normal_count, dt_anomaly_count, dt_normal_percentage, dt_anomaly_percentage, dt_anomaly_detection_percentage = calculate_counts(dt_predictions, true_anomalies)
            knn_normal_count, knn_anomaly_count, knn_normal_percentage, knn_anomaly_percentage, knn_anomaly_detection_percentage = calculate_counts(knn_predictions, true_anomalies)
            lr_normal_count, lr_anomaly_count, lr_normal_percentage, lr_anomaly_percentage, lr_anomaly_detection_percentage = calculate_counts(lr_predictions, true_anomalies)

            # Render the results in the report.html template
            return render_template('report.html', 
                                   dt_normal_count=dt_normal_count, dt_anomaly_count=dt_anomaly_count, 
                                   dt_normal_percentage=dt_normal_percentage, dt_anomaly_percentage=dt_anomaly_percentage,
                                   knn_normal_count=knn_normal_count, knn_anomaly_count=knn_anomaly_count, 
                                   knn_normal_percentage=knn_normal_percentage, knn_anomaly_percentage=knn_anomaly_percentage,
                                   lr_normal_count=lr_normal_count, lr_anomaly_count=lr_anomaly_count, 
                                   lr_normal_percentage=lr_normal_percentage, lr_anomaly_percentage=lr_anomaly_percentage,
                                   dt_metrics=dt_metrics, knn_metrics=knn_metrics, lr_metrics=lr_metrics,
                                   dt_anomaly_detection_percentage=dt_anomaly_detection_percentage,
                                   knn_anomaly_detection_percentage=knn_anomaly_detection_percentage,
                                   lr_anomaly_detection_percentage=lr_anomaly_detection_percentage)

    return redirect(url_for('index.html'))

if __name__ == '__main__':
    app.run(debug=True)
