from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your model and scaler
model = joblib.load('model.pkl')  # Replace with your model filename
scaler = joblib.load('scaler.pkl')  # Replace with your scaler filename

def create_plot(data, filename, feature_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='o', linestyle='-')
    plt.title(f'Sine Wave for {feature_name}')
    plt.xlabel('Sample Index')
    plt.ylabel(feature_name)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')  # This should point to your file for uploading

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Handle NaN values
        df.fillna(df.mean(), inplace=True)

        # Replace infinite values
        for column in df.select_dtypes(include=[np.float64, np.int64]).columns:
            df[column].replace([np.inf, -np.inf], df[column].max(), inplace=True)

        # Create interaction features
        # Assuming 'dx250', '_500dx', '_1000dx', and '_2000dx' are existing features in your DataFrame
        df['Interaction_1'] = df['dx250'] * df['_500dx']  # Example interaction
        df['Interaction_2'] = df['_1000dx'] + df['_2000dx']  # Example interaction

        # Extract expected feature names from the scaler
        expected_feature_names = scaler.get_feature_names_out()
        missing_features = [feature for feature in expected_feature_names if feature not in df.columns]

        if missing_features:
            return render_template('index.html', error=f'Missing features in input data: {", ".join(missing_features)}')

        # Prepare features for prediction
        X = df[expected_feature_names]

        if X.empty:
            return render_template('index.html', error='No features found for prediction.')

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        # Create a DataFrame for results
        results = pd.DataFrame(predictions, columns=['Decidera', 'Normoudente', 'Contratto'])

        # Calculate the percentage of normal hearing (assumed threshold for 'Normoudente')
        total_predictions = len(predictions)
        normal_count = sum(results['Normoudente'] > 0.5)  # Assuming >0.5 indicates normal hearing
        hearing_loss_count = total_predictions - normal_count

        normal_percentage = (normal_count / total_predictions) * 100
        hearing_loss_percentage = (hearing_loss_count / total_predictions) * 100

        # Create plots of the predictions
        plot_files = []
        for i, feature in enumerate(results.columns):
            plot_file = os.path.join(app.static_folder, f'prediction_plot_{i + 1}.png')
            create_plot(predictions[:, i], plot_file, feature)
            plot_files.append(f'prediction_plot_{i + 1}.png')

        return render_template('result.html', 
                               predictions=results.values.tolist()[:20],  # First 20 predictions
                               plots=plot_files,
                               full_predictions=results.values.tolist(),  # Full predictions for "See More"
                               normal_percentage=normal_percentage,
                               hearing_loss_percentage=hearing_loss_percentage)

    except Exception as e:
        return render_template('index.html', error=f'Error processing the file: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
