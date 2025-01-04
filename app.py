from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('heart_risk_prediction_model.h5')
label_encoder = LabelEncoder()
label_encoder.fit(['N', 'Q', 'SVEB', 'VEB', 'F'])  # Original arrhythmia types

# Initialize the scaler (Ensure it matches the scaler used during training)
scaler = MinMaxScaler()

# Risk level and precautions
risk_precautions = {
    'N': (10, "Low risk. Maintain a healthy lifestyle and regular check-ups."),
    'Q': (50, "Moderate risk. Monitor symptoms and consider lifestyle changes."),
    'SVEB': (70, "High risk. Avoid caffeine and stress, consult a cardiologist."),
    'VEB': (85, "Very high risk. Regular ECG monitoring and possible medication."),
    'F': (95, "Critical risk. Immediate medical attention and possibly advanced treatments.")
}

def preprocess_data(input_data):
    # Assume input_data is a DataFrame
    X = input_data.drop(columns=['record', 'type'], errors='ignore')
    X_scaled = scaler.fit_transform(X)  # Replace with scaler.transform in production
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    return X_scaled

def generate_report(predictions):
    # Decode predictions and create a report
    predicted_labels = np.argmax(predictions, axis=1)
    decoded_labels = label_encoder.inverse_transform(predicted_labels)
    report = []

    for i, label in enumerate(decoded_labels):
        risk_score, precaution = risk_precautions[label]
        confidence = predictions[i][predicted_labels[i]] * 100  # Model confidence
        report.append({
            "Arrhythmia Type": label,
            "Risk Score": f"{risk_score}/100",
            "Model Confidence": f"{confidence:.2f}%",
            "Precautions": precaution
        })
    
    return report

def create_chart(data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values(), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot to a bytes buffer and encode it as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'ecgData' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['ecgData']
    
    try:
        input_data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"File reading error: {str(e)}"}), 400

    # Verify column names and types
    expected_columns = [
        'record', 'type', '0_pre-RR', '0_post-RR', '0_pPeak', '0_tPeak', '0_rPeak', 
        '0_sPeak', '0_qPeak', '0_qrs_interval', '0_pq_interval', '0_qt_interval', 
        '0_st_interval', '0_qrs_morph0', '0_qrs_morph1', '0_qrs_morph2', '0_qrs_morph3', 
        '0_qrs_morph4', '1_pre-RR', '1_post-RR', '1_pPeak', '1_tPeak', '1_rPeak', 
        '1_sPeak', '1_qPeak', '1_qrs_interval', '1_pq_interval', '1_qt_interval', 
        '1_st_interval', '1_qrs_morph0', '1_qrs_morph1', '1_qrs_morph2', '1_qrs_morph3', 
        '1_qrs_morph4'
    ]
    missing_columns = [col for col in expected_columns if col not in input_data.columns]
    if missing_columns:
        return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

    try:
        X_processed = preprocess_data(input_data)
        predictions = model.predict(X_processed)
        report = generate_report(predictions)
        
        # Prepare data for charts
        risk_scores = {item["Arrhythmia Type"]: int(item["Risk Score"].split('/')[0]) for item in report}
        confidences = {item["Arrhythmia Type"]: float(item["Model Confidence"].split('%')[0]) for item in report}

        # Create and encode charts
        risk_chart_base64 = create_chart(risk_scores, "Risk Scores by Arrhythmia Type", "Arrhythmia Type", "Risk Score")
        confidence_chart_base64 = create_chart(confidences, "Confidence Levels by Arrhythmia Type", "Arrhythmia Type", "Confidence (%)")

        return jsonify({
            "report": report,
            "charts": {
                "risk_scores_chart": risk_chart_base64,
                "confidence_levels_chart": confidence_chart_base64
            }
        })
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
