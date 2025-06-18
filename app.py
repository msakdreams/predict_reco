
import os
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
import shap
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app)

def predict_and_append(df):
    df.replace("", float("nan"), inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Separate known and unknown target rows
    train_data = df[df.iloc[:, -1].notna()]
    test_data = df[df.iloc[:, -1].isna()]

    X_train = train_data.iloc[:, 0:4]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, 0:4]

    # Impute missing values in features if necessary
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)

    # Predict missing values
    predicted_values = model.predict(X_test_imputed)

    # Save predicted values in a separate column (optional)

    df.loc[df.iloc[:, -1].isna(), 'Predicted_Value'] = predicted_values
    df.loc[df.iloc[:, 5].isna(), df.columns[5]] = predicted_values



    # Generate SHAP explanations
    explainer = shap.Explainer(model, X_train_imputed)
    shap_values = explainer(X_test_imputed)
    feature_names = df.columns[:4].tolist()

    explanations = []
    for row in shap_values:
        all_features = zip(feature_names, row.values)
        explanation = ", ".join([f"{name} contributed {value:.2f}" for name, value in all_features])
        # top_features = sorted(
        #     zip(feature_names, row.values),
        #     key=lambda x: abs(x[1]),
        #     reverse=True
        # )[:2]
        # explanation = ", ".join([f"{name} contributed {value:.2f}" for name, value in top_features])
        explanations.append(explanation)
    print(test_data.index)
    df.loc[test_data.index, 'Prediction_Explanation'] = explanations
    return df


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'excel' not in request.files:
            return "No file uploaded."

        file = request.files['excel']
        if file.filename == '':
            return "Empty file name."

        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        df = pd.read_excel(path)

        if df.shape[1] < 6:
            return "Excel file must have at least 6 columns."

        result_df = predict_and_append(df)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_' + filename)
        result_df.to_excel(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    return render_template('upload.html')





@socketio.on('connect')

def handle_connect():

    print('Client connected')

@socketio.on('disconnect')

def handle_disconnect():

    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000, debug=True)
    # socketio.run(app, debug=True)