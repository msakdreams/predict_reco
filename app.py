import os
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.ensemble import RandomForestRegressor
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
socketio = SocketIO(app)

def predict_and_append(df):
    good_data = df[df.iloc[:, 5] == 1]

    X_train = good_data.iloc[:, 0:4]
    y_train = good_data.iloc[:, 4]

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict for ALL rows using full features
    X_all = df.iloc[:, 0:4]
    df['Predicted Output'] = model.predict(X_all)
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
