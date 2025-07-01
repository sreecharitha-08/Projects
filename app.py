import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open(r'E:\\project-1\\model.pkl', 'rb'))
scale = pickle.load(open(r'E:\\project-1\\encoder.pkl', 'rb'))  # should be StandardScaler trained on X only

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Extract and convert form values to float
        input_feature = [float(x) for x in request.form.values()]
        
        # Ensure correct number of features
        if len(input_feature) != 11:
            return f"Error: Expected 11 input values, got {len(input_feature)}."

        # Column names must match model training features
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
                 'hours', 'minutes', 'seconds']
        
        # Build DataFrame
        df = pd.DataFrame([input_feature], columns=names)

        # Apply scaling (transform only, not fit_transform)
        # df_scaled will be a NumPy array of shape (1, 11)
        df_scaled = scale.transform(df)

        # Predict
        # Explicitly ensure the input is a 2D NumPy array of shape (1, 11).
        # This step re-confirms the correct input format, even if df_scaled is already there.
        # This can sometimes help with subtle internal data handling issues within the model.
        X_input_for_predict = np.array(df_scaled).reshape(1, -1) #
        
        prediction = model.predict(X_input_for_predict) #
        
        result_text = "Estimated Traffic Volume is: " + str(int(prediction[0]))

        # Render output page
        return render_template("output.html", result=result_text)

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)