import pickle
import numpy as np
from flask import Flask, jsonify, request, render_template

flask_app = Flask(__name__)

model = pickle.load(open('bank_model.pkl', 'rb'))  

input_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    'marital_married', 'marital_single', 'housing_yes', 'loan_yes',
    'contact_telephone', 'poutcome_nonexistent', 'poutcome_success',
    'job_group_Business & Self-Employed', 'job_group_Housemaid',
    'job_group_Management & Administration', 'job_group_Non-Active Workforce',
    'job_group_Professional & Technical', 'education_group_Others',
    'education_group_Vocational/Professional & University'
]


@flask_app.route("/")
def Home():
    return render_template("file.html",input_features=input_features)

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    prediction_text = "Yes" if prediction[0] == 1 else "No"
    return render_template("file.html", prediction_text=prediction_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
