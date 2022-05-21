from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app=Flask(__name__, template_folder='templates')

model = pickle.load(open('knn_pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/form_predict')
def form_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        out = 'Jenis Tumor Malignant'
    else:
        out = 'Jenis Tumor Benign'

    return render_template('result_predict.html', prediction_text='{}'.format(out))

if __name__=="__main__":
    app.run(debug=True)