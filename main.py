from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        P1 = request.form['P1']
        P2 = request.form['P2']
        P3 = request.form['P3']
        P4 = request.form['P4']
        P5 = request.form['P5']
        P6 = request.form['P6']
        P7 = request.form['P7']
        P8 = request.form['P8']
        P9 = request.form['P9']
        P10 = request.form['P10']
        P11 = request.form['P11']
        P12 = request.form['P12']
        P13 = request.form['P13']
        P14 = request.form['P14']
        P15 = request.form['P15']
        P16 = request.form['P16']
        P17 = request.form['P17']
        P18 = request.form['P18']
        P19 = request.form['P19']
        P20 = request.form['P20']
        P21 = request.form['P21']
        P22 = request.form['P22']
        P23 = request.form['P23']
        P24 = request.form['P24']
        P25 = request.form['P25']
        P26 = request.form['P26']
        model_choice = request.form['model_choice']

        sample_data = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23,
                       P24, P25, P26]

        ex1=np.array(sample_data).reshape(1,-1)

        if model_choice == 'weighted_random_forest':
            rf = pickle.load(open('rf', 'rb'))
            result_prediction = rf.predict(ex1)
        elif model_choice == 'decision_tree_adasyn':
            dt = pickle.load(open('dt', 'rb'))
            result_prediction = dt.predict(ex1)

    return render_template('predict.html', result_prediction = result_prediction, model_selected=model_choice)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9010)
