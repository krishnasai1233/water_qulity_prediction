from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


app = Flask(__name__, template_folder='templates')


@app.route("/")
def index():
  return render_template('index.html')


@app.route("/pda")
def pda():
  return render_template('pda.html')

@app.route("/results", methods=['POST'])
def results():
  if request.method == 'POST':
    ph = float(request.form['ph'])
    Solids = float(request.form['Solids'])
    Hardness = float(request.form['Hardness'])
    Sulfate = float(request.form['Sulfate'])
    Chloramines = float(request.form['Chloramines'])
    Organic_carbon = float(request.form['Organic_carbon'])
    Potability = request.form['Potability']
    if Potability == 'Not_Suitable':
      Potability = 0
    else:
      Potability = 1


    dt = pickle.load(open('ML Models/decision_tree_model.pkl', 'rb'))
    knn = pickle.load(open('ML Models/knn_model.pkl', 'rb'))
    lr = pickle.load(open('ML Models/logistic_regression_model.pkl', 'rb'))
    rf = pickle.load(open('ML Models/random_forest_model.pkl', 'rb'))


    data = [[ph,Hardness,Solids,Chloramines,Sulfate,Organic_carbon]]

    dt_prediction = dt.predict(data)[0]
    knn_prediction = knn.predict(data)[0]
    lr_prediction = lr.predict(data)[0]
    rf_prediction = rf.predict(data)[0]

    y_true = [Potability]  
    dt_accuracy = accuracy_score([y_true], [dt_prediction])
    knn_accuracy = accuracy_score([y_true], [knn_prediction])
    lr_accuracy = accuracy_score([y_true], [lr_prediction])
    rf_accuracy = accuracy_score([y_true], [rf_prediction])
    
    if dt_prediction == 0:
      dt_prediction = 'Not Suitable for Drinking'
    else:
      dt_prediction = 'Suitable for Drinking'

    if lr_prediction == 0:
      lr_prediction = 'Not Suitable for Drinking'
    else:
      lr_prediction = 'Suitable for Drinking'

    if knn_prediction == 0:
      knn_prediction = 'Not Suitable for Drinking'
    else:
      knn_prediction = 'Suitable for Drinking'

    if rf_prediction == 0:
      rf_prediction = 'Not Suitable for Drinking'
    else:
      rf_prediction = 'Suitable for Drinking'
    


    return render_template('results.html', dt_prediction=dt_prediction, knn_prediction=knn_prediction, lr_prediction=lr_prediction, rf_prediction=rf_prediction, dt_accuracy=dt_accuracy, knn_accuracy=knn_accuracy, rf_accuracy=rf_accuracy, lr_accuracy=lr_accuracy)



if __name__ == '__main__':
    app.run(port=5000, debug=True)