from flask import Flask, request, jsonify, render_template
import re
import pandas as pd
from joblib import dump, load
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
columns = ["OG","StyleID","Efficiency","BoilGravity"]
columnIndexes = [2,0,10,9]
Lr = load('../model_Linear_Regression.joblib') 
Rf = load('../model_Random_Forest.joblib')
cat = ["Low","Medium","High"]

@app.route('/')
def index():
    return render_template('form.html',columns=columns)



@app.route('/query', methods=['POST'])
def traiter_requete_ajax():
    try:
        data = request.json
        dataForDf = {
            "OG":[float(data["OG"])],
            "StyleID":[int(round(float(data["StyleID"])))],
            "Efficiency":[float(data["Efficiency"])],
            "BoilGravity":[float(data["BoilGravity"])],
        }
        df = pd.DataFrame(dataForDf)
        
        abv = Lr.predict(df[["OG"]])
        df["ABV"] = abv

        ibu = Rf.predict(df[["StyleID", "Efficiency", "OG", "BoilGravity", "ABV"]])

        abv = "{:.4f}".format(abv[0,0])

        for i in range(len(ibu[0])):
            if ibu[0,i] == True:
                ibu = cat[i]
                break
            
        return jsonify({"abv":abv,"ibu":ibu}), 200
    except Exception as e:
        return jsonify({'error': 'Erreur lors du traitement des données'}), 400