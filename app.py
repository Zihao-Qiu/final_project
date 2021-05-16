import pandas as pd
import numpy as np
import joblib
from flask import Flask,render_template,redirect,session,url_for,request

app = Flask(__name__,template_folder='templates',static_folder='statics')
app.debug = True
app.secret_key='asides'
model = joblib.load("train_model.m")

cols = [ 'gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type','smoking_status', 'age','avg_glucose_level', 'bmi'
       ]

@app.route('/',methods=['GET'])
def original_dataset():
    df = pd.read_csv('train_strokes.csv')
    df = df.head(20)
    return render_template(
        "index.html",
        data = df.to_html(classes="data",index =False)
   )

@app.route('/requests/',methods=['POST','GET'])
def requests():
    data = pd.DataFrame([request.form.to_dict()],dtype=np.float)
    weight = data['weight'][0]
    height = data['height'][0]
    bmi = weight/(height*height)
    data['bmi'] = (bmi - 10.1)/(90.0 - 10.1)
    data.drop(columns='weight')
    data.drop(columns='height')
    data['age'] = (data['age'] - 10.0)/(82.0 - 10.0)
    data['avg_glucose_level'] = (data['avg_glucose_level'] - 55.01)/(291.05 - 55.01)
    data = data.reindex(columns=cols)
    y_prob = model.predict_proba(data.iloc[0:1,:].values)
    print(data.iloc[0:1,:])
    print(y_prob)
    return "the probability of this person getting stroke is %f" %(y_prob[:,1])


if __name__ == '__main__':
    app.run()