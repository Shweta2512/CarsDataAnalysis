
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv("Cleaned_Car_data.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/services.html')
def services():
    return render_template('services.html')

@app.route('/price.html',methods=['GET','POST'])
def price():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0,'Select Company')
    return render_template('price.html', companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))

    return str(np.round(prediction[0],2))

@app.route('/report.html')
def report():
    return render_template('report.html')
@app.route('/data.html')
def data():
    return render_template('data.html')
@app.route('/bodystyle.html')
def bodystyle():
    return render_template('bodystyle.html')

@app.route('/safety.html')
def safety():
    return render_template('safety.html')

@app.route('/fueltype.html')
def fueltype():
    return render_template('fueltype.html')

@app.route('/drivelayout.html')
def drivelayout():
    return render_template('drivelayout.html')

if __name__ == "__main__":  
    app.run(debug=True)
    print("test")