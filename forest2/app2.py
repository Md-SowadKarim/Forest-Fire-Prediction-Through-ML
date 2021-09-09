from typing import Text
from flask import Flask, render_template, request
import requests
import pickle
import numpy as np 

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/temperature', methods=['POST'])
def temperature():
     zipcode = request.form['zip']
     r = requests.get('http://api.openweathermap.org/data/2.5/weather?zip='+zipcode+',us&appid=d4e79c50457271f3d2a04a892a5e2940')
     json_object = r.json()
     temp_k = float(json_object['main']['temp'])
     hum = float(json_object['main']['humidity'])
     
     # temp_k=int(temp_k)
     # hum=int(hum)
     int_features=[temp_k,hum]

     # #temp_k=str(temp_k)
     # hum=str(hum)
     # #temp_f = (temp_k - 273.15) * 1.8 + 32
     
     # #return render_template('temperature.html', temp=temp_f)
     # return hum
     final=[np.array(int_features)]
     print(final)
     prediction_text_all_pickles = 'The percentage of Forest-Fire Occurence is:\n'
     list_of_model_pickles = ['gnb.pkl', 'rfc.pkl','dtc.pkl'] # add any model pickle file here
     #list_of_model_pickles = ['gnb.pkl'] # add any model pickle file here

     for model_file in list_of_model_pickles:
          f_pickle = open(model_file, 'rb')

          model = pickle.load(f_pickle)
          prediction=model.predict_proba(final)
          #prediction=model.predict(final)
          output=['{0:.{1}f}'.format(prediction[0][1], 2)]
          f_pickle.close()
          
          prediction_text_all_pickles += f' {output} % according to model in file {model_file}. \n'
     #return render_template('forest_fire.html', pred=prediction_text_all_pickles)
     return output

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=['temp','humid']
    
    final=[np.array(int_features)]
    print(final)
    prediction_text_all_pickles = 'The percentage of Forest-Fire Occurence is:\n'
    #list_of_model_pickles = ['gnb.pkl', 'rfc.pkl','dtc.pkl'] # add any model pickle file here
    list_of_model_pickles = ['gnb.pkl'] # add any model pickle file here

    for model_file in list_of_model_pickles:
        f_pickle = open(model_file, 'rb')

        model = pickle.load(f_pickle)
        prediction=model.predict_proba(final)
        #prediction=model.predict(final)
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        f_pickle.close()
        
        prediction_text_all_pickles += f' {output} % according to model in file {model_file}. \n'
    #return render_template('forest_fire.html', pred=prediction_text_all_pickles)
        return output


if __name__ == '__main__':
    app.run(debug=True)