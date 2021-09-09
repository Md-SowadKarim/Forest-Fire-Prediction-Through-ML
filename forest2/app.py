from flask import *
import requests
import pickle
import numpy as np

app = Flask(__name__)
 
@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/temperature', methods=['POST'])
def temperature():
    zipcode = request.form['zip']
    r = requests.get('http://api.openweathermap.org/data/2.5/weather?zip='+zipcode+',us&appid=fd38d62aa4fe1a03d86eee91fcd69f6e')
    json_object = r.json()
    temp_k = float(json_object['main']['temp'])
    hum = float(json_object['main']['hum'])
    temp_c = (temp_k - 273.15) * 1.8 
    return render_template('forest_fire.html', temp=temp_c, humid=hum)


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=['temp','humid']
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction_text_all_pickles = 'The percentage of Forest-Fire Occurence is:\n'
    list_of_model_pickles = ['gnb.pkl', 'rfc.pkl','dtc.pkl'] # add any model pickle file here
    for model_file in list_of_model_pickles:
        f_pickle = open(model_file, 'rb')

        model = pickle.load(f_pickle)
        prediction=model.predict_proba(final)
        output='{0:.{1}f}'.format(prediction[0][1], 2)
        f_pickle.close()
        
        prediction_text_all_pickles += f' {output} % according to model in file {model_file}. \n'
    return render_template('forest_fire.html', pred=prediction_text_all_pickles)

# At this point (when the entire loop finishes processing all your pickle files) your prediction_text_all_pickles is correctly populated with required information.

    
if __name__ == '__main__':
    app.run(debug=True)


