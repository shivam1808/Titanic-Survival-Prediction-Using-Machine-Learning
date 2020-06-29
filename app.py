from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

model = pickle.load(open('titanic_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    f1 = int(request.form['field1'])
    f2 = int(request.form['field2'])
    f3 = int(request.form['field3'])
    f4 = int(request.form['field4'])
    f5 = int(request.form['field5'])
    f6 = int(request.form['field6'])
    f7 = int(request.form['field7'])

        
    x = np.zeros(7)
    x[0] = f1
    x[1] = f2
    x[2] = f3
    x[3] = f4
    x[4] = f5
    x[5] = f6
    x[6] = f7

    output = model.predict([x])

    if output == 0:
        out = 'Oh no! You did not make it.'
    else:
        out = 'Nice! You Survived'
    
    return render_template('index.html', result='{}'.format(out))
    
if __name__ == "__main__":
    app.run(debug=True)