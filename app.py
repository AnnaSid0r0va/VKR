import flask
from flask import render_template
import pickle
import sklearn
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = flask.Flask(__name__, template_folder="templates")

@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        with open('VKR.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        exp = float(flask.request.form['experience'])
        y_pred = loaded_model.predict([[exp]])

        return render_template('main.html', result=y_pred)
from app import app
if __name__ == '__main__':
    app.run()

gunicorn --workers 4 wsgi:app