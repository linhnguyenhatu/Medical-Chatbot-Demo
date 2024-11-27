from flask import Flask, render_template, request,session
import numpy as np
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'

#dashboard
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')



#chatbot
dialog = []
@app.route('/chatbot', methods=['GET','POST'])

def chat():
    return render_template('chatbot.html',question = question, dialog=dialog)

if __name__ == '__main__':
    app.run(debug=True)
