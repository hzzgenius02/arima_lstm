import arima
import lstm
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        return render_template('index.html', fangfa='#', daima='', yuce10=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        return render_template('index.html', fangfa='#', daima='', yuce10=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


@app.route('/login', methods=['POST', 'GET'])
def login():
    print('login')
    if request.method == 'POST':
        fangfa = request.form['fangfa']
        daima = request.form['daima']
        if fangfa == '1':
            print('ARIMA')
            yuce10 = arima.cal(daima.__str__())
            return render_template('index.html', fangfa='ARIMA', daima=daima, yuce10=yuce10)
        else:
            print('LSTM')
            yuce10 = lstm.cal(daima.__str__())
            return render_template('index.html', fangfa='LSTM', daima=daima, yuce10=yuce10)
    else:
        print('GET')
        return render_template('index.html', fangfa='#', daima='', yuce10=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    app.run(debug=True, port=5687)
