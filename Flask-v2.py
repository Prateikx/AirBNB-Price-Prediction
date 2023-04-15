# flask-v1.py is the most updated Version

from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mariadb
import pickle


le = LabelEncoder()
app = Flask(__name__)

# Load the trained model
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

# Load database configuration from environment variables
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'admin',
    'database': 'airbnb_new'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    try:
        conn = mariadb.connect(**config)
        cur = conn.cursor()
        sql = "SELECT * FROM new_data;"
        cur.execute(sql)
        data = cur.fetchall()
        conn.close()
        return render_template('data.html', data=data)
    except mariadb.Error as e:
        return jsonify(error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    data = {
        'room_type': request.form['room_type'],
        'accommodates': int(request.form['accommodates']),
        'bathrooms': int(request.form['bathrooms']),
        'cancellation_policy': request.form['cancellation_policy'],
        'cleaning_fee': request.form['cleaning_fee'],
        'instant_bookable': request.form['instant_bookable'],
        'bedrooms': int(request.form['bedrooms']),
        'beds': int(request.form['beds'])
    }

    try:
        conn = mariadb.connect(**config)
        cur = conn.cursor()
        sql = "INSERT INTO new_data (room_type, accommodates, bathrooms, cancellation_policy, cleaning_fee, instant_bookable, bedrooms, beds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        vals = tuple(data.values())
        cur.execute(sql, vals)
        conn.commit()

        cur.close()
        conn.close()
    except mariadb.Error as e:
        return jsonify(error=str(e)), 500

    df = pd.DataFrame.from_dict([data])
    odf = df.copy()

#Performing Label Encoding before passing into the model
    odf['room_type'] = le.fit_transform(odf['room_type'])
    odf['cancellation_policy'] = le.fit_transform(odf['cancellation_policy'])
    odf['cleaning_fee'] = le.fit_transform(odf['cleaning_fee'])
    odf['instant_bookable'] = le.fit_transform(odf['instant_bookable'])

    pred = model.predict(odf)
    pred = round(pred[0], 3)
    print('Prediction: ',pred)
    
    # Render the result page with the DataFrame as a HTML table
    return render_template('result.html', data=df.to_html(index=False), res=pred)


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True, port=8055)
