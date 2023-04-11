from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import mariadb
import pickle

le = LabelEncoder()
app = Flask(__name__) # template_folder='E:\Bignalytics\my_project\templates')

config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'admin',
    'database': 'airbnb_new'
}


# Load the trained model
with open('E:/Bignalytics/my_project/model_pickle', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    conn = mariadb.connect(**config)
    cur = conn.cursor()
    sql = "SELECT * FROM new_data;"
    cur.execute(sql)
    row_headers=[x[0] for x in cur.description]
    rv = cur.fetchall()
    json_data=[]
    for result in rv:
            json_data.append(dict(zip(row_headers,result)))
    return json.dumps(json_data)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    room_type = request.form['room_type']
    accommodates = int(request.form['accommodates'])
    bathrooms = int(request.form['bathrooms'])
    cancellation_policy = request.form['cancellation_policy']
    cleaning_fee = request.form['cleaning_fee']
    instant_bookable = request.form['instant_bookable']
    bedrooms = int(request.form['bedrooms'])
    beds = int(request.form['beds'])

    conn = mariadb.connect(**config)
    # create a connection cursor
    cur = conn.cursor()
    # execute a SQL statement
    sql = "insert into new_data (room_type, accommodates, bathrooms, cancellation_policy, cleaning_fee, instant_bookable, bedrooms, beds) VALUES (%s, %s, %s,%s, %s, %s,%s, %s)"
    vals = (room_type, accommodates, bathrooms, cancellation_policy, cleaning_fee , instant_bookable ,bedrooms ,beds)
    cur.execute(sql, vals)

    # execute a SELECT query and get the results as a DataFrame
    select = 'SELECT * FROM new_data ORDER BY id DESC LIMIT 2'
    df_sql = pd.read_sql(select, conn)
    
    conn.commit()
    # Close the database connection
    cur.close()
    conn.close()

    # print( df_sql) # It will print last 2 rows from the MariaDB database


    # Store the values in a dictionary
    data = {'room_type': room_type,
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'cancellation_policy': cancellation_policy,
            'cleaning_fee': cleaning_fee,
            'instant_bookable': instant_bookable,
            'bedrooms': bedrooms,
            'beds': beds}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict([data])
    
    cat_cols = ['room_type','cancellation_policy','cleaning_fee','instant_bookable']

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        odf = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=False)], axis=1)
        odf.drop(col, axis=1, inplace=True)
    

    pred = model.predict(odf)
    print('Prediction: ',pred)
    
    # Render the result page with the DataFrame as a HTML table
    return render_template('result.html', data=df.to_html(index=False), res=pred[0])


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True, port=8505)