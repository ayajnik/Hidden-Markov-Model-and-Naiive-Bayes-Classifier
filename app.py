try:

    import os
    import numpy as np
    from flask import Flask, render_template,request,jsonify,redirect,url_for
    from flask import redirect, url_for, request, render_template, send_file
    import sqlite3
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    ##from flask_api import Form
    from flask_wtf.file import FileField
    from wtforms import SubmitField
    # import flask_api
    from flask_wtf import Form
    from io import BytesIO
    import sqlite3
    import pandas as pd
    import pickle
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.externals import joblib
    import sklearn
except:
    print("Some modules missing.")

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"   ##we use this whenever we use wt forms

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    ##when the button is pressed, it will give us a POST method
    if request.method == "POST":
        if form.validate_on_submit():
            file_name = form.file.data
            database(name=file_name.filename, data=file_name.read())
            ##print("FILE : ".format(file_name.filename))

            return render_template("result.html", form=form)

    return render_template("result.html", form=form)


@app.route('/codepage', methods=['GET','POST'])
def codepage():

    form = UploadForm()
    if request.method == "POST":
        if form.validate_on_submit():
            file_name_4 = form.file.data

            return redirect(url_for("result.html", form=form))

    return render_template("Generating_sentences.html", form=form)

class UploadForm(Form):
    file = FileField()
    submit = SubmitField("Submit")
    codepage = SubmitField("Page for Code.")

def database(name, data):
    ##giving the name of the connection
    conn = sqlite3.connect("Classification.db")
    ##printting the cursor
    cursor = conn.cursor()

    cursor.execute(""" CREATE TABLE IF NOT EXISTS my_data(name TEXT, data BLOP ) """)
    cursor.execute(""" INSERT INTO my_data(name, data) VALUES (?, ?)""",(name, data))

    conn.commit()
    cursor.close()
    conn.close()

def query():
    conn = sqlite3.connect("Classification.db")
    cursor = conn.cursor()
    print("IN DATABASE FUNCTION ")
    c = cursor.execute(""" SELECT * FROM  my_data """)

    for x in c.fetchall():
        name_v = x[0]
        data_v = x[1]
        break

    conn.commit()
    cursor.close()
    conn.close()

    return send_file(BytesIO(data_v), attachment_filename='flask.csv', as_attachment=True)


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("SalesData1.csv",error_bad_lines=False, encoding="UTF-8")

    #a = df['weight'].mean

    #a

    #df['weight'].fillna(a)

    types = {'simple': 1, 'bundle': 2, 'other': 3}
    df.product_type = [types[item] for item in df.product_type]

    df_x = df['name']
    df_y = df['product_type']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(df_x)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.50, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['name']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)