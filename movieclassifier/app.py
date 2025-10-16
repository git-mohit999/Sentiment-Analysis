from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from vectorizer import vect

app = Flask(__name__)

#loading clf
clf = pickle.load(open(os.path.join('C:/Users/Admin/Desktop/VS/movieclassifier', 'pkl_objects','classifier.pkl'),'rb'))

db = os.path.join('C:/Users/Admin/Desktop/VS/movieclassifier','review.sqlite')

def classify(document):
	label = {0:'neg', 1:'pos'}
	X =vect.transform([document])
	y = clf.predict(X)[0]
	proba = np.max(clf.predict_proba(X))
	return label[y],proba

def train(document,y):
	X = vect.transform([document])
	clf.partial_fit(X,[y])

def sqlite_entry(path,document, y):
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("INSERT INTO review_db (review, sentiment,date) VALUES (?,?, DATETIME('now'))", (document,y))

	conn.commit()
	conn.close()

class ReviewForm(Form):
	moviereview = TextAreaField('',validators = [validators.DataRequired(),validators.length(min = 15)])

@app.route('/')

def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form = form)

@app.route('/results', methods= ['POST'])

def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():	
		review = request.form['moviereview']
		y,proba = classify(review)
		return render_template('results.html', content = review, prediction = y, probability = round(proba*100,2))

	return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])

def feedback():
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']

	inv_label = {'neg':0, 'pos':1}
	y = inv_label[prediction]
	
	if feedback == 'Incorrect':
		y = int(not(y))
	train(review,y)
	sqlite_entry(db, review, y)
	
	return render_template('thanks.html')

# import update function from local dir
from update import update_model

if __name__ == '__main__':
	app.run(debug=True)
	clf = update_model(db_path=db, model=clf, batch_size=10000)
	
