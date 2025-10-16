import pickle
import sqlite3
import numpy as np
import os

from vectorizer import vect

def update_model(db_path, model, batch_size = 10000):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT * from review_db')
	
	results = c.fetchmany(batch_size)

	while results :
		data = np.array(results)
		X = data[:,0]
		Y = data[:,1].astype(int)

		classes = np.array([0,1])
		X_train = vect.transform(X)
		model.partial_fit(X_train, Y, classes = classes)
		results = c.fetchmany(batch_size)	

	conn.close()
	return model

clf = pickle.load(open(os.path.join('C:/Users/Admin/Desktop/VS/movieclassifier','pkl_objects','classifier.pkl'),'rb'))

db = os.path.join('C:/Users/Admin/Desktop/VS/movieclassifier', 'review.sqlite')

clf = update_model(db,clf,10000)

#perma-update model
pickle.dump(clf,open(os.path.join('C:/Users/Admin/Desktop/VS/movieclassifier','pkl_objects','classifier.pkl'), 'wb'),protocol = 4)