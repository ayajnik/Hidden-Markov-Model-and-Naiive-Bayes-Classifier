import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('RIDGE.xlsx')

df['base_price'] = pd.to_numeric(df['base_price'])
df['original_price'] = pd.to_numeric(df['original_price'])
df['discount_amount'] = pd.to_numeric(df['discount_amount'])
df['price'] = pd.to_numeric(df['price'])

from sklearn.model_selection import train_test_split

X = df.drop('qty_ordered',axis=1)
y = df['qty_ordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion = 'entropy')

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

conf_matrix=confusion_matrix(y_test,predictions)
accuracy=accuracy_score(y_test,predictions)

conf_matrix,accuracy

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
import pickle
import pydotplus

features = list(df.columns[1:])
features

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())