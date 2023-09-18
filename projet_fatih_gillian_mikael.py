import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, classification_report


pd.set_option('display.max_columns', None) # a check c'est quoi

df = pd.read_csv('./recipeData_clean.csv', encoding="latin1")

# Scatterplot pour mettre en avant => corrélation positive
plt.figure()
plt.scatter(df["OG"], df["ABV"])
plt.xlabel("OG")
plt.ylabel("ABV")
plt.title("La corrélation linéaire de OG et de ABV")
plt.show()

# BIN
bins = [0, 30, 60, 150]
labels = ["low", "medium", "high"]

df['bin_IBU'] = pd.cut(df['IBU'], bins, right=False)

# ONE HOT !

one_hot = pd.get_dummies(df["bin_IBU"])
np.sum(one_hot)
print(one_hot)
X = df[["OG"]]
y = df[["ABV"]]

# Regression Linear pour prédire AVB avec OG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RL = LinearRegression()
RL.fit(X_train,y_train)

y_predict = RL.predict(X_test)
MSE = 1/len(y_test) * np.sum((y_predict - y_test)**2)
print(MSE)


# Random Forest Classifier
X = df[["StyleID", "Efficiency", "OG", "BoilGravity", "ABV"]] # [[]]
y = one_hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=0)
param_grid = {"max_depth":[17, 18],"min_samples_leaf":[14,15],  "min_samples_split":[43,44], "max_features":[None]}

search = HalvingGridSearchCV(rf_classifier, param_grid, resource='n_estimators', max_resources=150,random_state=0, n_jobs=-1, verbose=2).fit(X_train, y_train)

print(search.best_params_)
rf_best =search.best_estimator_
y_pred = rf_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", "{0:.2f}".format(accuracy*100))

classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:", classification_report_result)
