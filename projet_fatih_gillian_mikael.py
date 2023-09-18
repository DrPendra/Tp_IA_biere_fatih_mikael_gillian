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
plt.scatter(df["ABV"], df["OG"])
plt.xlabel("ABV")
plt.ylabel("OG")
plt.title("La corrélation linéaire de ABV et de OG")
plt.show()

# BIN
bins = [0, 30, 60, 150]
labels = ["low", "medium", "high"]

df['bin_IBU'] = pd.cut(df['IBU'], bins, right=False)

# ONE HOT !

one_hot = pd.get_dummies(df["bin_IBU"])
np.sum(one_hot)
print(one_hot)
# X = pd.concat([one_hot, df[["feature1", "feature2"]]], axis=1)
X = df[["ABV"]] # [[]]
y = df[["OG"]]

#test corrélation avec IBU
#2D
'''
plt.figure()
plt.scatter(df["IBU"], df["OG"])
plt.xlabel("IBU")
plt.ylabel("OG")
#plt.title("La corrélation linéaire de IBU et de OG")
plt.show()
plt.close()

plt.figure()
plt.scatter(df["IBU"], df["ABV"])
plt.xlabel("IBU")
plt.ylabel("ABV")
#plt.title("La corrélation linéaire de IBU et de ABV")
plt.show()
plt.close()

plt.figure()
plt.scatter(df["IBU"], df["StyleID"])
plt.xlabel("IBU")
plt.ylabel("StyleID")
#plt.title("La corrélation linéaire de IBU et de StyleID")
plt.show()
plt.close()

#3D
plt.figure()
plt.scatter(df["ABV"], df["OG"], c=one_hot)
plt.xlabel("ABV")
plt.ylabel("OG")
#plt.title("La corrélation linéaire de ABV et de OG")
plt.show()
plt.close()

plt.figure()
plt.scatter(df["FG"], df["OG"], c=one_hot)
plt.xlabel("FG")
plt.ylabel("OG")
#plt.title("La corrélation linéaire de ABV et de OG")
plt.show()
plt.close()

plt.figure()
plt.scatter(df["StyleID"], df["BoilSize"], c=one_hot)
plt.xlabel("StyleID")
plt.ylabel("BoilSize")
#plt.title("La corrélation linéaire de ABV et de OG")
plt.show()
plt.close()
'''
# Regression Linear pour prédire AVB avec OG
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RL = LinearRegression()
RL.fit(X_train,y_train)

y_predict = RL.predict(X_test)
MSE = 1/len(y_test) * np.sum((y_predict - y_test)**2)
print(MSE)
MAE = 1/len(y_test) * np.sum(abs(y_predict - y_test))
print(MAE)


# Random Forest Classifier Linear pour prédire AVB avec OG
X = df[["StyleID", "Efficiency", "OG", "BoilGravity", "ABV"]] # [[]]
y = one_hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modèle Random Forest
rf_classifier = RandomForestClassifier(random_state=0)
param_grid = {"max_depth":[ 21,22, 23,24],"min_samples_leaf":[ 12, 13, 14],  "min_samples_split":[33, 35, 40], "max_features":[None]}

search = HalvingGridSearchCV(rf_classifier, param_grid, resource='n_estimators', max_resources=150,random_state=0, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(search.best_params_, search.best_estimator_)
rf_best =search.best_estimator_

rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:", classification_report_result)
