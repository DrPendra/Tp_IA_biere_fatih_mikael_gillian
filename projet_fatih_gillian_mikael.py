import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, classification_report


pd.set_option('display.max_columns', None) # a check c'est quoi

df = pd.read_csv('./recipeData.csv', encoding="latin1")

orig_leng = len(df)

df = df.drop(
 columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
          "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)
ser = df.isna().mean() * 100

# aze = df["Size(L)"].quantile(0.95)
df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]  #IBU max == 150 selon wikipedia & supérieur a zero
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

hist = df.hist(bins=50, log=True)

new_len = len(df)
# print(new_len / orig_leng)

one_hot = pd.get_dummies(df["BrewMethod"])
print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)

print(df)


plt.matshow(df.corr())
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=16)

#plt.show()
plt.close()
############### ^^"
# Scatterplot pour mettre en avant => corrélation positive
plt.figure()
plt.scatter(df["ABV"], df["OG"])
plt.xlabel("ABV")
plt.ylabel("OG")
plt.title("La corrélation linéaire de ABV et de OG")
#plt.show()
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

#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RL = LinearRegression()
RL.fit(X_train,y_train)

y_predict = RL.predict(X_test)
MSE = 1/len(y_test) * np.sum((y_predict - y_test)**2)
print(MSE)
MAE = 1/len(y_test) * np.sum(abs(y_predict - y_test))
print(MAE)
print(df[["FG"]])

# X = pd.concat([one_hot, df[["feature1", "feature2"]]], axis=1)
X = df[["StyleID", "Efficiency", "OG", "BoilSize", "ABV"]] # [[]]
y = one_hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modèle Random Forest
rf_classifier = RandomForestClassifier(random_state=0)
param_grid = {"criterion":["gini"], "max_depth":[20, 25, 30],"min_samples_leaf":[7, 10, 15],  "min_samples_split":[25, 30, 35], "max_features":[None]}

search = HalvingGridSearchCV(rf_classifier, param_grid, resource='n_estimators', max_resources=110,random_state=0, n_jobs=-1, verbose=2).fit(X_train, y_train)
print(search.best_params_, search.best_estimator_)
rf_best =search.best_estimator_

rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report_result = classification_report(y_test, y_pred)
print("Classification Report:", classification_report_result)
