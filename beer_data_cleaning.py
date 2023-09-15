import pandas as pd
import numpy as np

# pip install plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go



df = pd.read_csv('./recipeData.csv', encoding="latin1")

df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)

###

df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[(df["IBU"] <= 150) & (df["IBU"] > 0)]  #IBU max == 150 selon wikipedia & supérieur a zero
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

colonnes = ["Size(L)", "OG", "FG", "IBU", "BoilSize", "ABV"]

fig = sp.make_subplots(rows=2, cols=3, subplot_titles=colonnes)

for i, col in enumerate(colonnes, 1):
    hist = go.Histogram(x=df[col], nbinsx=50, name=col)
    fig.add_trace(hist, row=(i-1)//3+1, col=i%3+1)

fig.update_layout(title='Histogrammes de nos différentes valeurs ')
fig.update_xaxes(title_text='Valeur', row=2, col=2)
fig.show()

# BINING !
bins = [0, 30, 60, 150]
labels = ["low", "medium", "high"]

df['bin_IBU'] = pd.cut(df['IBU'], bins, labels=labels,  right=False)

# PANDAS PROFILING

# from pandas_profiling import ProfileReport
# profile = ProfileReport(df, title='Rapport d\'analyse', explorative=True)
# profile.to_file("rapport_analyse.html")

#%%
