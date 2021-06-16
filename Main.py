# Note: Code is based off of Mihir Thakkar and Mosh's recommender systems.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def getTitleFromIndex (index):
	return df[df.index == index]["title"].values[0]

def getIndexFromTitle (title):
	return df[df.title == title]["index"].values[0]

df = pd.read_csv ("movieDataset.csv")

features = ['keywords','cast','genres','director']

for feature in features:
	df[feature] = df[feature].fillna('')

def combineFeatures(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)	

df["combinedFeatures"] = df.apply (combineFeatures, axis = 1 )

cv = CountVectorizer()

countMatrix = cv.fit_transform (df["combinedFeatures"])

cosineSim = cosine_similarity (countMatrix) 
movieUserLikes = "Avatar"

movieIndex = getIndexFromTitle (movieUserLikes)

similarMovies =  list (enumerate(cosineSim[movieIndex]))

sortedSimilarMovies = sorted (similarMovies, key=lambda x : x[1], reverse = True)

i = 0
for element in sortedSimilarMovies:
		print (getTitleFromIndex(element[0]))
		i = i+1
		if i > 50:
			break
