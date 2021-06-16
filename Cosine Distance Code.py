from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris, London"]
cv = CountVectorizer()

countMatrix = cv.fit_transform(text)
similarityScore = cosine_similarity (countMatrix)

print (countMatrix.toarray())
print (similarityScore)
