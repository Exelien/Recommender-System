from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["Jack John Jack", "John John Jack"]
cv = CountVectorizer()

countMatrix = cv.fit_transform(text)
similarityScore = cosine_similarity (countMatrix)

print (countMatrix.toarray())
print (similarityScore)
