import ast
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
movies = pd.read_csv("tmdb_5000_movies.csv")
credit = pd.read_csv("tmdb_5000_credits.csv")

print(movies.head())
print(credit.head())

movies = movies.merge(credit, on = "title")

movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]


def convert(text):
    converted = []
    for i in ast.literal_eval(text):
        converted.append(i["name"])
    return converted


movies.dropna(inplace = True)

movies["genres"] = movies["genres"].apply(convert)
print(movies.head())

movies["keywords"] = movies["keywords"].apply(convert)
print(movies.head())


ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


def convert3(text):
    converted = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            converted.append(i['name'])
        counter += 1
    return converted


movies['cast'] = movies['cast'].apply(convert)
movies.head()

movies['cast'] = movies['cast'].apply(lambda x: x[0:3])


def fetch_director(text):
    director = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            director.append(i['name'])
    return director


movies['crew'] = movies['crew'].apply(fetch_director)

# movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


def collapse(x):
    collapse_list = []
    for i in x:
        collapse_list.append(i.replace(" ", ""))
    return collapse_list


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies.head()

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
# new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


cv = CountVectorizer(max_features=5000, stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

# vector.shape

similarity = cosine_similarity(vector)

# similarity

new[new['title'] == 'The Lego Movie'].index[0]


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse = True, key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


recommend('Gandhi')

pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
