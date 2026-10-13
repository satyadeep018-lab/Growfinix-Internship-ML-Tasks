import io, os, ssl, zipfile, urllib.request
import certifi
import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_movielens():
    folder = os.path.join(DATA_DIR, "ml-latest-small")
    if not os.path.exists(os.path.join(folder, "ratings.csv")):
        os.makedirs(DATA_DIR, exist_ok=True)
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    return ratings, movies
