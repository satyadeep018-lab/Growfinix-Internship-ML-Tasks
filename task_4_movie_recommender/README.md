# Task 4: Collaborative Filtering Movie Recommender

Part of the Growfinix Technology Machine Learning Internship (Month 2).

A Netflix-style recommender that suggests new movies to a user based only on their past ratings and the ratings of similar users, using matrix factorization (SVD) from Scikit-Surprise.

## Dataset
MovieLens `ml-latest-small` from GroupLens: about 100,000 real ratings from 610 users on roughly 9,700 movies. The data is not stored in this repo, because the MovieLens license does not allow redistribution. `movielens.py` downloads it automatically the first time you run the project.

## Method
- Ratings are split 80/20 at random (seed 42).
- Three models are compared on the held-out 20%: a random baseline, a bias baseline (average rating plus user and movie bias), and SVD matrix factorization (100 factors, 30 epochs).
- The final SVD model is retrained on all ratings and saved to `models/svd_model.pkl`.
- Recommendations: predict ratings for every movie the user has not seen and show the top 10. Only movies with at least 20 ratings are considered, so the picks are not dominated by obscure titles.

## Results
Error on the held-out ratings (lower is better):

| Model | RMSE | MAE |
|---|---|---|
| Random (baseline) | 1.4313 | 1.1427 |
| Bias baseline | 0.8785 | 0.6778 |
| SVD (matrix factorization) | 0.8828 | 0.6767 |

## Limitations
- RMSE measures how well ratings are predicted, not how good the top-10 ranking is.
- New users and new movies with no ratings cannot be handled (the cold-start problem).
- The saved model was retrained on all the data, so the table above comes from the models before the final retrain.

## Run it
Use Python 3.11. On Windows, Scikit-Surprise needs C++ build tools to install.

    pip install -r requirements.txt
    python train_surprise.py
    streamlit run app.py

## Files
- `movielens.py`: downloads and loads the dataset
- `train_surprise.py`: trains and compares the models
- `app.py`: Streamlit app
- `models/`: saved SVD model
- `outputs/`: model comparison table
