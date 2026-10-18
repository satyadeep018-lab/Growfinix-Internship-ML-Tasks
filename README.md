# Growfinix Technology: Machine Learning Internship (Month 2: Deep Learning, Audio and NLP)

Five end-to-end projects built for the Growfinix Technology Machine Learning internship. Each project trains on a real, public dataset, reports honest held-out results, and has a Streamlit demo app.

## Projects

| # | Project | Dataset | Model | Key result (held-out data) |
|---|---|---|---|---|
| 1 | Spoken Emotion Recognition | RAVDESS speech (angry, happy, sad; 576 clips) | CNN on mel spectrograms (TensorFlow/Keras, librosa) | Accuracy 71.5% on 6 unseen actors |
| 2 | Spam and Phishing Email Detection | Kaggle Phishing Email Dataset (82,078 emails) | Bidirectional LSTM (PyTorch, NLTK) | Accuracy 98.4% (simple baseline 98.2%) |
| 3 | Assembly Line Defect Detection | Kaggle casting product images (about 1,300 photos) | ResNet-18 transfer learning (PyTorch) | Accuracy 99.2%, defect recall 98.7% |
| 4 | Collaborative Filtering Movie Recommender | MovieLens ml-latest-small (100k ratings) | SVD matrix factorization (Scikit-Surprise) | RMSE 0.8828 (bias baseline 0.8785) |
| 5 | Automating Machine Learning (AutoML) | scikit-learn breast cancer dataset (569 patients) | PyCaret compare_models (13 models) | Best: Extra Trees Classifier, accuracy 95.6%, F1 0.966 |

Each task folder has its own README with the method, results table and limitations.

## Repository structure

    Growfinix-Internship-ML-Tasks/
      task_1_audio_emotion/       app.py, features.py, train_emotion.py, models/, outputs/, samples/
      task_2_phishing_nlp/        app.py, text_utils.py, train_phishing.py, models/, outputs/
      task_3_defect_vision/       app.py, train_defect.py, models/, outputs/, samples/
      task_4_movie_recommender/   app.py, movielens.py, train_surprise.py, models/, outputs/
      task_5_automl_pipeline/     app.py, train_automl.py, models/, outputs/
      .python-version             Python 3.11
      README.md

Large datasets are not stored in the repo. Each task README says where to download its data.

## Run a project

Use Python 3.11. Every task has its own `requirements.txt`:

    cd task_N_name
    pip install -r requirements.txt
    streamlit run app.py

To retrain, download the dataset described in that task's README and run its training script.

## Honest notes

- Task 1: about 71% accuracy on speakers the model never saw. The test set is small (144 clips) and the emotions are acted, so real-world accuracy would likely be lower.
- Task 2: the LSTM beats a simple TF-IDF baseline by only a small margin, so the dataset is fairly easy, and it mixes sources with different writing styles.
- Task 3: very uniform photos of a single part, so real production lines would be harder. The test set is small (260 images).
- Task 4: on this small dataset the bias baseline is about as good as SVD in RMSE (SVD is slightly better in MAE).
- Task 5: 569 rows is small, so the scores are good but not definitive.
- All five are learning projects and not production systems.

## Datasets and credits

RAVDESS (Livingstone and Russo, 2018), Kaggle Phishing Email Dataset (Al-Subaiey et al., 2024), Kaggle Real-life Industrial Dataset of Casting Product (Dabhi), MovieLens (GroupLens), scikit-learn breast cancer dataset. Please check each dataset's license on its source page.
