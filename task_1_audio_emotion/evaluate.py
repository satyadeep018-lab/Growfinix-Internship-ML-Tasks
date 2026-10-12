import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]
    print(classification_report(y_true, y_pred, target_names=["Angry", "Happy", "Sad"]))

if __name__ == "__main__":
    evaluate()
