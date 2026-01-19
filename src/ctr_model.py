from sklearn.linear_model import LogisticRegression
import joblib

def train_ctr_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
