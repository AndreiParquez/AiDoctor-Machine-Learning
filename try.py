import pandas as pd
import pickle

# Load data from CSV file
df = pd.read_csv('try.csv')

model_filename = 'svm_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

# Load the SVM model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open(vectorizer_filename, 'rb') as file:
    loaded_vectorizer = pickle.load(file)

print("Model and vectorizer loaded successfully")

# Verify the model works by making a prediction
new_symptom = ["my throat is sore"]
new_symptom_tfidf = loaded_vectorizer.transform(new_symptom)
predicted_label = loaded_model.predict(new_symptom_tfidf)
print("Predicted label:", predicted_label)
