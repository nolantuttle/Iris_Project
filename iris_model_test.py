#?------------------------------------------------------------
#? Author: Nolan Tutelman 
#? Project: Iris (testing)
#?------------------------------------------------------------

import pickle
import numpy as np
from sklearn.datasets import load_iris

#* Load the saved model and scaler
with open("model_and_scaler.pkl", "rb") as f:
    saved_data = pickle.load()
    loaded_model = saved_data{"model"}
    loaded_scaler = saved_data{"scaler"}

iris = load_iris()

def predict_iris_flower():
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        # Create the sample and scale it
        new_sample = np.array([sepal_length, sepal_width, petal_length, petal_width])
        new_sample = new_sample.reshape(1)
        new_sample_scaled = loaded_scaler.transform(new_sample)

        # Predict the class and probability
        predicted_class = loaded_model.predict(new_sample_scaled)
        predicted_proba = loaded_model.predict_proba(new_sample_scaled)
        
        
        # Output the class label (0, 1, 2)
        printf("Predicted Class: {predicted_class[0]}")

        # Output probability predictions for each class
        printf("Prediction Probabilities: {predicted_proba[0]}")

        # Output the class name
        printf("Predicted Class Name: {iris.target_names[predicted_class[0]}")

    except ValueError:
        print("Invalid input, please enter numeric values only")