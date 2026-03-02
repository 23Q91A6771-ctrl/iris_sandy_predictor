import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    # 1. Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train a model (Logistic Regression here)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 4. Evaluate the model
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    # 5. Create the 'models' folder (if not exists)
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # 6. Save the model as a .pkl file
    model_path = os.path.join(save_dir, "iris_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved successfully at: {model_path}")

if __name__ == "__main__":
    main()