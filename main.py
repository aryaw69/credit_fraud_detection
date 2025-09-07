import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv("data/creditcard.csv")

# Load dataset
data = pd.read_csv("data/creditcard.csv")

X = data.drop(["Class", "Time"], axis=1)  # all columns except Class and Time 
Y = data["Class"]              # target variable

print("Columns used for training:", X.columns.tolist())
print("Feature count:", X.shape[1])

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1,1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

joblib.dump(model, "fraud.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved !")