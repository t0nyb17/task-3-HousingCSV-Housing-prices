import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Housing.csv")

data = pd.get_dummies(data, drop_first=True)

data.to_csv("Housing_Updated.csv", index=False)

X = data.drop("price", axis=1) 
y = data["price"]               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.scatter(X_test['area'], y_test, color='blue', label='Actual')
plt.scatter(X_test['area'], predictions, color='red', label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {round(coef, 2)}")
