import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data from the table
fan_in = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
# instr_tupl_l2 = np.array([2.84, 5.66, 7.65, 9.92, 12.97, 18.77, 33.73, 81.97, 252.17, 924.28])
instr_tupl_l2 = np.array([2.87, 5.64, 7.54, 9.57, 11.7, 14.91, 20.48, 36.22, 89.11, 283.29])
tupl_per_circ_buffer_l2 = np.array([262133, 87373, 37441, 17468, 8449, 4153, 2056, 1020, 505, 249])

# Preparing data for linear regression
X = np.column_stack((fan_in, tupl_per_circ_buffer_l2))
y = instr_tupl_l2

# Performing linear regression
model = LinearRegression()
model.fit(X, y)

# Get coefficients and intercept
a, b = model.coef_
c = model.intercept_

# Model equation
equation = f"Instr./Tupl. MWay (L2) = {a:.2f} * Fan-in + {b:.2e} * Tupl. per Circ. Buffer MWay (L2) + {c:.2f}"

# Plotting the data and the model's prediction
plt.scatter(fan_in, instr_tupl_l2, color='blue', label="Data points")
plt.scatter(fan_in, model.predict(X), color='red', label="Fitted model")

plt.xlabel("Fan-in")
plt.ylabel("Instr./Tupl. MWay (L2)")
plt.title("Instr./Tupl. MWay (L2) vs Fan-in and Tupl. per Circ. Buffer MWay (L2)")
plt.legend()
plt.grid(True)
plt.show()

print(equation)
