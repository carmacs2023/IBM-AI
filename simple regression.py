import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import requests
from sklearn import linear_model


def download(url, filename):
    # Send a GET request to the specified URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        with open(filename, 'wb') as f:
            f.write(response.content)


# path = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/"
#         "labs/Module%202/data/FuelConsumptionCo2.csv")
# file = "FuelConsumption.csv"
# download(path, file)

# File Structure

# MODELYEAR** e.g. 2014
# MAKE** e.g. Acura
# MODEL** e.g. ILX
# VEHICLE CLASS** e.g. SUV
# ENGINE SIZE** e.g. 4.7
# CYLINDERS** e.g 6
# TRANSMISSION** e.g. A6
# FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()
# summarize the data
print(df.describe())
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# print first 9 rows of the data
print(cdf.head(9))

viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

# Creating train and test dataset

msk = np.random.rand(len(df)) < 0.8     # 80% of the data will be used for training
train = cdf[msk]                        # 80% of the data
test = cdf[~msk]                        # 20% of the data

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
