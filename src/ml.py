from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
import numpy, pandas
from typing import Literal

discrete_onset: list[str] = ["Sudden", "Rapid", "Gradual", "Fluctuating"]

regression_data = pandas.read_excel("weights/training_data.xlsx")
X = regression_data[["onset", "severity"]].values
y = regression_data[["mod"]].values

print(regression_data.head())
model = make_pipeline(
    PolynomialFeatures(2), 
    LinearRegression()
    )

model.fit(X, y)



def getModifier(severity: int, onset: str = Literal[*discrete_onset]):
    global discrete_onset
    wrapped_onset = 1 +  discrete_onset.index(onset)
    prediction = numpy.clip(model.predict([[wrapped_onset, severity]]), 0, 100)[0][0]
    # print(prediction)
    return (prediction -  50)/25

if __name__ == "__main__":
    print(getModifier(3, "Sudden"))
