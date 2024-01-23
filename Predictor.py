import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class models:
    def __init__(self):

        self.df=pd.read_csv("seattle-weather.csv")

        self.df['weather']=LabelEncoder().fit_transform(self.df['weather'])

        self.features=["precipitation", "temp_max", "temp_min" ,"wind"]
        X=self.df[self.features]
        y=self.df.weather
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y,random_state = 0)

    def model1(self):
        def DTR():
            return DecisionTreeRegressor(random_state=1)
        model = DTR()
        model.fit(self.train_X, self.train_y)
        pred1=model.predict(self.test_X)
        print("Mean Absolute Error(DecisionTree): %f" %(mean_absolute_error(self.test_y, pred1)))
        return DTR()
    
    def model2(self):
        def RFR():
            return RandomForestRegressor(random_state=1)
        model = RFR()
        model.fit(self.train_X, self.train_y)
        pred2=model.predict(self.test_X)
        print("Mean Absolute Error(RandomForest): %f" %(mean_absolute_error(self.test_y, pred2)))
        return RFR()

    def model3(self):  
        def XGB():  
            return XGBRegressor(n_estimators=100, learning_rate=0.04)
        model = XGB()
        model.fit(self.train_X, self.train_y)
        pred3=model.predict(self.test_X)
        print("Mean Absolute Error(XGB): %f" %(mean_absolute_error(self.test_y, pred3)))
        return XGB()


reg = models()
model1 = reg.model1()
model2 = reg.model2()
model3 = reg.model3()
