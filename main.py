import Predictor
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from PIL import Image, ImageTk

df=pd.read_csv("seattle-weather.csv")

root = tk.Tk()
root.title("Weather Prediction")
root.geometry("400x400")

image_path = "Weather.png"  
image = Image.open(image_path)
image = image.resize((200, 100)) 
weather_image = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=weather_image)
image_label.pack(pady=10)

heading_label = tk.Label(root, text="Weather Predictor", font=("Arial", 18))
heading_label.pack()

precepetation_label = tk.Label(root, text="Enter Precepetation (%):")
precepetation_label.pack()

entry_precepetation = tk.Entry(root)
entry_precepetation.pack()

temp_label = tk.Label(root, text="Enter Temperature (°C):")
temp_label.pack()

entry_temp = tk.Entry(root)
entry_temp.pack()

wind_label = tk.Label(root, text="Enter Wind Speed(km/h):")
wind_label.pack()

entry_wind = tk.Entry(root)
entry_wind.pack()


def predict_weather():
    precepetation = entry_precepetation.get()
    temperature = entry_temp.get()
    wind = entry_wind.get()

    print(f"Precepetation: {precepetation} %")
    print(f"Temperature: {temperature} °C")
    print(f"Wind: {wind} km/h")

    df['weather']=LabelEncoder().fit_transform(df['weather'])

    features=["precipitation", "temp_max", "wind"]
    X=df[features]
    y=df.weather
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    x = float(precepetation)
    y = float(temperature)
    z = float(wind)

    reg = Predictor.models()
    model = reg.model3()
    model.fit(train_X, train_y)
    pred = model.predict([[x,y,z]])
    realtime = int(pred)
    #print(realtime)

    if(realtime == 0):
        weather_prediction = "Drizzling"
    elif(realtime == 1):
        weather_prediction = "Fog"
    elif(realtime == 2):
        weather_prediction = "Rainy"
    elif(realtime == 3):
        weather_prediction = "Snow"
    elif(realtime == 4):
        weather_prediction = "Sunny"
    else:
        weather_prediction = "Invalid input!"

    print("Predicted weather: ", weather_prediction)
    output_label.config(text=f"Weather Prediction: {weather_prediction}")

predict_button = tk.Button(root, text="Predict Weather", command=predict_weather)
predict_button.pack(pady=10)

output_label = tk.Label(root, text="Predicted Weather: ", font=("Arial", 12))
output_label.pack()

root.mainloop()

