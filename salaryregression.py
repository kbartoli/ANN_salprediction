import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime


data = pd.read_csv("Churn_Modelling.csv")
#print(data.head())

# Preprocess the data
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

# One-hot encode geo
onehot_encoder_geo = OneHotEncoder(handle_unknown="ignore")
geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

#Combine one-hot encoded columns with original data
data = pd.concat([data.drop("Geography",axis=1), geo_encoded_df], axis=1)
#print(data.head())

# Split data into features and target
X = data.drop("EstimatedSalary", axis=1)
y = data["EstimatedSalary"]

# save files in the pickle so we can reuse (encoders and scaler)
with open("label_encoder_gender2.pkl", "wb") as file:
    pickle.dump(label_encoder_gender,file)

with open("label_encoder_geo2.pkl", "wb") as file:
    pickle.dump(onehot_encoder_geo,file)

## Split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## Scale these features
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)

with open("scaler2.pkl", "wb") as file:
    pickle.dump(scaler,file)

## Build the ANN model

model = Sequential([
    Dense(64,activation="relu",input_shape=(X_train.shape[1],)),  ## first HL connected with input layer
    Dense(32,activation="relu"),  ## Input shape is not needed as it is connected to HL1
    Dense(1) ## O/P layer, since here we did not put "activation" it will automatically take regression
])

# compile model
model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mae"])
print(model.summary())

# Set up TensorBoard
log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d=%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights=True)

#Train model
history = model.fit(
    X_train, y_train, validation_data = (X_test, y_test),epochs=100, callbacks=[early_stopping_callback, tensorboard_callback]
)

# to check training data run in command: tensorboard --logdir regressionlogs/fit

#Evaluate model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

model.save("regression_model.keras")
