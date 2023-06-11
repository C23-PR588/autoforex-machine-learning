# Machine Learning Autoforex-App

Machine Learning for forecasting some currency into Indonesia Rupiah.

## Contributing
Team Machine Learning on :
- [Agung Rashif Madani M200DKX4537](https://www.linkedin.com/in/agung-rashif-madani-905b75222/)
- []()

## Dataset
We use API to get the Historical Data from [FreeCurrencyAPI](https://freecurrencyapi.com/)
There are two kind of forecast, 
1. Forecast a week which using per daily dataset(~3600 samples),
2. Forecast a month which using resample into per 3 days dataset(~1200 samples).

## Library
Library we use a lot to create the model

```tensorflow
matplotlib.pyplot
pandas
numpy
csv
os
```
## Exploratory Data Analysis
[EDA](https://github.com/C23-PR588/autoforex-machine-learning/blob/agung-madani/EDA_exploratory_data_analysis.ipynb)

## Forecasting Currency Exchange Rate to IDR
This forecasting has 10 features currency which is:
- 1: EUR
- 2: USD 
- 3: JPY
- 4: GBP
- 5: SGD
- 6: AUD
- 7: CNY
- 8: CAD 
- 9: MYR
- 10: RUB

We choose 2 best tensorflow models architecture from all possible models we trained. We also choose `Adam` as Optimizer and `Huber` as loss function.
First model is ANN 4 layers with Regularization:
```
def first_model(train_set, valid_set, window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=[window_size], activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1)
    ], name=f"1_model_{currency[0]}")

    # Get initial weights
    init_weights = model.get_weights()

    # Reset states generated by Keras
    tf.keras.backend.clear_session()

    # Reset the weights
    model.set_weights(init_weights)

    model.summary()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])

    # Train the model
    history = model.fit(train_set,
                          epochs=100,
                          verbose=1,
                          batch_size=batch_size,
                          validation_data=valid_set,
                          callbacks=[create_model_checkpoint()])
    model.evaluate(valid_set)
    plot_loss(history)
    
    return model
```
Second Model is ANN with 4 layers without regularization
```
def second_model(train_set, valid_set, window_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ], name=f"2_model_{currency[0]}")

    # Get initial weights
    init_weights = model.get_weights()

    # Reset states generated by Keras
    tf.keras.backend.clear_session()

    # Reset the weights
    model.set_weights(init_weights)

    model.summary()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                    optimizer=optimizer,
                    metrics=["mae"])

    # Train the model
    history = model.fit(train_set,
                          epochs=100,
                          verbose=1,
                          batch_size=batch_size,
                          validation_data=valid_set,
                          callbacks=[create_model_checkpoint()])
    model.evaluate(valid_set)
    plot_loss(history)
    
    return model
```
## Forecast a Week futures ahead
Below are MAE, MSE, RMSE, MAPE, MASE for each currency:

First Model:

![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/af85c3bf-dbbb-4801-952d-0dff0cb6d05b)

Second Model:

![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/74a13d86-2eed-47dc-a5d6-36297be9f13d)

Forecast:
#### EUR/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/6faf2dc0-85f4-45ca-bce1-88abace722d0)
#### USD/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/f62dab0b-ce36-4bfc-8463-97e10422d6e3)
#### JPY/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/519179af-068c-4ce7-ae12-b498018f4caf)
#### GBP/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/c4cf6bb1-43a5-46ff-b43f-94ee26861a17)
#### SGD/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/587b86dd-236d-4a7b-a118-a741f014bc1b)
#### AUD/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/82466601-d179-43b4-98c9-72a9b45f54a7)
#### CNY/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/330fa4b9-8dea-4196-bf43-8b75717960b4)
#### CAD/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/20153225-1596-4e3f-9b5f-f89cebddd4cf)
#### MYR/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/2fc59cf2-d4f1-4a17-a422-f8d0913d45eb)
#### RUB/IDR
![image](https://github.com/C23-PR588/autoforex-machine-learning/assets/121701309/79819003-8463-44f2-84c5-d0d1b384b129)

## Forecast a Month futures ahead
Below are MAE, MSE, RMSE, MAPE, MASE for each currency:

First Model:


Second Model:


Forecast:
#### EUR/IDR
#### USD/IDR
#### JPY/IDR
#### GBP/IDR
#### SGD/IDR
#### AUD/IDR
#### CNY/IDR
#### CAD/IDR
#### MYR/IDR
#### RUB/IDR
