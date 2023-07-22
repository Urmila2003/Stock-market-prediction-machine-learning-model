# Stock-market-prediction-machine-learning-model

![image](https://github.com/Urmila2003/Stock-market-prediction-machine-learning-model/assets/109129599/4159f773-a137-4e82-a519-3f9c32fd8d31)

# Why do we need Stock Price Prediction?
The stock market is known for being volatile and hard to predict. Many researchers have attempted to use time-series data to forecast future stock prices, which could be highly profitable if successful. However, there are numerous variables that influence market fluctuations, and only a few can be accurately quantified, such as historical stock data, trading volume, and current prices. Other fundamental factors, like a company's intrinsic value, assets, performance, and strategies, can also impact investor confidence and stock prices, but they are difficult to incorporate into mathematical models. As a result, using machine learning for stock price prediction is challenging and can be unreliable. Additionally, unforeseen events, such as a pandemic or a war, can quickly impact the stock market, making accurate predictions even more difficult.
Rather than trying to accurately predict exact values, analysts typically focus on making short-term predictions that provide a probability estimate of the market's future performance. By using historical data and relevant features, mathematical and machine learning models can forecast short-term fluctuations in the market on an average day. However, predicting the impact of unexpected events, such as market-shattering news, is extremely challenging.

# Stock Price Prediction using Machine Learning
Using machine learning, stock price prediction involves forecasting the future value of stocks traded on stock exchanges to generate profits. Since there are numerous variables involved in predicting stock prices, achieving high accuracy is challenging, and that's where machine learning comes in handy.

# Downloading the Stock Prices Dataset
I downloaded the dataset from ('https://r.search.yahoo.com/_ylt=Awr.1hCiQrtklfsOyJdXNyoA;_ylu=Y29sbwNncTEEcG9zAzMEdnRpZANDQVEyNTUyM0NPXzEEc2VjA3Ny/RV=2/RE=1690022691/RO=10/RU=https%3a%2f%2fwww.kaggle.com%2fdatasets%2fjainshukal%2fnetflix-stock-price/RK=2/RS=c8C6.rhApXDgs2jhwJFR1wrRVxE-'). Once downloaded, the CSV file will include data for Open, High, Low, Close, Adj Close, and Volume for each date.

# Extracting the Stock Prices Dataset
We can use Pandas to load the downloaded CSV file into a DataFrame. Since each row corresponds to a specific date, we can index the DataFrame by the date column. The data we have collected spans from 22nd Feb 2022 to 22nd Feb 2023, which includes the unpredictable changes and after effects caused by the COVID-19 pandemic. This presents a challenge for our model to accurately predict stock prices during a period of significant market volatility.
Plotting the High and Low points of the market, we see the below graph.
![image](https://github.com/Urmila2003/Stock-market-prediction-machine-learning-model/assets/109129599/1d112dcc-e885-4d60-95c7-fa499559f300)

# Data Preprocessing
When working with real-world data in machine learning models, it's important to normalize or rescale the data to a fixed range. By doing so, we can prevent features with larger numeric values from dominating the model and introducing biases, which can help the model converge more quickly.
                                
# Train and Test Sets for Stock Price Prediction
After defining our custom function for splitting the data, we proceed to split it into training and testing sets. Since shuffling is not allowed in time-series datasets, we perform a simple split. To predict the current value, we take two steps worth of past data, meaning that the model considers yesterday's and today's values to predict today's closing price.

# Building the LSTM model
![image](https://github.com/Urmila2003/Stock-market-prediction-machine-learning-model/assets/109129599/ec9c60ee-1f0d-497b-8ef7-390882fb938b)

To construct a basic LSTM model with a single unit, we will utilize the Sequential and LSTM modules offered by Tensorflow Keras.

# Performance Evaluation on Test Set
To access the performance of the model, we initially plot the curve for the actual values and compare it to the curve for the predicted values.

From the comparison between the actual and predicted values, we can observe that the LSTM model is capable of reproducing the trends in the stock prices up to a certain degree. Furthermore, it appears to have accurately captured the recent decrease in prices.

# LSTM vs. Simple Moving Average vs. Exponential Moving Average for Stock Price Prediction

# Simple Moving Average
Test RMSE: 39.336
Test MAPE: 0.096
![image](https://github.com/Urmila2003/Stock-market-prediction-machine-learning-model/assets/109129599/75ed5c4b-8120-42a6-be4c-2ccd21f0bb07)

# Exponential Moving Average
Instead of creating our own implementation of EMA, we can use the SimpleExpSmoothing module provided by statsmodels in Python. By adjusting the smoothing_level parameter, we can fine-tune the model to achieve optimal performance. We observed that a lower value for smoothing_level resulted in better results.
Test RMSE: 37.475
Test MAPE: 0.092
Comparing the performance of SMA and EMA with LSTM in predicting stock price data, we find that the latter significantly outperforms the former. Hyperparameter tuning, including adjusting the number of cells, batch size, and loss function, could further improve the LSTM model's performance.
