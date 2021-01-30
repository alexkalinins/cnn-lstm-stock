# A CNN-LSTM Stock Prediction Algorithm

A deep learning model for predicting the next three closing prices of a stock, index, currency pair, etc. based on the past 10 days of trading history
(Open, High, Low, Close, Volume, Day of Week). Neural network architecture based on [this paper](https://doi.org/10.1155/2020/6622927) (Lu et al., 2020).


## Data

Data (for training and predicting) is sourced from Yahoo! Finance. Up to 10 years of daily information from 52 American and Canadian symbols was used to train the model. Each variable in each sequence of
10 days was standardized (z-score). The last 5% of the data was reserved for validation. The training data is not included in the git repository to save space. 


## Usage

You can use the project to predict the next three daily closing prices of a ticker listed on Yahoo! Finance using the following command:

```bash

python3 predict.py <TICKER>

```

**DISCLAIMER:** The predictions made by this algorithm should not be taken as financial advice. I am not responsible for any losses caused
as a result of using this model.

## TODO

This project is still ongoing. The accuracy of the prediction still leaves much to be desired. The project can still be improved:

 - [ ] Better data processing; z-score standardization may not be enough.
 - [ ] More balanced selection of training tickers; there are more bullish stocks than bearish.
 - [ ] Experimentation with different sequence lengths.
 - [ ] Experimentation with different architectures (might mean renaming the repo :wink:).
 - [ ] Plotting.

Contributions to the repo are very welcome.

## Implementation Details

The *current* model is implemented in `model1.py`. The architecture does not differ too much from Lu et al., but adjustments were made to improve the loss. There is still work to be done to improve the model. The following is the architecture in `model1.py`

 1. The input data consists of 10 time steps of 6 variables each.
 2. A 1D Convolution (kernel size of 1; 32 filters) is applied to the data. Each filter is akin to a dot product between a time step and some parameter vector.
 3. Swish activation function. Lu et al. use tanh, however tanh caused exploding gradients.
 4. Data passes through LSTM layer (64 hidden units). Swish activation function is also used.
 5. The last layer is a Dense layer to convert the output from the LSTM layer into a single number.
 6. The output is then destandardized.

This architecture was used to train a model for each prediction day (3 total). The trained models with the lowest validation loss (Mean Absolute Error) are included with the repo.
If you wish to add more tickers, or to update the model with newer data, you can train the model yourself by running `download-data.py` to download and preprocess data, and then running `model1.py`; making sure to specify `DAY` to select the prediction day.


## Built With:

 * TensorFlow
 * Pandas
 * NumPy
 * yfinance
 * Jupyter-Lab
 * TensorBoard

