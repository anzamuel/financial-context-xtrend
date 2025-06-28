# Financial Context X-Trend

Cross attention based, momentum inspired model for few-shot learning patterns in financial time-series.

TODO:
- the context lengths are currently unused by the LSTM!
- remove MLE loss and only focus on Sharpe
- implement warmup steps
- check validation
- use framework for ML experiments

IMPORTANT:
- only train on every 63 days (target)
- make it so that validation uses new contexts
- simplify the model architecture, reduce parameters, avoid overfitting
- learn `h_0` and `c_0` of the LSTMs on a per asset basis
- test the non-XTrend LSTM to have a baseline
