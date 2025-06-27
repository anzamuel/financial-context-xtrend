# Financial Context X-Trend

Cross attention based, momentum inspired model for few-shot learning patterns in financial time-series.

TODO:
- the context lengths are currently unused by the LSTM!
- fine tune `CPD_THRESHOLD` and `MAX_CONTEXT_LEN` for each asset **@Vikram**
- remove MLE loss and only focus on Sharpe
- implement warmup steps
- check validation, early stopping, find good drouput etc. against overfitting
- check if attention is adding anything to the model by removing it and comparing sharpe
- check trades being made, trade frequency, rebalancing, etc...
- use framework for ML experiment tracking
