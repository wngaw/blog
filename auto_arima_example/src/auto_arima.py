import pandas as pd
from pyramid.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

# Import data
data = pd.read_csv("data/industrial_production.csv", index_col=0)

# Formatting
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

# Visualize
ax = data.plot()
fig = ax.get_figure()
fig.savefig("output/arima_raw_data_line_plot.png")

# Decomposition plot
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
fig.savefig("output/seasonal_decompose_plot.png")

# Perform Seasonal ARIMA
stepwise_model = auto_arima(data,
                            start_p=1, d=1, start_q=1,
                            max_p=1, max_d=1, max_q=1,
                            start_P=1, D=1, start_Q=1,
                            max_P=1, max_D=1, max_Q=1,
                            max_order=5, m=12,
                            seasonal=True, stationary=False,
                            information_criterion='aic',
                            alpha=0.05,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True,
                            n_jobs=-1,
                            maxiter=1)

print(f'best_aic: {stepwise_model.aic()}')

# Train Test Split
train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]

# Train
stepwise_model.fit(train)

# Evaluation
future_forecast = stepwise_model.predict(n_periods=30)
future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['prediction'])
test_data_evaluation = pd.concat([test, future_forecast], axis=1)

ax = test_data_evaluation.plot()
fig = ax.get_figure()
fig.savefig("output/arima_evaluation_test_data_line_plot.png")

full_data_evaluation = pd.concat([data, future_forecast], axis=1)
ax = full_data_evaluation.plot()
fig = ax.get_figure()
fig.savefig("output/arima_evaluation_full_data_line_plot.png")
