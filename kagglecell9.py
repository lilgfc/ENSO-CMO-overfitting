import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import warnings

from darts import TimeSeries
from darts.models.forecasting.rnn_model import RNNModel as LSTMModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.models.forecasting.tft_model import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore") 

# daa loading
try:
    enso = pd.read_csv('/kaggle/input/cafeyenso/ENSO.csv')
    coffee = pd.read_csv('/kaggle/input/cafeyenso/CMOdata.txt', sep='\t', header=1)
    print("Successfully loaded data from Kaggle input.")
except FileNotFoundError:
    print("Kaggle paths not found. Trying local files...")
    try:
        enso = pd.read_csv('ENSO.csv')
        coffee = pd.read_csv('CMOdata.txt', sep='\t', header=1)
        print("Successfully loaded local files.")
    except Exception as e:
        print(f"Error loading files: {e}")
        raise e 
except Exception as e:
    print(f"An error occurred during loading: {e}")
    raise e

# data parsing
enso['Date'] = pd.to_datetime(enso['Date'], format='%m/%d/%Y')
coffee['Date'] = pd.to_datetime(coffee['Year'], format='%YM%m') 
coffee = coffee.rename(columns={'($/kg)': 'Price'})
df = pd.merge(coffee, enso[['Date', 'ONI']], on='Date', how='inner')
df = df.set_index('Date') 

# Create features
df['ElNino'] = (df['ONI'] > 0.5).astype(int)
df['LaNina'] = (df['ONI'] < -0.5).astype(int)
lag_months = 13
df['ElNino_Lagged'] = df['ElNino'].shift(lag_months)
df['LaNina_Lagged'] = df['LaNina'].shift(lag_months)
df = df.dropna()

# target and features
target_col = 'Price'
feature_cols = ['ElNino_Lagged', 'LaNina_Lagged']

# train test split
test_split_size = int(len(df) * 0.2) 
train_df = df.iloc[:-test_split_size]
test_df = df.iloc[-test_split_size:]
y_true = test_df[target_col]
predictions = {}

# arimax
y_train_arimax = train_df[target_col]
X_train_arimax = train_df[feature_cols]
X_test_arimax = test_df[feature_cols]

arimax_model = pm.auto_arima(y_train_arimax, 
                             exogenous=X_train_arimax,
                             start_p=1, start_q=1,
                             test='adf', max_p=3, max_q=3,
                             m=12, d=None, seasonal=True,
                             start_P=0, D=1, trace=False,
                             error_action='ignore', suppress_warnings=True, 
                             stepwise=True)    

y_pred_arimax = arimax_model.predict(n_periods=len(test_df), 
                                     exogenous=X_test_arimax)
predictions['ARIMAX'] = y_pred_arimax
print("ARIMAX model training complete.")


ts_target = TimeSeries.from_series(df[target_col], freq='MS')
ts_covariates_enso = TimeSeries.from_dataframe(df, value_cols=feature_cols, freq='MS')
ts_covariates_time = datetime_attribute_timeseries(ts_target, attribute='month', one_hot=True)
ts_covariates = ts_covariates_enso.stack(ts_covariates_time)

ts_target_train, ts_target_test = ts_target.split_before(ts_target.time_index[len(train_df)])

# Scale data
scaler_target = Scaler()
scaler_past_covs = Scaler()
scaler_future_covs = Scaler()

ts_target_train_scaled = scaler_target.fit_transform(ts_target_train)
ts_past_covs_scaled = scaler_past_covs.fit_transform(ts_covariates)
ts_future_covs_scaled = scaler_future_covs.fit_transform(ts_covariates)

# models
IN_LEN = 24 
OUT_LEN = 1
N_EPOCHS = 50 
pl_kwargs = {"accelerator": "auto", "enable_progress_bar": False} # Disables progress bars

model_lstm = LSTMModel(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, 
                       model='LSTM', n_epochs=N_EPOCHS, random_state=42, 
                       save_checkpoints=True, force_reset=True, pl_trainer_kwargs=pl_kwargs)

model_tcn = TCNModel(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, 
                     n_epochs=N_EPOCHS, random_state=42, 
                     save_checkpoints=True, force_reset=True, pl_trainer_kwargs=pl_kwargs)
                     
model_tft = TFTModel(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN, 
                     n_epochs=N_EPOCHS, random_state=42, 
                     save_checkpoints=True, force_reset=True, pl_trainer_kwargs=pl_kwargs)

# training nn
print("Training LSTM...")
model_lstm.fit(ts_target_train_scaled, 
               future_covariates=ts_future_covs_scaled,
               verbose=False)

print("Training TCN...")
model_tcn.fit(ts_target_train_scaled, 
              past_covariates=ts_past_covs_scaled,
              verbose=False)

print("Training TFT...")
model_tft.fit(ts_target_train_scaled, 
              future_covariates=ts_future_covs_scaled,
              verbose=False)

print("Neural network training complete.")

# prediction
n_predict = len(ts_target_test)

pred_lstm = model_lstm.predict(n=n_predict, 
                               series=ts_target_train_scaled, 
                               future_covariates=ts_future_covs_scaled)

pred_tcn = model_tcn.predict(n=n_predict, 
                             series=ts_target_train_scaled, 
                             past_covariates=ts_past_covs_scaled)

pred_tft = model_tft.predict(n=n_predict, 
                             series=ts_target_train_scaled, 
                             future_covariates=ts_future_covs_scaled)

def timeseries_to_series(ts):
    try:
        return ts.pd_dataframe().iloc[:, 0]
    except AttributeError:
        try:
            return ts.pd_series()
        except AttributeError:
            values = ts.values().flatten()
            index = pd.to_datetime(ts.time_index)
            return pd.Series(values, index=index)

predictions['LSTM'] = timeseries_to_series(scaler_target.inverse_transform(pred_lstm))
predictions['TCN'] = timeseries_to_series(scaler_target.inverse_transform(pred_tcn))
predictions['TFT'] = timeseries_to_series(scaler_target.inverse_transform(pred_tft))

y_true_aligned = y_true.loc[predictions['ARIMAX'].index]

results = {}
for model_name, y_pred in predictions.items():
    y_pred_aligned = y_pred.loc[y_true_aligned.index]
    
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned)) * 100
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    
    results[model_name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

results_df = pd.DataFrame(results).T
results_df = results_df[['MAE', 'RMSE', 'MAPE', 'R²']]

print("\n--- Forecasting Performance Metrics ---")
print(results_df.sort_values(by='RMSE'))
#graphs
plt.figure(figsize=(16, 9))

plt.plot(y_true_aligned.index, y_true_aligned, label='Actual Price', 
         color='black', linewidth=2.5, marker='o', markersize=4)

colors = {'ARIMAX': 'blue', 'LSTM': 'red', 'TCN': 'green', 'TFT': 'purple'}
for model_name, y_pred in predictions.items():
    y_pred_aligned = y_pred.loc[y_true_aligned.index]
    plt.plot(y_pred_aligned.index, y_pred_aligned, 
             label=f'{model_name} Forecast', 
             linestyle='--', alpha=0.9, marker='x', markersize=4,
             color=colors.get(model_name, 'gray'))

plt.title('Coffee Price Forecasting: Model Comparison', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price ($/kg)', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('model_forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Final Summary ---")
print(f"Best model by RMSE: {results_df['RMSE'].idxmin()}")
print(f"Best model by MAE: {results_df['MAE'].idxmin()}")
print(f"Best model by R²: {results_df['R²'].idxmax()}")
