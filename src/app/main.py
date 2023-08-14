from prophet import Prophet
from pmdarima import auto_arima  # SARIMA Model
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import datetime
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


from src.features.build_lstm import *
import src.models.lstm_predict as lstm_predict

pd.set_option('display.max_colwidth', 200)


# Hide table row index
hide_table_row_index_style = """
    <style>
        thead tr th:first-child {
            display:none
        }
        tbody th {
            display:none
        }
    </style>
"""
st.markdown(hide_table_row_index_style, unsafe_allow_html=True)

# Set web app title
st.title('Apziva - ValueInvestor')

# Sidebar
# Github
# st.sidebar.subheader('Github')
# st.sidebar.write("[readme.md](https://github.com/Ahmant/apziva-potential-talents/tree/main#readme)")

MODELS = ['SARIMA', 'Prophet', 'LSTM', 'Single Exponential Smoothing']

def get_results():
    # ------------
    # Get datasets
    # ------------
    yahoo_data = yf.Ticker(st.session_state.ticker)

    df = yahoo_data.history(start=st.session_state.ds_start_date, end=st.session_state.ds_end_date).dropna()
    test_days_count = st.session_state.ds_test_days_count
    df_train = df[:-test_days_count] if test_days_count > 0 else df
    df_test = df[-test_days_count:] if test_days_count > 0 else pd.DataFrame()

    forecast_steps = test_days_count

    with results_expander_datasets:
        if len(df) > 0:
            results_expander_datasets.subheader('Dataset')
            results_expander_datasets.dataframe(df)

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(df[st.session_state.ds_prediction_column])
            ax.set_xlabel('Date')
            ax.set_ylabel(st.session_state.ds_prediction_column)
            results_expander_datasets.pyplot(fig)

    # ------------
    # Models
    # ------------
    with results_expander_models:
        tabs = results_expander_models.tabs(MODELS)

        # SARIMA
        with tabs[0]:
            if st.session_state.models_enabled_sarima is not True:
                st.write('Model is disabled')
            else:
                st.header("SARIMA")
                sarima_model = auto_arima(
                    df_train[st.session_state.ds_prediction_column],
                    seasonal=False,
                    error_ignore='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    approximation=False
                )

                # Extract information from the SARIMA model object
                st.subheader("Summary")
                sarima_model_table_1 = pd.read_html(
                    sarima_model.summary().tables[0].as_html(), header=None, index_col=0)[0]
                sarima_model_table_2 = pd.read_html(
                    sarima_model.summary().tables[1].as_html(), header=None, index_col=0)[0]
                sarima_model_table_3 = pd.read_html(
                    sarima_model.summary().tables[2].as_html(), header=None, index_col=0)[0]
                st.table(sarima_model_table_1)
                st.table(sarima_model_table_2)
                st.table(sarima_model_table_3)

                # Predictions
                sarima_forecast = sarima_model.predict(
                    n_periods=forecast_steps, return_conf_int=True, alpha=0.05)
                sarima_forecast = [
                    pd.DataFrame(sarima_forecast[0], columns=[
                                'prediction']).reset_index(),
                    pd.DataFrame(sarima_forecast[1], columns=[
                                'lower_95', 'upper_95'])
                ]
                sarima_forecast = pd.concat(sarima_forecast, axis=1)
                if len(df_test) > 0:
                    sarima_forecast = sarima_forecast.set_index(df_test.index)

                # Visualization
                fig, ax = plt.subplots(1, figsize=(12, 8))
                if len(df_test) > 0:
                    ax.plot(df_test[st.session_state.ds_prediction_column],
                            color='black', label='Actual')
                ax.plot(sarima_forecast['prediction'],
                        color='red', label='Prediction')
                ax.fill_between(
                    sarima_forecast.index, sarima_forecast['lower_95'], sarima_forecast['upper_95'], alpha=0.2, facecolor='red')
                ax.set(title='Stock Price - Actual vs Predicted',
                    xlabel='Date', ylabel=st.session_state.ds_prediction_column)
                plt.legend(loc='upper left')
                st.pyplot(fig)

        # Prophet
        with tabs[1]:
            if st.session_state.models_enabled_prophet is not True:
                st.write('Model is disabled')
            else:
                # Preprocess Dataset
                prophet_df_train = pd.DataFrame(
                    df_train, columns=[st.session_state.ds_prediction_column])
                prophet_df_train.rename(
                    columns={st.session_state.ds_prediction_column: 'y'}, inplace=True)
                prophet_df_train['ds'] = df_train.index
                prophet_df_train['ds'] = prophet_df_train['ds'].dt.tz_localize(
                    None)

                prophet_df_test = pd.DataFrame(
                    df_test, columns=[st.session_state.ds_prediction_column])
                prophet_df_test.rename(
                    columns={st.session_state.ds_prediction_column: 'y'}, inplace=True)
                prophet_df_test['ds'] = df_test.index
                prophet_df_test['ds'] = prophet_df_test['ds'].dt.tz_localize(None)

                # Model
                prophet_model = Prophet(daily_seasonality=True)
                prophet_model.fit(prophet_df_train)

                # Predictions
                prophet_future = prophet_df_test if len(
                    prophet_df_test) > 0 else prophet_model.make_future_dataframe(periods=forecast_steps)
                prophet_forecast = prophet_model.predict(prophet_future)

                # Visualization
                fig, ax = plt.subplots(1, figsize=(12, 8))
                if len(prophet_df_test) > 0:
                    ax.plot(
                        prophet_df_test['ds'], df_test[st.session_state.ds_prediction_column], color='black', label='Actual')
                ax.plot(prophet_forecast[-forecast_steps:]['ds'],
                        prophet_forecast[-forecast_steps:]['yhat'], color='red', label='Prediction')
                ax.set(title='Stock Price - Actual vs Predicted',
                    xlabel='Date', ylabel=st.session_state.ds_prediction_column)
                plt.legend(loc='upper left')
                st.pyplot(fig)

        # LSTM
        with tabs[2]:
            if st.session_state.models_enabled_lstm is not True:
                st.write('Model is disabled')
            else:
                WINDOW_SIZE = st.session_state.models_lstm_window_size
                HORIZON = st.session_state.models_lstm_horizon_size
                X_train, y_train = df_to_X_y(df_train[st.session_state.ds_prediction_column], WINDOW_SIZE, HORIZON)

                # Build the model
                lstm_model = Sequential()
                lstm_model.add(InputLayer((WINDOW_SIZE, 1)))
                lstm_model.add(LSTM(64))
                lstm_model.add(Dense(64, 'relu'))
                lstm_model.add(Dense(32, 'relu'))
                lstm_model.add(Dense(HORIZON, 'linear'))
                # lstm_model.summary()

                # Model Compile
                lstm_model.compile(
                    loss=MeanSquaredError(),
                    optimizer=Adam(learning_rate=0.001),
                    metrics=[RootMeanSquaredError()]
                )

                # Fit the model
                lstm_model.fit(
                    X_train,
                    y_train,
                    epochs=10,
                )

                # Predict
                x = np.array(df_train[-WINDOW_SIZE:][st.session_state.ds_prediction_column]).tolist()
                lstm_predictions = lstm_model.predict([x]).flatten()
                fig, ax = plt.subplots(1, figsize=(12, 8))
                if len(df_test) > 0:
                    _df_test = df_test[st.session_state.ds_prediction_column][:HORIZON]
                    ax.plot(_df_test, color='black', label='Actual')
                    ax.plot(_df_test.index, lstm_predictions, color='red', label='Prediction')
                else:
                    ax.plot(lstm_predictions, color='red', label='Prediction')
                plt.legend(loc='upper left')
                st.pyplot(fig)

        # RandomForest
        # with tabs[3]:
        #     if st.session_state.models_enabled_random_forest is not True:
        #         st.write('Model is disabled')
        #     else:
        #         # Build the model
        #         rf_model = RandomForestRegressor(
        #             n_estimators=200,
        #             min_samples_split=50,
        #             random_state=1
        #         )

        #         # Fit the model
        #         df_train_target = df_train.shift(1)[st.session_state.ds_prediction_column]
        #         rf_model.fit(
        #             df_train[st.session_state.ds_prediction_column],
        #             df_train_target
        #         )

                # # Get predictions
                # rf_predictions = []
                # for i in range(forecast_steps):
                #     rf_predictions.append(
                #         rf_model.predict(
                #             [df_test[st.session_state.ds_prediction_column][-1]]
                #             if len(rf_predictions) == 0
                #             else rf_predictions[-1]
                #         )
                #     )
                # print(rf_predictions)
                #     # rf_predictions = pd.Series(rf_predictions, index=df_test.index)
                # fig, ax = plt.subplots(1, figsize=(12, 8))
                # if len(df_test) > 0:
                #     _df_test = df_test[st.session_state.ds_prediction_column][:forecast_steps]
                #     ax.plot(_df_test, color='black', label='Actual')
                #     ax.plot(_df_test.index, rf_predictions, color='red', label='Prediction')
                # else:
                #     ax.plot(rf_predictions, color='red', label='Prediction')
                # plt.legend(loc='upper left')
                # st.pyplot(fig)

        # Single Exponential Smoothing
        with tabs[3]:
            if st.session_state.models_enabled_ses is not True:
                st.write('Model is disabled')
            else:
                ses_model = SimpleExpSmoothing(df_train[st.session_state.ds_prediction_column])
                ses_model_fitted = ses_model.fit()
                ses_predictions = ses_model_fitted.predict(start=len(df_train), end=len(df_train) + forecast_steps - 1)

                # Plot
                fig, ax = plt.subplots(1, figsize=(12, 8))
                if len(df_test) > 0:
                    _df_test = df_test[st.session_state.ds_prediction_column][:forecast_steps]
                    ax.plot(_df_test, color='black', label='Actual')
                    ax.plot(_df_test.index, ses_predictions, color='red', label='Prediction')
                else:
                    ax.plot(ses_predictions, color='red', label='Prediction')
                plt.legend(loc='upper left')
                st.pyplot(fig)



with st.form("my_form"):
    general_cols = st.columns(2)
    with general_cols[0]:
        st.selectbox('Ticker', ['GOOGL', 'TSLA', 'AAPL'], key='ticker')
    with general_cols[1]:
        st.selectbox('Prediction Column', [
                 'Close', 'Open', 'High', 'Low'], key='ds_prediction_column')

    # Dataset
    st.header('Dataset')
    ds_cols = st.columns(2)
    with ds_cols[0]:
        st.date_input("Start Date", key='ds_start_date',
                      value=datetime.datetime(2020, 1, 1))
    with ds_cols[1]:
        st.date_input("End Date", key='ds_end_date',
                      max_value=datetime.datetime.today())

    # Test Dataset
    st.number_input('Days to test on', min_value=0, max_value=30, value=10, step=1, key='ds_test_days_count')

    # Models Configurations
    st.header('Models Configurations')
    tabs = st.tabs(MODELS)
    with tabs[0]:
        st.checkbox('Is enabled?', key='models_enabled_sarima')
    with tabs[1]:
        st.checkbox('Is enabled?', key='models_enabled_prophet')
    with tabs[2]:
        st.checkbox('Is enabled?', key='models_enabled_lstm')
        
        cols = st.columns(2)
        with cols[0]:
            st.number_input('Window Size (Input)', min_value=0, max_value=60, value=10, step=1, key='models_lstm_window_size')
        with cols[1]:
            st.number_input('Horizon Size (Output)', min_value=0, max_value=10, value=10, step=1, key='models_lstm_horizon_size')

    # with tabs[3]:
    #     st.checkbox('Is enabled?', key='models_enabled_random_forest')

    with tabs[3]:
        st.checkbox('Is enabled?', key='models_enabled_ses')


    st.form_submit_button('Get Results', on_click=get_results)

results_expander_datasets =st.expander("Datasets")
results_expander_models =st.expander("Models")
