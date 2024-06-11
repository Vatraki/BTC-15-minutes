import json
import pandas as pd
import mt5_lib
import os
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the path to the settings file
setting_filepath = "settings.json"


# Function to load project settings from a JSON file
def get_project_settings(import_filepath):
    if os.path.exists(import_filepath):
        # Open and read the JSON file
        f = open(import_filepath, "r")
        project_settings = json.load(f)
        f.close()
        return project_settings
    else:
        # Raise an error if the file does not exist
        raise ImportError("Settings.json does not exist in location")


# Function to start up MetaTrader 5 with the given settings
def start_up(project_settings):
    # Initialize MT5 with provided settings
    startup = mt5_lib.start_mt5(project_settings=project_settings)
    symbols = project_settings['mt5']['symbols']
    if startup:
        print("Metatrader is connected successfully")
        # Initialize symbols in MT5
        init_symbol = mt5_lib.initialize_symbol(symbols)
        if init_symbol is True:
            print(f"Symbol {symbols} initialized")
        else:
            raise Exception(f"{symbols} not initialized")
        # Return True if startup is successful
        return True
    return False


if __name__ == '__main__':
    print("Let's build a bot")

    # Load project settings
    project_settings = get_project_settings(import_filepath=setting_filepath)

    # Start MT5 and initialize symbols
    startup = start_up(project_settings=project_settings)

    # Extract timeframe and symbols from settings
    timeframe = project_settings['mt5']['timeframe']
    symbol = project_settings['mt5']['symbols']

    # Get candlestick data
    df = mt5_lib.get_candlesticks(
        symbol=symbol,
        timeframe=timeframe,
        number_of_candles=1000)

    # List of discrete features for analysis
    discrete_features = [
        "RSI_Over_70", "RSI_Under_30", "VWAP_Cross_From_Above", "SUPER_UP",
        'HIGH', 'LOW', 'ADX_UP', 'ADX_DOWN', 'DOJI', 'Chaiken_Signal',
        "14EMA/21EMA_Cross, 14EMA>21EMA", 'MACD_Signal', "hh_signal",
        'lh_signal', 'll_signal', 'hl_signal', 'green body> 500',
        'red body> 500'
    ]

    # Calculate various indicators and add to dataframe
    # Function to calculate Higher highs, lower lows ...
    df = mt5_lib.HH_HL_LL_LH(df)
    df = mt5_lib.MACD2(df)
    df = mt5_lib.set_time_vwap(df)
    df = mt5_lib.set_prev(df)

    # Get supertrend data
    st, ut, dt = mt5_lib.get_supertrend2(df['high'], df['low'], df['close'], 14, 3)

    # Remove duplicate indices
    df = df[~df.index.duplicated()]
    ut = ut[~ut.index.duplicated()]
    dt = dt[~dt.index.duplicated()]
    st = st[~st.index.duplicated()]

    # Concatenate supertrend data with main dataframe
    df = pd.concat([df, ut, dt, st], axis=1)

    # Get resistance and support levels
    pivots = mt5_lib.get_resistance_support2(df)

    # Calculate distance to close support/resistance
    df = mt5_lib.distance_to_close_supres(pivots, df)

    # Add ADX data
    df = mt5_lib.ADX_data(df, 14)

    # Calculate Chaikin oscillator
    df['Chaikin'] = mt5_lib.chaikin_oscillator(df, fast_period=3, slow_period=10)

    # Add CCI, EMA, and other indicators
    df = mt5_lib.CCI(df)
    df = mt5_lib.EMA(df)
    df = mt5_lib.ATRMACD(df)
    df = mt5_lib.Signal(df)
    df = mt5_lib.Linreg(df)
    df = mt5_lib.German(df, 5000)
    df = mt5_lib.TPSL(df)

    # Print value counts for discrete features
    for i in discrete_features:
        print(df[i].value_counts())

    # Filter dataframe based on Bollinger Bands
    filtered_df = df[((df['Bollinger_Bands_Above_Upper_BB'] != 0) | (df['Bollinger_Bands_Below_Lower_BB'] != 0))]

    # Print exit signal counts
    print(df["Exit"].value_counts())

    # Split the data into training and testing sets
    t = 0.8
    split = int(t * len(filtered_df))

    # Prepare features and target variable
    discrete_X = filtered_df[discrete_features]
    discrete_y = filtered_df["Exit"]

    # Optimize random state for splitting data
    best_random_state = mt5_lib.optimize_random_state(discrete_X, discrete_y, split)

    # Split data into training and testing sets
    discrete_X_train = discrete_X.iloc[:split]
    discrete_y_train = discrete_y.iloc[:split]
    discrete_X_test = discrete_X.iloc[split:]
    discrete_y_test = discrete_y.iloc[split:]

    # Perform machine learning on the data
    df, rf_predictions_df, rf_predictions_df2 = mt5_lib.Machinelearning(
        df, filtered_df, discrete_X, discrete_y,
        discrete_X_train, discrete_y_train,
        discrete_X_test, discrete_y_test,
        split, best_random_state
    )

    # Print buy and hold returns
    print("Buy and hold returns =", round(list(df["daily returns"].iloc[split:].cumsum())[-1], 4) * 100, "%")

    # Print strategy returns
    print("Strategy returns =", round(list(rf_predictions_df["trading_algorithm_returns_new"].cumsum())[-1], 4) * 100,
          "%")

    # Backtest the strategy
    rf_predictions_df2 = mt5_lib.Backtest(rf_predictions_df2)