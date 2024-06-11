import MetaTrader5  # Importing MetaTrader5 library for trading operations
import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations
from imblearn.under_sampling import RandomUnderSampler  # Importing undersampling technique from imbalanced-learn
from scipy import signal  # Importing signal processing functions from scipy
import datetime  # Importing datetime for date and time manipulation
from pandas import DataFrame  # Importing DataFrame directly from pandas
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from sklearn.metrics import classification_report  # Importing classification report from sklearn
from finta import TA  # Importing technical analysis functions from finta
from collections import deque  # Importing deque from collections for optimized list operations
from scipy.signal import argrelextrema  # Importing function to find local extrema from scipy
import pandas_ta as tap  # Importing pandas_ta for technical analysis
from sklearn import tree  # Importing decision tree from sklearn


def start_mt5(project_settings):
    """
    Initialize MetaTrader5 with given project settings.
    :param project_settings: Dictionary containing MetaTrader5 login details.
    :return: True if login is successful, otherwise False.
    """
    # Set variables
    username = project_settings['mt5']['username']
    username = int(username)
    password = project_settings['mt5']['password']
    server = project_settings['mt5']['server']
    mt5_pathway = project_settings['mt5']['mt5_pathway']

    mt5_init = False
    try:
        # Initialize MetaTrader5
        mt5_init = MetaTrader5.initialize(
            login=username,
            password=password,
            server=server,
            path=mt5_pathway
        )
    except Exception as e:
        print(f"Error to initialize:{e}")
        mt5_init = False

    mt5_login = False
    if mt5_init:
        try:
            # Login to MetaTrader5
            mt5_login = MetaTrader5.login(
                login=username,
                password=password,
                server=server,
                path=mt5_pathway
            )
        except Exception as e:
            print(f"Error to login{e}")
            mt5_login = False

    if mt5_login:
        return True
    return False

def initialize_symbol(symbol):
    """
    Initialize trading symbol in MetaTrader5.
    :param symbol: Trading symbol to initialize.
    :return: True if symbol is initialized, otherwise False.
    """
    all_symbols = MetaTrader5.symbols_get()
    symbol_names = [sym.name for sym in all_symbols]

    if symbol in symbol_names:
        try:
            # Select the symbol
            MetaTrader5.symbol_select(symbol, True)
            return True
        except Exception as e:
            print(f"Error enabling {symbol}. Error: {e}")
            return False
    else:
        print(f"Symbol {symbol} doesn't exist. The list is {symbol_names}")
        return False

def get_candlesticks(symbol, timeframe, number_of_candles):
    """
    Retrieve candlestick data for a given symbol and timeframe.
    :param symbol: Trading symbol.
    :param timeframe: Timeframe for the candlesticks.
    :param number_of_candles: Number of candles to retrieve.
    :return: DataFrame containing candlestick data.
    """
    if number_of_candles > 150000:
        raise ValueError("No more than 150000 candles can be retrieved")
    mt5_timeframe = set_query_timeframe(timeframe=timeframe)
    candles = MetaTrader5.copy_rates_from_pos(symbol, mt5_timeframe, 1, number_of_candles)
    dataframe = pd.DataFrame(candles)
    return dataframe

def set_query_timeframe(timeframe):
    """
    Set MetaTrader5 query timeframe based on input string.
    :param timeframe: Timeframe string.
    :return: Corresponding MetaTrader5 timeframe constant.
    """
    timeframes = {
        "M1": MetaTrader5.TIMEFRAME_M1,
        "M2": MetaTrader5.TIMEFRAME_M2,
        "M3": MetaTrader5.TIMEFRAME_M3,
        "M4": MetaTrader5.TIMEFRAME_M4,
        "M5": MetaTrader5.TIMEFRAME_M5,
        "M6": MetaTrader5.TIMEFRAME_M6,
        "M10": MetaTrader5.TIMEFRAME_M10,
        "M12": MetaTrader5.TIMEFRAME_M12,
        "M15": MetaTrader5.TIMEFRAME_M15,
        "M20": MetaTrader5.TIMEFRAME_M20,
        "M30": MetaTrader5.TIMEFRAME_M30,
        "H1": MetaTrader5.TIMEFRAME_H1,
        "H2": MetaTrader5.TIMEFRAME_H2,
        "H3": MetaTrader5.TIMEFRAME_H3,
        "H4": MetaTrader5.TIMEFRAME_H4,
        "H6": MetaTrader5.TIMEFRAME_H6,
        "H8": MetaTrader5.TIMEFRAME_H8,
        "H12": MetaTrader5.TIMEFRAME_H12,
        "D1": MetaTrader5.TIMEFRAME_D1,
        "W1": MetaTrader5.TIMEFRAME_W1,
        "MN1": MetaTrader5.TIMEFRAME_MN1,
    }
    if timeframe in timeframes:
        return timeframes[timeframe]
    else:
        print(f"Incorrect timeframe provided: {timeframe}")
        raise ValueError("Input incorrect timeframe")

def RSI(df, n):
    """
    Calculate Relative Strength Index (RSI) for a given DataFrame and period length.
    :param df: DataFrame containing price data.
    :param n: Period length for RSI calculation.
    :return: DataFrame with RSI values.
    """
    rsi = []
    diff = np.diff(df['prev_close'])

    for i in range(n):
        rsi.append(None)

    for i in range(len(diff) - n + 1):
        avgGain = diff[i:n + i]
        avgLoss = diff[i:n + i]
        avgGain = abs(sum(avgGain[avgGain >= 0]) / n)
        avgLoss = abs(sum(avgLoss[avgLoss < 0]) / n)

        if avgLoss == 0:
            rsi.append(100)
        elif avgGain == 0:
            rsi.append(0)
        else:
            rs = avgGain / avgLoss
            rsi.append(100 - (100 / (1 + rs)))

    df['RSI'] = rsi
    return df

def PROC(df, n):
    """
    Calculate Price Rate of Change (PROC) for a given DataFrame and period length.
    :param df: DataFrame containing price data.
    :param n: Period length for PROC calculation.
    :return: DataFrame with PROC values.
    """
    proc = []
    price = list(df['prev_close'])

    for i in range(n):
        proc.append(None)

    for i in range(len(price) - n):
        if len(price) <= n:
            proc.append(None)
        else:
            calculated = (price[i + n] - price[i]) / price[i]
            proc.append(calculated)

    df['PROC'] = proc
    return df

def SO(df, n):
    """
    Calculate Stochastic Oscillator (SO) for a given DataFrame and period length.
    :param df: DataFrame containing price data.
    :param n: Period length for SO calculation.
    :return: DataFrame with SO values.
    """
    so = []
    price = list(df['prev_close'])

    for i in range(n):
        so.append(None)

    for i in range(len(price) - n):
        C = price[i]
        H = max(price[i:i + n])
        L = min(price[i:i + n])
        so.append(100 * ((C - L) / (H - L)))

    df['SO'] = so
    return df

def Williams_R(df, n):
    """
    Calculate Williams %R for a given DataFrame and period length.
    :param df: DataFrame containing price data.
    :param n: Period length for Williams %R calculation.
    :return: DataFrame with Williams %R values.
    """
    wr = []
    price = list(df['prev_close'])

    for i in range(n):
        wr.append(None)

    for i in range(n - 1, len(price) - 1):
        C = price[i]
        H = max(price[i - n + 1:i])
        L = min(price[i - n + 1:i])
        wr_one = ((H - C) / (H - L)) * -100

        if wr_one <= -100:
            wr.append(-100)
        elif wr_one >= 100:
            wr.append(100)
        else:
            wr.append(wr_one)

    df['WR'] = wr
    return df


def calculate_targets(df, n):
    """
    Calculate target values for a given DataFrame and period length.
    :param df: DataFrame containing price data.
    :param n: Period length for target calculation.
    :return: DataFrame with target values.
    """
    targets = []
    price = list(df['prev_close'])

    for i in range(0, len(price) - n):
        targets.append(np.sign(price[i + n] - price[i]))

    for i in range(len(price) - n, len(price)):
        targets.append(None)

    df["Target({})".format(n)] = targets
    return df


def On_Balance_Volume(df):
    """
    Calculate On Balance Volume (OBV) for a given DataFrame.
    :param df: DataFrame containing price and volume data.
    :return: DataFrame with OBV values.
    """
    obv = []
    price = list(df['close'])
    volume = list(df['tick_volume'])
    obv.append(0)

    for i in range(1, len(price)):
        C_old = price[i - 1]
        C = price[i]

        if C > C_old:
            obv.append(obv[i - 1] + volume[i])
        elif C < C_old:
            obv.append(obv[i - 1] - volume[i])
        else:
            obv.append(obv[i - 1])

    df['OBV'] = obv
    return df


def calculate_ewma(df):
    """
    Calculate Exponential Weighted Moving Average (EWMA) for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with EWMA values.
    """
    df["EWMA"] = pd.Series.ewm(df['prev_close'], com=0.5, adjust=True, min_periods=0, ignore_na=False).mean()
    return df


def detrend(df):
    """
    Detrend price data for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with detrended price values.
    """
    trend = None
    price = list(df['prev_close'])

    if trend is None:
        trend = list(signal.detrend(price))
    else:
        trend.extend(signal.detrend(price))

    print("len(trend):{} len(df['Symbol']):{}".format(len(trend), len(price)))
    print("len(trend):{} len(df):{}".format(len(trend), len(df)))

    df['detrendedClose'] = trend
    return df


#def kelner(df, kc_lookback=20, multiplier=2, atr_lookback=10, plot=False):
    """
    Calculate Keltner Channels for a given DataFrame.
    :param df: DataFrame containing price data.
    :param kc_lookback: Lookback period for Keltner Channels.
    :param multiplier: Multiplier for Keltner Channels.
    :param atr_lookback: Lookback period for ATR calculation.
    :param plot: Boolean to indicate whether to plot the channels.
    :return: DataFrame with Keltner Channel values.
    """
    #df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['prev_high'], df['prev_low'], df['prev_close'],
                                                              #kc_lookback, multiplier, atr_lookback)
    #df['kc_signal'] = implement_kc_strategy(df['close'], df['kc_upper'], df['kc_lower'])
    #if plot:
    #    plot_kelner(df)
    #return df


def plot_kelner(df):
    """
    Plot Keltner Channels with close prices.
    :param df: DataFrame containing price and Keltner Channel data.
    """
    plt.plot(df['prev_close'], linewidth=2, label='INTC')
    plt.plot(df['kc_upper'], linewidth=2, color='orange', linestyle='--', label='KC UPPER 20')
    plt.plot(df['kc_middle'], linewidth=1.5, color='grey', label='KC MIDDLE 20')
    plt.plot(df['kc_lower'], linewidth=2, color='orange', linestyle='--', label='KC LOWER 20')
    plt.legend(loc='lower right')
    plt.title('INTC KELTNER CHANNEL 20 TRADING SIGNALS')
    plt.show()


def get_adx(high, low, close, lookback):
    """
    Calculate Average Directional Index (ADX) for a given DataFrame.
    :param high: Series containing high prices.
    :param low: Series containing low prices.
    :param close: Series containing close prices.
    :param lookback: Lookback period for ADX calculation.
    :return: Tuple containing plus DI, minus DI, and ADX values.
    """
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1 / lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1 / lookback).mean()
    return plus_di, minus_di, adx_smooth


def ADX_data_prev(df, adx_period=14, plot=False):
    """
    Calculate ADX data for a given DataFrame with previous period values.
    :param df: DataFrame containing price data.
    :param adx_period: Period for ADX calculation.
    :param plot: Boolean to indicate whether to plot the ADX data.
    :return: DataFrame with ADX values.
    """
    df['ADX'] = pd.DataFrame(get_adx(df['prev_high'], df['prev_low'], df['prev_close'], adx_period)[2]).rename(
        columns={0: 'ADX'})
    df['PLUS_DI'] = pd.DataFrame(get_adx(df['prev_high'], df['prev_low'], df['prev_close'], adx_period)[0]).rename(
        columns={0: 'PLUS_DI'})
    df['MINUS_DI'] = pd.DataFrame(get_adx(df['prev_high'], df['prev_low'], df['prev_close'], adx_period)[1]).rename(
        columns={0: 'MINUS_DI'})
    df['ADX_signal'] = pd.DataFrame(
        implement_adx_strategy(df['prev_close'], df['PLUS_DI'], df['MINUS_DI'], df['ADX'])).rename(
        columns={0: 'ADX_signal'})
    if plot:
        plot_adx(df)
    return df


def plot_adx_prev(df):
    """
    Plot ADX with close prices.
    :param df: DataFrame containing price and ADX data.
    """
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['prev_close'], linewidth=3, color='#ff9800', alpha=0.6)
    ax1.set_title('df CLOSING PRICE')
    ax2.plot(df['PLUS_DI'], color='#26a69a', label='+ DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['MINUS_DI'], color='#f44336', label='- DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['ADX'], color='#2196f3', label='ADX 14', linewidth=3)
    ax2.axhline(25, color='grey', linewidth=2, linestyle='--')
    ax2.legend()
    ax2.set_title('df ADX 14')
    plt.show()


def implement_adx_strategy(prices, pdi, ndi, adx):
    """
    Implement ADX trading strategy.
    :param prices: Series containing close prices.
    :param pdi: Series containing plus DI values.
    :param ndi: Series containing minus DI values.
    :param adx: Series containing ADX values.
    :return: List of trading signals based on ADX strategy.
    """
    adx_signal = [0]
    signal = 0

    for i in range(1, len(prices)):
        if adx[i - 1] < 25 and adx[i] > 25 and pdi[i] > ndi[i]:
            if signal != 1:
                signal = 1
                adx_signal.append(signal)
            else:
                adx_signal.append(signal)
        elif adx[i - 1] < 25 and adx[i] > 25 and ndi[i] > pdi[i]:
            if signal != -1:
                signal = -1
                adx_signal.append(signal)
            else:
                adx_signal.append(signal)
        else:
            adx_signal.append(adx_signal[i - 1])
    return adx_signal


def Supertrend(df, atr_period, multiplier):
    """
    Calculate Supertrend indicator for a given DataFrame.
    :param df: DataFrame containing price data.
    :param atr_period: Period for ATR calculation.
    :param multiplier: Multiplier for Supertrend calculation.
    :return: DataFrame with Supertrend values.
    """
    high = df['prev_high']
    low = df['prev_low']
    close = df['prev_close']

    # calculate ATR
    price_diffs = [high - low, high - close.shift(), close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)

    # initialize Supertrend column to True
    supertrend = [True] * len(df)

    for i in range(1, len(df.index)):
        curr, prev = i, i - 1

        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        else:
            supertrend[curr] = supertrend[prev]

            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan

    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final Lowerband': final_lowerband,
        'Final Upperband': final_upperband
    }, index=df.index)


def get_supertrend2(high, low, close, lookback, multiplier):
    """
    Calculate Supertrend indicator with a different approach.
    :param high: Series containing high prices.
    :param low: Series containing low prices.
    :param close: Series containing close prices.
    :param lookback: Lookback period for calculation.
    :param multiplier: Multiplier for Supertrend calculation.
    :return: Tuple containing Supertrend, upper band, and lower band DataFrames.
    """
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.ewm(lookback).mean()

    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    final_bands = pd.DataFrame(columns=['upper', 'lower'])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (close[i - 1] > final_bands.iloc[i - 1, 0]):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i - 1, 1]) | (close[i - 1] < final_bands.iloc[i - 1, 1]):
                final_bands.iloc[i, 1] = lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    supertrend = pd.DataFrame(columns=[f'supertrend_{lookback}'])
    supertrend.iloc[:, 0] = [x for x in final_bands['upper'] - final_bands['upper']]

    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]

    upt = []
    dt = []

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.DataFrame(supertrend.iloc[:, 0]), pd.DataFrame(upt).rename(columns={0: 'upt'}), pd.DataFrame(
        dt).rename(columns={0: 'dt'})
    st = st.rename(columns={0: 'supertrend'})
    upt.index, dt.index = supertrend.index, supertrend.index

    return st, upt, dt


def vwap(g):
    """
    Calculate Volume Weighted Average Price (VWAP) for a given DataFrame.
    :param g: DataFrame containing price and volume data.
    :return: DataFrame with VWAP values.
    """
    g['tp'] = (g['low'] + g['close'] + g['high']) / 3
    g['vwap'] = (g.tp * g.real_volume).cumsum() / g.real_volume.cumsum()
    return g


def VWAP_function(df, plot=False):
    """
    Calculate VWAP for a given DataFrame and optionally plot it.
    :param df: DataFrame containing price and volume data.
    :param plot: Boolean to indicate whether to plot the VWAP.
    :return: DataFrame with VWAP values.
    """
    copy_df = df.copy()
    copy_df['tp'] = (copy_df['low'] + copy_df['close'] + copy_df['high']) / 3
    copy_df['time'] = pd.to_datetime(copy_df['time'], unit='s')
    copy_df['day'] = copy_df['time'].dt.day
    copy_df = copy_df.groupby(pd.Grouper(key='day')).apply(lambda x: vwap(x))
    det = copy_df['vwap'].droplevel(level='day')
    df['vwap'] = det
    df['vwap_distance'] = abs(df['vwap'] - df['close'])
    if plot:
        vwap_plot(df)
    return df


def vwap_new(data):
    """
    Calculate VWAP for a given DataFrame.
    :param data: DataFrame containing price and volume data.
    :return: DataFrame with VWAP values.
    """
    data['vwap'] = (((data['high'] + data['low'] + data['close']) / 3) * data['tick_volume']).cumsum() / data[
        'tick_volume'].cumsum()
    return data


def vwap_code(df):
    """
    Calculate VWAP using a different method.
    :param df: DataFrame containing price and volume data.
    :return: DataFrame with VWAP values.
    """
    q = df.tick_volume.values
    p = df.close.values
    return df.assign(vwap=(p * q).cumsum() / q.cumsum())


def vwap_plot(df):
    """
    Plot VWAP with close prices.
    :param df: DataFrame containing price and VWAP data.
    """
    plt.plot(df['close'], linewidth=2, label='close')
    plt.plot(df['vwap'], linewidth=2, color='orange', linestyle='--', label='vwap')
    plt.legend(loc='lower right')
    plt.title('Close VWAP CHANNEL')
    plt.show()


def EMA(df, longest_MA_window=200):
    """
    Calculate Exponential Moving Averages (EMA) and Simple Moving Averages (SMA) for a given DataFrame.
    :param df: DataFrame containing price data.
    :param longest_MA_window: Window for the longest SMA.
    :return: DataFrame with EMA and SMA values.
    """
    df["14EMA"] = TA.EMA(df, 14)
    df["21EMA"] = TA.EMA(df, 21)
    df["30EMA"] = TA.EMA(df, 30)
    df["50EMA"] = TA.EMA(df, 50)
    df["200SMA"] = TA.SMA(df, longest_MA_window)
    df['4SMA'] = df.rolling(4).mean()['close']
    return df


def ATRMACD(df):
    """
    Calculate ATR, Bollinger Bands, MACD, and RSI for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with ATR, Bollinger Bands, MACD, and RSI values.
    """
    df["ATR"] = TA.ATR(df)
    bbands_df = TA.BBANDS(df)
    macd_df = TA.MACD(df)
    df["RSI"] = TA.RSI(df)
    bbands_df = pd.concat([bbands_df, macd_df], axis=1)
    df = pd.concat([df, bbands_df], axis=1)
    df.drop(columns="SIGNAL", inplace=True)
    return df


def Signal(df):
    """
    Generate trading signals based on various technical indicators.
    :param df: DataFrame containing price data and technical indicators.
    :return: DataFrame with trading signals.
    """
    discrete_features = ["Bollinger_Bands_Above_Upper_BB", "Bollinger_Bands_Below_Lower_BB",
                         "RSI_Over_70", "50EMA/200SMA_Cross, 50EMA>200SMA",
                         "RSI_Under_30", "VWAP_Cross_From_Above", "VWAP_Cross_From_Below", "SUPER_UP", 'SUPER_DOWN',
                         'PIVOTS', 'HIGH', 'LOW', 'ADX_UP', 'ADX_DOWN', 'DOJI', 'Direction', 'Chaiken_Signal',
                         "14EMA/21EMA_Cross, 14EMA>21EMA", 'MACD_Signal',
                         'green body> 500', 'red body> 500', 'slope_signal', 'slope_vwap']

    for feature in discrete_features:
        df[feature] = 0.0

    for index, row in df.iterrows():
        if row["close"] < row["BB_LOWER"]:
            df.loc[index, "Bollinger_Bands_Below_Lower_BB"] = 1
        if row["close"] > row["BB_UPPER"]:
            df.loc[index, "Bollinger_Bands_Above_Upper_BB"] = 1

    df['SUPER_UP'] = np.where(~df['dt'].isna(), 1, 0)
    df['SUPER_DOWN'] = np.where(~df['upt'].isna(), 1, 0)

    df["14EMA/21EMA_Cross, 14EMA>21EMA"][14:] = np.where(df["14EMA"][14:] > df["21EMA"][14:], 1.0, 0.0)
    df["50EMA/200SMA_Cross, 50EMA>200SMA"][50:] = np.where(df["50EMA"][50:] > df["200SMA"][50:], 1.0, 0.0)

    for index, row in df.iterrows():
        if 30 > row["RSI"]:
            df.loc[index, "RSI_Under_30"] = 1
        if 70 < row["RSI"]:
            df.loc[index, "RSI_Over_70"] = 1

    df["VWAP_Cross_From_Above"] = np.where(df["vwap"] <= df["close"], 1.0, 0)
    df["VWAP_Cross_From_Below"] = np.where(df["vwap"] > df["close"], 1.0, 0)

    df['PIVOTS'] = np.where(df['distancebetweenpivot'] < 10, 1, 0)
    df['ADX_UP'] = np.where((df['ADX'] > 25) & (df['PLUS_DI'] > df['MINUS_DI']), 1, 0)
    df['ADX_DOWN'] = np.where((df['ADX'] > 25) & (df['PLUS_DI'] < df['MINUS_DI']), 1, 0)
    df['HIGH'] = np.where(df['high'].shift(1) < df['high'], 1, 0)
    df['LOW'] = np.where(df['low'].shift(1) > df['low'], 1, 0)
    body_size = abs(df['open'] - df['close'])
    total_range = df['high'] - df['low']
    doji_condition = (body_size < 0.1 * total_range)
    df['DOJI'] = np.where(doji_condition, 1, 0)
    df['Direction'] = np.where(df['close'] > df['open'], 1, 0)
    df['Chaiken_Signal'] = np.where(df['Chaikin'] > 0, 1, 0)

    green_candle_condition = (df['close'] > df['open'])
    red_candle_condition = (df['close'] < df['open'])
    body_size_condition = (body_size > 500)
    df['green body> 500'] = np.where((green_candle_condition & body_size_condition).rolling(5).sum() > 0, 1, 0)
    df['red body> 500'] = np.where((red_candle_condition & body_size_condition).rolling(5).sum() > 0, 1, 0)
    df['MACD_Signal'] = np.where(df['MACD'] > 0, 1, 0)
    df['HIGH2'] = np.where((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2)), 1, 0)
    df['LOW2'] = np.where((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2)), 1, 0)
    df = channel(df, 20, 40)
    df['Superup_close'] = np.where(abs(df['upt'] - df['low']) < 50, 1, 0)
    df['Superdown_close'] = np.where(abs(df['dt'] - df['high']) < 50, 1, 0)
    df['slope'] = np.degrees(np.arctan(df['4SMA'].diff() / 4))
    df['slope_signal'] = np.where(df['slope'] > 0, 1, 0)

    vwap_diff = df['vwap'].diff()
    df['slope_vwap'] = np.where(vwap_diff > 0, 1, 0)

    return df


def getHigherLows(data: np.array, order=5, K=2):
    """
    Find consecutive higher lows in a price pattern.
    :param data: Array containing price data.
    :param order: Number of periods to consider for each local extremum.
    :param K: Number of consecutive higher lows required.
    :return: List of higher lows.
    """
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] < lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getLowerHighs(data: np.array, order=5, K=2):
    """
    Find consecutive lower highs in a price pattern.
    :param data: Array containing price data.
    :param order: Number of periods to consider for each local extremum.
    :param K: Number of consecutive lower highs required.
    :return: List of lower highs.
    """
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] > highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getHigherHighs(data: np.array, order=5, K=2):
    """
    Find consecutive higher highs in a price pattern.
    :param data: Array containing price data.
    :param order: Number of periods to consider for each local extremum.
    :param K: Number of consecutive higher highs required.
    :return: List of higher highs.
    """
    high_idx = argrelextrema(data, np.greater, order=5)[0]
    highs = data[high_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def getLowerLows(data: np.array, order=5, K=2):
    """
    Find consecutive lower lows in a price pattern.
    :param data: Array containing price data.
    :param order: Number of periods to consider for each local extremum.
    :param K: Number of consecutive lower lows required.
    :return: List of lower lows.
    """
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i - 1]:
            ex_deque.clear()

        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema


def plot_stock_with_ema_sma(df):
    """
    Plot stock prices with 50EMA and 200SMA.
    :param df: DataFrame containing price and moving average data.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(df['close'], label='Close Price', color='black', linewidth=1.2)
    plt.plot(df['50EMA'], label='50EMA', color='blue', linestyle='--', linewidth=1)
    plt.plot(df['200SMA'], label='200SMA', color='red', linestyle='--', linewidth=1)

    plt.title('Stock Price with 50EMA and 200SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def TPSL(df, longest_MA_window=200):
    """
    Calculate Take Profit and Stop Loss (TPSL) for a given DataFrame.
    :param df: DataFrame containing price data.
    :param longest_MA_window: Window for the longest moving average.
    :return: DataFrame with TPSL values.
    """
    num_rows_in_df = df.shape[0]
    df['Exit Loss'] = 0
    df["Exit Price"] = 0
    df["Exit"] = 0
    df['Long_lost'] = 0
    df['Short_lost'] = 0

    reward = 3  # Reward to risk ratio
    risk = 2

    for j in range(longest_MA_window, num_rows_in_df):
        entry = df["close"].iloc[j]
        atr = df["ATR"].iloc[j]

        long_stop_loss = entry - (risk * atr)
        long_take_profit = entry + (reward * atr)
        short_stop_loss = entry + (risk * atr)
        short_take_profit = entry - (reward * atr)

        a = True
        b = True
        df['Long_lost'].iloc[j] = long_stop_loss
        df['Short_lost'].iloc[j] = short_stop_loss

        for k in range(j + 1, num_rows_in_df):
            curr_low = df["low"].iloc[k]
            curr_high = df["high"].iloc[k]

            if curr_low <= long_stop_loss or curr_low <= short_take_profit:
                a = False
                if a == False and b == False:
                    break
                if curr_low <= short_take_profit and b == True:
                    df["Exit Price"].iloc[j] = short_take_profit
                    df['Exit Loss'].iloc[j] = short_stop_loss
                    df["Exit"].iloc[j] = -1
                    break

            elif curr_high >= short_stop_loss or curr_high >= long_take_profit:
                b = False
                if a == False and b == False:
                    break
                if curr_high >= long_take_profit and a == True:
                    df["Exit Price"].iloc[j] = long_take_profit
                    df['Exit Loss'].iloc[j] = long_stop_loss
                    df["Exit"].iloc[j] = 1
                    break

    df = df[longest_MA_window:]
    return df


def Bollingerplot(df):
    """
    Plot Bollinger Bands with close prices.
    :param df: DataFrame containing price and Bollinger Band data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='Close Price', color='black', linewidth=1.2)
    plt.plot(df['BB_UPPER'], label='Upper Bollinger Band', color='red', linestyle='--', linewidth=1)
    plt.plot(df['BB_LOWER'], label='Lower Bollinger Band', color='blue', linestyle='--', linewidth=1)

    below_lower_band = df[df["Bollinger_Bands_Below_Lower_BB"] == 1]
    plt.scatter(below_lower_band.index, below_lower_band["close"], c='blue', marker='o', label='Close Below Lower Band')

    above_upper_band = df[df["Bollinger_Bands_Above_Upper_BB"] == 1]
    plt.scatter(above_upper_band.index, above_upper_band["close"], c='red', marker='o', label='Close Above Upper Band')

    plt.title('Bollinger Bands and Close Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def plot_supertrend_with_signals(df1):
    """
    Plot Supertrend signals with close prices.
    :param df1: DataFrame containing price and Supertrend data.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(df1['close'], label='Close Price', color='black', linewidth=1.2)
    plt.scatter(df1.index, df1['upt'], color='green', linestyle='--', linewidth=1, label='Uptrend Signal')
    plt.scatter(df1.index, df1['dt'], color='red', linestyle='--', linewidth=1, label='Downtrend Signal')

    plt.title('Close Price with Uptrend and Downtrend Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def get_resistance_support2(df):
    """
    Identify support and resistance levels.
    :param df: DataFrame containing price data.
    :return: List of tuples containing support and resistance levels.
    """
    pivots = []
    max_list = []
    min_list = []
    for i in range(10, len(df) - 10):
        high_range = df['high'][i - 10:i]
        current_max = high_range.max()
        if current_max not in max_list:
            max_list = []
        max_list.append(current_max)
        if len(max_list) == 5 and is_far_from_level(current_max, pivots, df):
            pivots.append((high_range.idxmax(), current_max))

        low_range = df['low'][i - 10:i]
        current_min = low_range.min()
        if current_min not in min_list:
            min_list = []
        min_list.append(current_min)
        if len(min_list) == 5 and is_far_from_level(current_min, pivots, df):
            pivots.append((low_range.idxmin(), current_min))

    return pivots


def distance_to_close_supres(pivots, df):
    """
    Calculate distance from price to nearest support or resistance level.
    :param pivots: DataFrame containing support and resistance levels.
    :param df: DataFrame containing price data.
    :return: DataFrame with distance to nearest pivot level.
    """
    pivots = pd.DataFrame(pivots)
    df['closest_pivot'] = df.apply(lambda row: min(pivots[1], key=lambda x: abs(x - row['high'])), axis=1)
    df['distancebetweenpivot'] = abs(df['closest_pivot'] - df['high'])
    return df


def plot_all_resistance_support(levels, df):
    """
    Plot all identified support and resistance levels with close prices.
    :param levels: List of tuples containing support and resistance levels.
    :param df: DataFrame containing price data.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(range(len(df)), df['close'])
    for level in levels:
        plt.hlines(level[1], xmin=level[0], xmax=max(range(len(df))), colors='blue', linestyle='--')
    plt.show()


def ADX_data(df, adx_period=14, plot=False):
    """
    Calculate ADX data for a given DataFrame.
    :param df: DataFrame containing price data.
    :param adx_period: Period for ADX calculation.
    :param plot: Boolean to indicate whether to plot the ADX data.
    :return: DataFrame with ADX values.
    """
    df['ADX'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[2]).rename(columns={0: 'ADX'})
    df['PLUS_DI'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[0]).rename(
        columns={0: 'PLUS_DI'})
    df['MINUS_DI'] = pd.DataFrame(get_adx(df['high'], df['low'], df['close'], adx_period)[1]).rename(
        columns={0: 'MINUS_DI'})
    df['ADX_signal'] = pd.DataFrame(implement_adx_strategy(df['close'], df['PLUS_DI'], df['MINUS_DI'], df['ADX']))
    if plot:
        plot_adx(df)
    return df


def plot_adx(df):
    """
    Plot ADX with close prices.
    :param df: DataFrame containing price and ADX data.
    """
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['close'], linewidth=3, color='#ff9800', alpha=0.6)
    ax1.set_title('df CLOSING PRICE')
    ax2.plot(df['PLUS_DI'], color='#26a69a', label='+ DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['MINUS_DI'], color='#f44336', label='- DI 14', linewidth=3, alpha=0.3)
    ax2.plot(df['ADX'], color='#2196f3', label='ADX 14', linewidth=3)
    ax2.axhline(25, color='grey', linewidth=2, linestyle='--')
    ax2.legend()
    ax2.set_title('df ADX 14')
    plt.show()


def chaikin_oscillator(data, fast_period=3, slow_period=10):
    """
    Calculate the Chaikin Oscillator for a given DataFrame.
    :param data: DataFrame containing price and volume data.
    :param fast_period: Number of periods for the fast EMA.
    :param slow_period: Number of periods for the slow EMA.
    :return: Series with Chaikin Oscillator values.
    """
    adl = ((2 * data['close'] - data['low'] - data['high']) / (data['high'] - data['low'])) * data['tick_volume']
    adl = adl.cumsum()

    fast_ema = adl.ewm(span=fast_period, min_periods=fast_period).mean()
    slow_ema = adl.ewm(span=slow_period, min_periods=slow_period).mean()

    chaikin_oscillator = fast_ema - slow_ema

    return chaikin_oscillator


def CCI(df, ndays=14):
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame.
    :param df: DataFrame containing price data.
    :param ndays: Number of periods for the CCI calculation.
    :return: DataFrame with CCI values.
    """
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma'] = df['TP'].rolling(ndays).mean()
    df['mad'] = df['TP'].rolling(window=ndays).apply(lambda x: (x - x.mean()).abs().mean())
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
    return df


def channel(df, lookback, threshold):
    """
    Calculate price channels for a given DataFrame.
    :param df: DataFrame containing price data.
    :param lookback: Lookback period for channel calculation.
    :param threshold: Threshold for determining signals.
    :return: DataFrame with channel signals.
    """
    df['Low_Signal'] = 0
    df['High_Signal'] = 0
    df['upper'] = df['close'].rolling(lookback).max().shift(1)
    df['lower'] = df['close'].rolling(lookback).min().shift(1)
    low_abs = abs(df['lower'] - df['low'])
    approach_signal1 = (low_abs <= threshold)
    df['Low_Signal'] = np.where(approach_signal1, 1, 0)
    df['High_break'] = np.where(df['close'] > df['upper'], 1, 0)

    high_abs = abs(df['upper'] - df['high'])
    approach_signal = (high_abs <= threshold)
    df['High_Signal'] = np.where(approach_signal, 1, 0)
    df['Low_break'] = np.where(df['close'] < df['lower'], 1, 0)

    return df


def Linreg(df):
    """
    Calculate linear regression for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with linear regression values.
    """
    df['Linreg'] = tap.linreg(df['close'], length=14)
    df['Linreg_signal'] = np.where(df['close'] > df['Linreg'], 1, 0)
    return df


def German(df, historical_period, scaling_factor=1.0):
    """
    Calculate Garman-Klass volatility for a given DataFrame.
    :param df: DataFrame containing price data.
    :param historical_period: Number of periods for historical volatility calculation.
    :param scaling_factor: Scaling factor for volatility threshold.
    :return: DataFrame with Garman-Klass volatility values.
    """
    df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * (
                (np.log(df['close']) - np.log(df['open'])) ** 2)

    historical_volatility = df['garman_klass_vol'].tail(historical_period)
    volatility_threshold = historical_volatility.median() * scaling_factor

    df['prev_volatility'] = df['garman_klass_vol'].shift(1)
    df['volatility_breakout_signal'] = (df['garman_klass_vol'] > volatility_threshold) & (
                df['garman_klass_vol'] > df['prev_volatility'])
    df['long_signal1'] = np.where((df['close'] > df['high'].shift(1)) & df['volatility_breakout_signal'], 1, 0)
    df['short_signal1'] = np.where((df['close'] < df['low'].shift(1)) & df['volatility_breakout_signal'], 1, 0)
    df['German_signal'] = np.where(df['garman_klass_vol'] > volatility_threshold, 1, 0)
    return df


def MACD2(df):
    """
    Calculate MACD for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with MACD values.
    """
    macd1 = tap.macd(df.close)
    df['MACD1'], df['MACD_signal'], df['MACD_hist'] = macd1.iloc[:, 0], macd1.iloc[:, 1], macd1.iloc[:, 2]
    df['TotalSignal'] = df.apply(lambda row: total_signal(df, row.name) if row.name != 0 else 0, axis=1)
    return df


def total_signal(df, current_candle):
    """
    Calculate total signal based on MACD.
    :param df: DataFrame containing price data and MACD values.
    :param current_candle: Current candle index.
    :return: Total signal value.
    """
    macd_values_3_2 = df.loc[current_candle - 3:current_candle - 2, "MACD1"]
    macd_values_1 = df.loc[current_candle - 1:current_candle, "MACD1"]
    macd_signal_values_3_2 = df.loc[current_candle - 3:current_candle - 2, "MACD_signal"]
    macd_signal_values_1 = df.loc[current_candle - 1:current_candle, "MACD_signal"]

    if (all(macd_values_3_2 < macd_signal_values_3_2) and all(macd_values_1 > macd_signal_values_1)):
        return 1
    elif (all(macd_values_3_2 > macd_signal_values_3_2) and all(macd_values_1 < macd_signal_values_1)):
        return -1
    else:
        return 0


def HH_HL_LL_LH(df):
    """
    Identify higher highs, higher lows, lower highs, and lower lows.
    :param df: DataFrame containing price data.
    :return: DataFrame with identified levels.
    """
    from matplotlib.lines import Line2D

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    close = df['close'].values
    dates = df.index

    order = 5
    K = 2

    hh = getHigherHighs(close, order, K)
    hl = getHigherLows(close, order, K)
    ll = getLowerLows(close, order, K)
    lh = getLowerHighs(close, order, K)

    df['hh'] = 0
    df['lh'] = 0
    df['ll'] = 0
    df['hl'] = 0
    df['hh_signal'] = 0
    df['lh_signal'] = 0
    df['ll_signal'] = 0
    df['hl_signal'] = 0

    for i, (start, end) in enumerate(hh):
        df.loc[end, 'hh'] = 1

    for i, (start, end) in enumerate(lh):
        df.loc[end, 'lh'] = 1

    for i, (start, end) in enumerate(ll):
        df.loc[end, 'll'] = 1

    for i, (start, end) in enumerate(hl):
        df.loc[end, 'hl'] = 1

    for index, row in df.iterrows():
        if row['hh'] == 1:
            start_index = index + 5
            while index + 1 < len(df) - 1 and (df.loc[index + 1, 'hh'] != 1 and df.loc[index + 1, 'lh'] != 1):
                index += 1
            end_index = index + 5
            df.loc[start_index:end_index + 1, 'hh_signal'] = 1

        if row['lh'] == 1:
            start_index = index + 5
            while index + 1 < len(df) - 1 and (df.loc[index + 1, 'hh'] != 1 and df.loc[index + 1, 'lh'] != 1):
                index += 1
            end_index = index + 5
            df.loc[start_index:end_index + 1, 'lh_signal'] = 1

    for index, row in df.iterrows():
        if row['ll'] == 1:
            start_index = index + 5
            while index + 1 < len(df) - 1 and (df.loc[index + 1, 'll'] != 1 and df.loc[index + 1, 'hl'] != 1):
                index += 1
            end_index = index + 5
            df.loc[start_index:end_index + 1, 'll_signal'] = 1

        if row['hl'] == 1:
            start_index = index + 5
            while index + 1 < len(df) - 1 and (df.loc[index + 1, 'll'] != 1 and df.loc[index + 1, 'hl'] != 1):
                index += 1
            end_index = index + 5
            df.loc[start_index:end_index + 1, 'hl_signal'] = 1

    plt.figure(figsize=(15, 8))
    plt.plot(df['close'])
    _ = [plt.plot(dates[i], close[i], c=colors[1]) for i in hh]
    _ = [plt.plot(dates[i], close[i], c=colors[2]) for i in hl]
    _ = [plt.plot(dates[i], close[i], c=colors[3]) for i in ll]
    _ = [plt.plot(dates[i], close[i], c=colors[4]) for i in lh]
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title(f'Potential Divergence Points for Closing Price')
    legend_elements = [
        Line2D([0], [0], color=colors[0], label='Close'),
        Line2D([0], [0], color=colors[1], label='Higher Highs'),
        Line2D([0], [0], color=colors[2], label='Higher Lows'),
        Line2D([0], [0], color=colors[3], label='Lower Lows'),
        Line2D([0], [0], color=colors[4], label='Lower Highs')
    ]
    plt.legend(handles=legend_elements)
    print(df[['ll', 'll_signal', 'hl', 'hl_signal', 'hh', 'hh_signal', 'lh', 'lh_signal']].tail(100))
    #plt.show()

    return df


def set_time_vwap(df):
    """
    Set the time index and calculate VWAP for a given DataFrame.
    :param df: DataFrame containing price and volume data.
    :return: DataFrame with VWAP values.
    """
    df['time'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.set_index('time')

    copy = df.copy()
    copy.index = pd.to_datetime(copy.index)
    copy = df.groupby(copy.index.date, group_keys=False).apply(vwap_code)
    det = copy['vwap']
    df['vwap'] = det
    return df


def set_prev(df):
    """
    Set previous period values for a given DataFrame.
    :param df: DataFrame containing price data.
    :return: DataFrame with previous period values.
    """
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_real_volume'] = df['real_volume'].shift(1)

    df['OO'] = df['open'] - df['open'].shift(1)
    df['OC'] = df['open'] - df['prev_close']
    return df


def optimize_random_state(discrete_X, discrete_y, split):
    """
    Optimize the random state for undersampling.
    :param discrete_X: DataFrame containing feature data.
    :param discrete_y: Series containing target data.
    :param split: Index to split the data into training and testing sets.
    :return: Best random state.
    """
    random_states = np.arange(start=0, stop=101)
    scores = []

    for state in random_states:
        discrete_X_train = discrete_X.iloc[:split]
        discrete_y_train = discrete_y.iloc[:split]

        discrete_X_test = discrete_X.iloc[split:]
        discrete_y_test = discrete_y.iloc[split:]

        discrete_rus = RandomUnderSampler(random_state=state)
        undersampled_discrete_X_train, undersampled_discrete_y_train = discrete_rus.fit_resample(discrete_X_train,
                                                                                                 discrete_y_train)

        rf_model = tree.DecisionTreeClassifier(random_state=state)
        rf_model = rf_model.fit(undersampled_discrete_X_train, undersampled_discrete_y_train)

        predictions = rf_model.predict(discrete_X_test)
        rf_training_signal_predictions = rf_model.predict(undersampled_discrete_X_train)
        rf_training_report = classification_report(undersampled_discrete_y_train, rf_training_signal_predictions)

        rf_testing_signal_predictions = rf_model.predict(discrete_X_test)
        rf_testing_report = classification_report(discrete_y_test, rf_testing_signal_predictions)

        scores.append(rf_model.score(discrete_X_test, discrete_y_test))

    results = pd.DataFrame({'random_state': random_states, 'score': scores})
    best_random_state = results.loc[results['score'].idxmax()]

    print("Best random state:", best_random_state['random_state'])
    print("Best score:", best_random_state['score'])

    return int(best_random_state['random_state'])


def Machinelearning(df, filtered_df, discrete_X, discrete_y, discrete_X_train, discrete_y_train, discrete_X_test,
                    discrete_y_test, split, best_random_state):
    """
    Perform machine learning on the given data.
    :param df: DataFrame containing price data.
    :param filtered_df: DataFrame containing filtered data.
    :param discrete_X: DataFrame containing feature data.
    :param discrete_y: Series containing target data.
    :param discrete_X_train: Training feature data.
    :param discrete_y_train: Training target data.
    :param discrete_X_test: Testing feature data.
    :param discrete_y_test: Testing target data.
    :param split: Index to split the data into training and testing sets.
    :param best_random_state: Best random state for the model.
    :return: DataFrames with machine learning results.
    """
    discrete_rus = RandomUnderSampler(random_state=best_random_state)
    undersampled_discrete_X_train, undersampled_discrete_y_train = discrete_rus.fit_resample(discrete_X_train,
                                                                                             discrete_y_train)

    rf_model = tree.DecisionTreeClassifier(random_state=best_random_state)
    rf_model = rf_model.fit(undersampled_discrete_X_train, undersampled_discrete_y_train)

    predictions = rf_model.predict(discrete_X_test)
    rf_training_signal_predictions = rf_model.predict(undersampled_discrete_X_train)
    rf_training_report = classification_report(undersampled_discrete_y_train, rf_training_signal_predictions)
    rf_testing_signal_predictions = rf_model.predict(discrete_X_test)
    rf_testing_report = classification_report(discrete_y_test, rf_testing_signal_predictions)

    def calculate_algo_returns(exit_price, close):
        if exit_price == 0:
            return 0
        else:
            return (exit_price - close) / close

    rf_predictions_df = pd.DataFrame(index=discrete_X_test.index)
    rf_predictions_df["predicted_signal"] = rf_testing_signal_predictions
    rf_predictions_df["Exit"] = df['Exit']
    rf_predictions_df["Exit Price"] = df['Exit Price']
    rf_predictions_df["Exit_Loss"] = df['Exit Loss']
    rf_predictions_df["Long_lost"] = df['Long_lost']
    rf_predictions_df["Short_lost"] = df['Short_lost']
    rf_predictions_df["close"] = df['close']

#ths rf_predictions_df2 is for the backtest
    rf_predictions_df2 = rf_predictions_df.copy()
    rf_predictions_df2["predicted_signal"] = rf_testing_signal_predictions
    rf_predictions_df2 = rf_predictions_df2.reindex(df.index)
    rf_predictions_df2['predicted_signal'].fillna(0, inplace=True)
    rf_predictions_df2['Close'] = df['close']
    rf_predictions_df2['ATR'] = df['ATR']
    rf_predictions_df2['Open'] = df['open']
    rf_predictions_df2['High'] = df['high']
    rf_predictions_df2['Low'] = df['low']
    rf_predictions_df2.reset_index(inplace=True)

    rf_predictions_df["actual_returns"] = df["close"].pct_change()
    rf_predictions_df["algo_returns"] = filtered_df.apply(
        lambda row: calculate_algo_returns(row["Exit Price"], row["close"]), axis=1)

    def calculate_algorithm_return(row):
        """
        Calculate the algorithm return based on exit and signal values.
        :param row: Row of DataFrame containing exit and signal values.
        :return: Calculated return.
        """
        if row['predicted_signal'] == row['Exit'] and row['predicted_signal'] != 0:
            return (row['Exit Price'] - row['close']) / row['close']
        elif row['Exit'] != 0 and row['predicted_signal'] != row['Exit']:
            return (row['Exit_Loss'] - row['close']) / row['close']
        elif row['Exit'] == 0 and row['predicted_signal'] == 1:
            return (row['Long_lost'] - row['close']) / row['close']
        elif row['Exit'] == 0 and row['predicted_signal'] == -1:
            return (row['Short_lost'] - row['close']) / row['close']
        elif row['predicted_signal'] == 0:
            return 0
        else:
            return None

    rf_predictions_df['algorithm_return_new'] = rf_predictions_df.apply(calculate_algorithm_return, axis=1)
    rf_predictions_df["trading_algorithm_returns_new"] = np.where(
        rf_predictions_df["predicted_signal"] == 0,
        0,
        np.where(
            rf_predictions_df['Exit'] != 0,
            rf_predictions_df["algorithm_return_new"] * rf_predictions_df["Exit"],
            rf_predictions_df["algorithm_return_new"] * rf_predictions_df["predicted_signal"]
        )
    )
    rf_predictions_df["trading_algorithm_returns"] = (
            rf_predictions_df["algo_returns"] * rf_predictions_df["predicted_signal"]
    )

    # Plot the returns
    (1 + rf_predictions_df[["actual_returns", "trading_algorithm_returns_new"]]).cumprod().plot(
        title='RF Actual vs Algo Returns')
    plt.savefig('_rf_vs_act_returns', facecolor='white', edgecolor='white', transparent='false', bbox_inches='tight')

    df["daily returns"] = df["close"].pct_change()

    plt.show()

    importances = rf_model.feature_importances_
    # List the top 10 most important features
    importances_sorted = sorted(zip(rf_model.feature_importances_, discrete_X.columns), reverse=True)
    print(importances_sorted[:10])

    return df, rf_predictions_df, rf_predictions_df2

def Backtest(rf_predictions_df2):
    """
    Perform backtesting on the given data.
    :param rf_predictions_df2: DataFrame containing predictions and price data.
    :return: Backtest statistics.
    """
    from backtesting import Strategy, Backtest

    def SIGNAL():
        return rf_predictions_df2.predicted_signal

    class MyStrat(Strategy):
        mysize = 1
        slcoef = 2
        TPSLRatio = 1.5

        def init(self):
            super().init()
            self.signal = self.I(SIGNAL)

        def next(self):
            super().next()
            slatr = self.slcoef * self.data.ATR[-1]
            TPSLRatio = self.TPSLRatio

            if self.signal == 1:
                sl1 = self.data.Close[-1] - slatr
                tp1 = self.data.Close[-1] + slatr * TPSLRatio
                self.buy(sl=sl1, tp=tp1, size=self.mysize)
            elif self.signal == -1:
                sl1 = self.data.Close[-1] + slatr
                tp1 = self.data.Close[-1] - slatr * TPSLRatio
                self.sell(sl=sl1, tp=tp1, size=self.mysize)

    bt = Backtest(rf_predictions_df2, MyStrat, cash=100000, margin=1 / 30)
    stats = bt.run()
    bt.plot()
    print(stats)

    return bt


def is_support(df,i):
  cond1 = df['low'][i] < df['low'][i-1]
  cond2 = df['low'][i] < df['low'][i+1]
  cond3 = df['low'][i+1] < df['low'][i+2]
  cond4 = df['low'][i-1] < df['low'][i-2]
  return (cond1 and cond2 and cond3 and cond4)

def is_resistance(df,i):
  cond1 = df['high'][i] > df['high'][i-1]
  cond2 = df['high'][i] > df['high'][i+1]
  cond3 = df['high'][i+1] > df['high'][i+2]
  cond4 = df['high'][i-1] > df['high'][i-2]
  return (cond1 and cond2 and cond3 and cond4)


def is_far_from_level(value, levels, df):
  ave =  np.mean(df['high'] - df['low'])
  return np.sum([abs(value-level)<ave for _,level in levels])==0


