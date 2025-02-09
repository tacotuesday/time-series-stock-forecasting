from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch.unitroot import KPSS, PhillipsPerron, ZivotAndrews, VarianceRatio
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Tuple, List, Union
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from tqdm.notebook import tqdm
from joblib import Parallel, delayed


def test_stationarity_robust(
    timeseries: Union[pd.Series, np.ndarray], alpha: float = 0.05
) -> None:
    """
    Perform robust stationarity tests on a time series.

    This function applies four different stationarity tests:
    1. Augmented Dickey-Fuller (ADF) test
    2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    3. Phillips-Perron test
    4. Zivot-Andrews test (if the series has more than 100 observations)

    Args:
        timeseries (Union[pd.Series, np.ndarray]): The time series to test for stationarity.
        alpha (float, optional): Significance level for the tests. Default is 0.05.

    Raises:
        ValueError: If the input is not a pandas Series or numpy array.
        RuntimeError: If any of the statistical tests fail to run.

    Returns:
        None: Results are printed to the console.
    """
    if not isinstance(timeseries, (pd.Series, np.ndarray)):
        raise ValueError("Input must be a pandas Series or numpy array.")

    try:
        # ADF test
        ADF_result = adfuller(timeseries)
        print(f"ADF Statistic: {ADF_result[0]:.6f}")
        print(f"p-value: {ADF_result[1]:.6f}")
        if ADF_result[1] < alpha:
            print(
                "Reject the null hypothesis that there is a unit root. The time series may be stationary."
            )
        else:
            print(
                "Fail to reject the null hypothesis that there is a unit root. The time series is non-stationary."
            )

        # KPSS test (statsmodels implementation)
        KPSS_result = kpss(timeseries)
        print(f"\nKPSS Statistic (statsmodels): {KPSS_result[0]:.6f}")
        print(f"p-value: {KPSS_result[1]:.6f}")
        if KPSS_result[1] < alpha:
            print(
                "Reject the null hypothesis that the time series is stationary. The time series is non-stationary."
            )
        else:
            print(
                "Fail to reject the null hypothesis that the time series is stationary. The time series is stationary."
            )

        # KPSS test (arch implementation)
        kpss_arch_result = kpss(timeseries)

        # Display the results
        print(
            f"\nKPSS Statistic (arch): {kpss_arch_result[0]}"
        )  # This line gives me the error AttributeError: 'tuple' object has no attribute 'stat'
        print(f"P-Value: {kpss_arch_result[1]}")
        print("Critical Values:")
        for key, value in kpss_arch_result[3].items():
            print(f"   {key}: {value}")

        # Interpretation:
        if kpss_arch_result[1] < alpha:
            print("The series is likely non-stationary.")
        else:
            print("The series is likely stationary.")

        # Phillips-Perron test
        PP_result = PhillipsPerron(timeseries)
        print(f"\nPhillips-Perron Statistic: {PP_result.stat:.6f}")
        print(f"p-value: {PP_result.pvalue:.6f}")
        if PP_result.pvalue < alpha:
            print(
                "Reject the null hypothesis that there is a unit root. The time series may be stationary."
            )
        else:
            print(
                "Fail to reject the null hypothesis that there is a unit root. The time series is non-stationary."
            )

        # Zivot-Andrews test
        if len(timeseries) > 100:
            ZA_result = ZivotAndrews(timeseries)
            print(f"\nZivot-Andrews Statistic: {ZA_result.stat:.6f}")
            print(f"p-value: {ZA_result.pvalue:.6f}")
            if ZA_result.pvalue < alpha:
                print(
                    "Reject the null hypothesis that there is a unit root with a single structural break. The time series may be stationary."
                )
            else:
                print(
                    "Fail to reject the null hypothesis that there is a unit root with a single structural break. The time series is non-stationary."
                )
        else:
            print(
                "\nToo few observations to run the Zivot-Andrews test with trend c and 10 lags."
            )

        # Variance Ratio test
        VR_result = VarianceRatio(timeseries, lags=2)
        print(f"\nVariance Ratio statistic: {VR_result.stat:.4f}")
        print(f"P-value: {VR_result.pvalue:.4f}")
        if VR_result.pvalue < alpha:
            print(
                "Reject the null hypothesis of a random walk. The series is not a random walk per the Variance Ratio Test."
            )
        else:
            print(
                "Fail to reject the null hypothesis. The series may follow a random walk per the Variance Ratio Test."
            )

    except Exception as e:
        raise RuntimeError(f"An error occurred during the statistical tests: {str(e)}")


def evaluate_forecast(
    actual: Union[List[float], pd.Series], predicted: Union[List[float], pd.Series]
) -> Tuple[float, float, float]:
    """
    Evaluate forecast performance using RMSE, MAE, and MAPE metrics.

    Args:
        actual (List[float] or pd.Series): The actual observed values.
        predicted (List[float] or pd.Series): The predicted values.

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - RMSE (Root Mean Squared Error)
            - MAE (Mean Absolute Error)
            - MAPE (Mean Absolute Percentage Error multiplied by 100)

    Raises:
        TypeError: If 'actual' or 'predicted' is not a list or pandas Series.
        ValueError: If the lengths of 'actual' and 'predicted' do not match.
        Exception: If an unexpected error occurs during metric calculations.
    """
    # Type checking
    if not isinstance(actual, (list, pd.Series)):
        raise TypeError(f"'actual' must be a list or pandas Series, got {type(actual)}")
    if not isinstance(predicted, (list, pd.Series)):
        raise TypeError(
            f"'predicted' must be a list or pandas Series, got {type(predicted)}"
        )

    # Length checking
    if len(actual) != len(predicted):
        raise ValueError("The length of 'actual' and 'predicted' must be the same.")

    try:
        rmse = root_mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
    except Exception as e:
        raise Exception(f"Error calculating forecast metrics: {e}")

    return rmse, mae, mape


def rolling_ma_forecast(
    train: Union[List[float], pd.Series], test: Union[List[float], pd.Series], q: int
) -> Tuple[pd.Series, float]:
    """
    Perform rolling mean average (MA) forecasting using an ARIMA model.

    Args:
        train (List[float] or pd.Series): The training dataset.
        test (List[float] or pd.Series): The testing dataset.
        q (int): The order of the MA component in the ARIMA model.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - predictions (pd.Series): The forecasted values.
            - rmse (float): The Root Mean Squared Error of the predictions.

    Raises:
        TypeError: If 'train' or 'test' is not a list or pandas Series, or if 'q' is not an integer.
        ValueError: If 'q' is negative or if 'test' is empty.
        Exception: If an unexpected error occurs during forecasting.
    """
    # Type checking
    if not isinstance(train, (list, pd.Series)):
        raise TypeError(f"'train' must be a list or pandas Series, got {type(train)}")
    if not isinstance(test, (list, pd.Series)):
        raise TypeError(f"'test' must be a list or pandas Series, got {type(test)}")
    if not isinstance(q, int):
        raise TypeError(f"'q' must be an integer, got {type(q)}")
    if q < 0:
        raise ValueError("'q' must be a non-negative integer.")
    if len(test) == 0:
        raise ValueError("'test' dataset must not be empty.")

    try:
        history = list(train)
        predictions = []

        for t in range(len(test)):
            # Fit ARIMA model on the current history
            model = ARIMA(history, order=(0, 0, q))
            model_fit = model.fit()

            # Forecast the next value
            forecast = model_fit.forecast()[0]
            predictions.append(forecast)

            # Append the actual value to history for next step
            history.append(test.iloc[t] if isinstance(test, pd.Series) else test[t])

        rmse = root_mean_squared_error(test, predictions)
        predictions_series = pd.Series(
            predictions,
            index=(
                test.index if isinstance(test, pd.Series) else range(len(predictions))
            ),
        )
    except Exception as e:
        raise Exception(f"Error during rolling MA forecast: {e}")

    return predictions_series, rmse


def rolling_ar_forecast(
    train: Union[List[float], pd.Series], test: Union[List[float], pd.Series], lags: int
) -> Tuple[pd.Series, float]:
    """
    Perform rolling AutoReg (AR) forecasting.

    Args:
        train (List[float] or pd.Series): The training dataset.
        test (List[float] or pd.Series): The testing dataset.
        lags (int): The number of lag observations to include in the model.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - predictions (pd.Series): The forecasted values.
            - rmse (float): The Root Mean Squared Error of the predictions.

    Raises:
        TypeError: If 'train' or 'test' is not a list or pandas Series, or if 'lags' is not an integer.
        ValueError: If 'lags' is negative or if 'test' is empty.
        Exception: If an unexpected error occurs during forecasting.
    """
    # Type checking
    if not isinstance(train, (list, pd.Series)):
        raise TypeError(f"'train' must be a list or pandas Series, got {type(train)}")
    if not isinstance(test, (list, pd.Series)):
        raise TypeError(f"'test' must be a list or pandas Series, got {type(test)}")
    if not isinstance(lags, int):
        raise TypeError(f"'lags' must be an integer, got {type(lags)}")
    if lags < 0:
        raise ValueError("'lags' must be a non-negative integer.")
    if len(test) == 0:
        raise ValueError("'test' dataset must not be empty.")

    try:
        history = list(train)
        predictions = []

        for t in range(len(test)):
            model = AutoReg(history, lags=lags).fit()
            pred = model.predict(start=len(history), end=len(history))
            predictions.append(pred[0])
            history.append(test.iloc[t] if isinstance(test, pd.Series) else test[t])

        predictions_series = pd.Series(
            predictions,
            index=(
                test.index if isinstance(test, pd.Series) else range(len(predictions))
            ),
        )
        rmse = root_mean_squared_error(test, predictions)
    except Exception as e:
        raise Exception(f"Error during rolling AR forecast: {e}")

    return predictions_series, rmse


def rolling_arima_forecast(
    train: Union[List[float], pd.Series],
    test: Union[List[float], pd.Series],
    order: Tuple[int, int, int],
) -> Tuple[pd.Series, float]:
    """
    Perform rolling ARIMA forecasting.

    Args:
        train (List[float] or pd.Series): The training dataset.
        test (List[float] or pd.Series): The testing dataset.
        order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - predictions (pd.Series): The forecasted values.
            - rmse (float): The Root Mean Squared Error of the predictions.

    Raises:
        TypeError: If 'train' or 'test' is not a list or pandas Series, or if 'order' is not a tuple of three integers.
        ValueError: If any element in 'order' is negative or if 'test' is empty.
        Exception: If an unexpected error occurs during forecasting.
    """
    # Type checking
    if not isinstance(train, (list, pd.Series)):
        raise TypeError(f"'train' must be a list or pandas Series, got {type(train)}")
    if not isinstance(test, (list, pd.Series)):
        raise TypeError(f"'test' must be a list or pandas Series, got {type(test)}")
    if not (
        isinstance(order, tuple)
        and len(order) == 3
        and all(isinstance(x, int) for x in order)
    ):
        raise TypeError("'order' must be a tuple of three integers (p, d, q).")
    if any(x < 0 for x in order):
        raise ValueError("All elements in 'order' must be non-negative integers.")
    if len(test) == 0:
        raise ValueError("'test' dataset must not be empty.")

    try:
        history = list(train)
        predictions = []

        for t in range(len(test)):
            # Fit ARIMA model on the current history
            model = ARIMA(history, order=order)
            model_fit = model.fit(method_kwargs={"maxiter": 500})

            # Forecast the next value
            forecast = model_fit.forecast()[0]
            predictions.append(forecast)

            # Append the actual value to history for next step
            history.append(test.iloc[t] if isinstance(test, pd.Series) else test[t])

        rmse = root_mean_squared_error(test, predictions)
        predictions_series = pd.Series(
            predictions,
            index=(
                test.index if isinstance(test, pd.Series) else range(len(predictions))
            ),
        )
    except Exception as e:
        raise Exception(f"Error during rolling ARIMA forecast: {e}")

    return predictions_series, rmse


def optimize_SARIMA_parallelized(
    endog: Union[pd.Series, List[float]],
    order_list: List[Tuple[int, int, int, int]],
    d: int,
    D: int,
    s: int,
) -> pd.DataFrame:
    """
    Optimize SARIMA model parameters by selecting the best (p, q, P, Q) orders based on AIC using parallel processing.

    Args:
        endog (pd.Series or List[float]): The endogenous time series data.
        order_list (List[Tuple[int, int, int, int]]): A list of (p, q, P, Q) tuples representing AR, MA, seasonal AR, and seasonal MA orders.
        d (int): The degree of non-seasonal differencing.
        D (int): The degree of seasonal differencing.
        s (int): The length of the seasonal cycle.

    Returns:
        pd.DataFrame: A DataFrame containing the (p, q, P, Q) orders and their corresponding AIC values,
                      sorted in ascending order of AIC.

    Raises:
        TypeError: If 'endog' is not a list or pandas Series, 'order_list' is not a list of tuples,
                   or if 'd', 'D', or 's' are not integers.
        ValueError: If 'order_list' is empty or no models could be fitted successfully.
        Exception: If an unexpected error occurs during SARIMA optimization.
    """
    # Type checking
    if not isinstance(endog, (list, pd.Series)):
        raise TypeError(f"'endog' must be a list or pandas Series, got {type(endog)}")
    if not isinstance(order_list, list):
        raise TypeError(
            f"'order_list' must be a list of tuples, got {type(order_list)}"
        )
    if not all(isinstance(order, tuple) and len(order) == 4 for order in order_list):
        raise TypeError("'order_list' must be a list of (p, q, P, Q) tuples.")
    if not all(isinstance(x, int) for x in [d, D, s]):
        raise TypeError("'d', 'D', and 's' must be integers.")
    if not order_list:
        raise ValueError("'order_list' must not be empty.")

    def fit_order(order: Tuple[int, int, int, int]) -> Union[List, None]:
        try:
            model = SARIMAX(
                endog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False,
            ).fit(disp=False)
            aic = model.aic
            return [order, aic]
        except Exception as e:
            print(f"Failed for order {order}: {e}")
            return None

    try:
        results = Parallel(n_jobs=-1)(
            delayed(fit_order)(order)
            for order in tqdm(order_list, desc="Optimizing SARIMA")
        )

        # Remove None results
        results = [r for r in results if r is not None]

        if not results:
            raise ValueError("No models were successfully fitted.")

        result_df = pd.DataFrame(results, columns=["(p,q,P,Q)", "AIC"])

        # Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(
            drop=True
        )
    except Exception as e:
        raise Exception(f"Error during SARIMA optimization: {e}")

    return result_df


def rolling_sarima_forecast(
    train: Union[List[float], pd.Series],
    test: Union[List[float], pd.Series],
    order: Tuple[int, int, int, int, int, int, int],
) -> Tuple[pd.Series, float]:
    """
    Perform rolling SARIMA forecasting.

    Args:
        train (List[float] or pd.Series): The training dataset.
        test (List[float] or pd.Series): The testing dataset.
        order (Tuple[int, int, int, int, int, int, int]): The SARIMA model order as (p, d, q, P, D, Q, s).

    Returns:
        Tuple[pd.Series, float]: A tuple containing:
            - predictions (pd.Series): The forecasted values.
            - rmse (float): The Root Mean Squared Error of the predictions.

    Raises:
        TypeError: If 'train' or 'test' is not a list or pandas Series, or if 'order' is not a tuple of seven integers.
        ValueError: If any element in 'order' is negative or if 'test' is empty.
        Exception: If an unexpected error occurs during forecasting.
    """
    # Type checking
    if not isinstance(train, (list, pd.Series)):
        raise TypeError(f"'train' must be a list or pandas Series, got {type(train)}")
    if not isinstance(test, (list, pd.Series)):
        raise TypeError(f"'test' must be a list or pandas Series, got {type(test)}")
    if not (
        isinstance(order, tuple)
        and len(order) == 7
        and all(isinstance(x, int) for x in order)
    ):
        raise TypeError(
            "'order' must be a tuple of seven integers (p, d, q, P, D, Q, s)."
        )
    if any(x < 0 for x in order):
        raise ValueError("All elements in 'order' must be non-negative integers.")
    if len(test) == 0:
        raise ValueError("'test' dataset must not be empty.")

    try:
        history = list(train)
        predictions = []

        for t in range(len(test)):
            # Fit SARIMA model on the current history
            model = SARIMAX(
                history,
                order=(order[0], order[1], order[2]),
                seasonal_order=(order[3], order[4], order[5], order[6]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit(disp=False)

            # Forecast the next value
            forecast = model_fit.forecast()[0]
            predictions.append(forecast)

            # Append the actual value to history for next step
            history.append(test.iloc[t] if isinstance(test, pd.Series) else test[t])

        rmse = root_mean_squared_error(test, predictions)
        predictions_series = pd.Series(
            predictions,
            index=(
                test.index if isinstance(test, pd.Series) else range(len(predictions))
            ),
        )
    except Exception as e:
        raise Exception(f"Error during rolling SARIMA forecast: {e}")

    return predictions_series, rmse
