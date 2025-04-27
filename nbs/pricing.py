import QuantLib as ql
import numpy as np

print(ql.__version__ )

def price_american_call_pde(
    spot, strike, rate, div_yield, maturity, volatility, dividends, dividend_times
):
    """
    Prices an American call option using the PDE method in QuantLib.

    Args:
        spot (float): Current spot price.
        strike (float): Strike price.
        rate (float): Risk-free interest rate.
        div_yield (float): Dividend yield.
        maturity (float): Time to maturity in years.
        volatility (float): Volatility.
        dividends (list of float): List of dividend amounts.
        dividend_times (list of float): List of times when dividends are paid (in years).

    Returns:
        float: American call option price.
    """

    calculation_date = ql.Date(1, 1, 2024)  # Arbitrary calculation date
    ql.Settings.instance().evaluationDate = calculation_date

    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, rate, day_count)
    )
    div_yield_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, div_yield, day_count)
    )
    volatility_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.AmericanExercise(
        calculation_date, calculation_date + ql.Period(int(maturity * 365), ql.Days)
    )

    option = ql.VanillaOption(payoff, exercise)

    process = ql.BlackScholesMertonProcess(
        spot_handle, div_yield_handle, rate_handle, volatility_handle
    )

    dividend_dates_ql = [calculation_date + ql.Period(int(t * 365), ql.Days) for t in dividend_times]

    dividend_schedule = ql.DividendSchedule()

    for date, div in zip(dividend_dates_ql, dividends):
        dividend_schedule.append(ql.FixedDividend(div, date))

    engine = ql.FDAmericanEngine(process, timeSteps=100, gridPoints=100, dividendSchedule = dividend_schedule)
    option.setPricingEngine(engine)

    return option.NPV()

def implied_volatility_american_call_pde(
    target_price, spot, strike, rate, div_yield, maturity, dividends, dividend_times, guess=0.2
):
    """
    Calculates implied volatility of an American call option using the PDE method.

    Args:
        target_price (float): Market price of the option.
        spot (float): Current spot price.
        strike (float): Strike price.
        rate (float): Risk-free interest rate.
        div_yield (float): Dividend yield.
        maturity (float): Time to maturity in years.
        dividends (list of float): List of dividend amounts.
        dividend_times (list of float): List of times when dividends are paid (in years).
        guess (float): Initial guess for volatility.

    Returns:
        float: Implied volatility.
    """

    def objective_function(vol):
        price = price_american_call_pde(
            spot, strike, rate, div_yield, maturity, vol, dividends, dividend_times
        )
        return price - target_price

    from scipy.optimize import newton

    try:
        implied_vol = newton(objective_function, guess)
        return implied_vol
    except RuntimeError:
        return None  # Return None if Newton's method fails to converge.

# Example usage:
spot = 100.0
strike = 100.0
rate = 0.05
div_yield = 0.02
maturity = 1.0
volatility = 0.2
dividends = [2.0, 1.5]
dividend_times = [0.25, 0.75]

option_price = price_american_call_pde(
    spot, strike, rate, div_yield, maturity, volatility, dividends, dividend_times
)
print(f"American call option price: {option_price}")

target_price = option_price * 1.05 # Adding 5% to the calculated price to test implied vol.
implied_vol = implied_volatility_american_call_pde(
    target_price, spot, strike, rate, div_yield, maturity, dividends, dividend_times, guess=0.2
)

if implied_vol is not None:
    print(f"Implied volatility: {implied_vol}")
else:
    print("Implied volatility calculation failed.")