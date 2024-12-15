import os
from typing import Dict, Any, List
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json

def get_financial_metrics(
    ticker: str,
    report_period: str,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial metrics from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError("No financial metrics returned")
    return financial_metrics

def get_cash_flow_statements(ticker: str):
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)

        # Get cash flow data (from financials)
        cash_flow = stock.cashflow
        # Handle missing data points gracefully using try-except
        try:
            data = {
                "ticker": ticker,
                "net_income": cash_flow.loc['Net Income From Continuing Operations'].iloc[0] if 'Net Income From Continuing Operations' in cash_flow.index else None,
                "depreciation_and_amortization": cash_flow.loc['Depreciation And Amortization'].iloc[0] if 'Depreciation And Amortization' in cash_flow.index else None,
                "share_based_compensation": cash_flow.loc['Stock Based Compensation'].iloc[0] if 'Stock Based Compensation' in cash_flow.index else None,  # Updated field name
                "net_cash_flow_from_operations": cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else None,
                "net_cash_flow_from_investing": cash_flow.loc['Investing Cash Flow'].iloc[0] if 'Investing Cash Flow' in cash_flow.index else None,
                "capital_expenditure": cash_flow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cash_flow.index else None,
                "property_plant_and_equipment": None,  # Not available
                "business_acquisitions_and_disposals": cash_flow.loc['Net Business Purchase And Sale'].iloc[0] if 'Net Business Purchase And Sale' in cash_flow.index else None,
                "investment_acquisitions_and_disposals": cash_flow.loc['Net Investment Purchase And Sale'].iloc[0] if 'Net Investment Purchase And Sale' in cash_flow.index else None,
                "net_cash_flow_from_financing": cash_flow.loc['Financing Cash Flow'].iloc[0] if 'Financing Cash Flow' in cash_flow.index else None,
                "issuance_or_repayment_of_debt_securities": cash_flow.loc['Net Issuance Payments Of Debt'].iloc[0] if 'Net Issuance Payments Of Debt' in cash_flow.index else None,
                "issuance_or_purchase_of_equity_shares": cash_flow.loc['Net Common Stock Issuance'].iloc[0] if 'Net Common Stock Issuance' in cash_flow.index else None,  # Updated field name
                "dividends_and_other_cash_distributions": cash_flow.loc['Common Stock Dividend Paid'].iloc[0] if 'Common Stock Dividend Paid' in cash_flow.index else None,
                "change_in_cash_and_equivalents": cash_flow.loc['Changes In Cash'].iloc[0] if 'Changes In Cash' in cash_flow.index else None,
                "effect_of_exchange_rate_changes": None,  # Not available
                "ending_cash_balance": cash_flow.loc['End Cash Position'].iloc[0] if 'End Cash Position' in cash_flow.index else None,
                "free_cash_flow": cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else None
            }

        except KeyError as e:
            # In case a specific key is missing
            print(f"Missing data for key: {e}")
            data = {key: None for key in [
                'net_income', 'depreciation_and_amortization', 'share_based_compensation',
                'net_cash_flow_from_operations', 'net_cash_flow_from_investing', 'capital_expenditure',
                'property_plant_and_equipment', 'business_acquisitions_and_disposals',
                'investment_acquisitions_and_disposals', 'net_cash_flow_from_financing',
                'issuance_or_repayment_of_debt_securities', 'issuance_or_purchase_of_equity_shares',
                'dividends_and_other_cash_distributions', 'change_in_cash_and_equivalents',
                'effect_of_exchange_rate_changes', 'ending_cash_balance', 'free_cash_flow'
            ]}
            data["ticker"] = ticker
        
        # Return data in desired format
        return {
            "cash_flow_statements": [data]
        }
    
    except Exception as e:
        # Catch all other errors and log
        print(f"An error occurred while fetching cash flow data: {e}")
        return None

def get_insider_trades(ticker, limit=10):
    try:
        # Fetch the data for the given ticker
        stock = yf.Ticker(ticker)
        
        # Get insider transactions data (this may not be available for all tickers)
        insider_data = stock.insider_transactions
        
        # Check if data is available
        if insider_data is None or insider_data.empty:
            raise ValueError("No insider trades returned")
        
        # Convert 'Transaction' and 'Start Date' to datetime
        insider_data['Transaction'] = pd.to_datetime(insider_data['Transaction'], errors='coerce')
        insider_data['Start Date'] = pd.to_datetime(insider_data['Start Date'], errors='coerce')
        
        # Get the date one year ago
        one_year_ago = datetime.now() - timedelta(days=365)
        
        # Filter the insider transactions to only include those within the last year
        insider_data = insider_data[insider_data['Start Date'] >= one_year_ago]

        # Sort by 'Transaction' date in descending order (latest transactions first)
        insider_data = insider_data.sort_values(by='Transaction', ascending=False)
        
        # Limit to the latest `limit` transactions
        insider_data = insider_data.head(limit)

        # Structure the data to match your required JSON format
        insider_trades = {
            "insider_trades": []
        }
        
        # Loop through insider transactions and structure the data
        for _, row in insider_data.iterrows():
            # Convert 'Start Date' to datetime, then format to string (ISO 8601 format)
            filing_date = pd.to_datetime(row['Start Date'], errors='coerce')

            # Only apply strftime if the date is not NaT (Not a Time)
            filing_date = filing_date.strftime('%Y-%m-%d') if not pd.isna(filing_date) else None
            
            # Determine whether the transaction was a "buy", "sell", or "gift" based on the 'Text' column
            if "Sale" in row['Text']:
                transaction_type = "sell"
            elif "Gift" in row['Text'] or not row['Text']:
                # Skip rows with "Gift" transactions
                continue
            else:
                transaction_type = "buy"
            
            # Structure the required fields and exclude unnecessary ones
            trade = {
                "name": row['Insider'],  # Insider's name
                "title": row['Position'],  # Insider's title
                "is_board_director": row['Ownership'] == 'D',  # Assuming 'D' means Director
                "transaction_shares": row['Shares'],
                "filing_date": filing_date,  # Use formatted date
                "security_title": "Common Stock",  # Assuming this is always the case
                "transaction_type": transaction_type  # Added field
            }
            insider_trades["insider_trades"].append(trade)
        
        return insider_trades
    
    except ValueError as e:
        # If no insider trades are found or another error occurs, raise an exception
        raise ValueError("No insider trades returned") from e
    except Exception as e:
        # General exception handling
        raise Exception(f"An error occurred while processing insider data: {str(e)}") from e

def get_market_cap(ticker: str) -> float:
    try:
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        # Get the market capitalization
        market_cap = stock.info.get('marketCap', None)
        
        if market_cap is None:
            print(f"Market cap not available for {ticker}")
        return market_cap

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    try:
        # Fetch the data for the given ticker
        interval = "1d"  # Daily data (1-day interval)
        stock = yf.Ticker(ticker)

        # Get historical market data for the specified date range
        price_data = stock.history(start=start_date, end=end_date)

        # Check if price data is available
        if price_data.empty:
            raise ValueError("No price data returned")

        # Structure the data to match your required JSON format
        prices = []

        # Loop through each row of the historical data
        for index, row in price_data.iterrows():
            # Format the data as required
            price = {
                "open": row['Open'],
                "close": row['Close'],
                "high": row['High'],
                "low": row['Low'],
                "volume": row['Volume'],
                "time": index.strftime('%Y-%m-%dT%H:%M:%S')  # Use ISO 8601 format for time
            }
            prices.append(price)

        return {"prices": prices}

    except Exception as e:
        # General exception handling
        raise Exception(f"An error occurred while fetching prices: {str(e)}") from e

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

# Update the get_price_data function to use the new functions
def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

def calculate_confidence_level(signals: dict) -> float:
    """Calculate confidence level based on the difference between SMAs."""
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value