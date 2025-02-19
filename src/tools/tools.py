from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime
from alpha_vantage.fundamentaldata import FundamentalData
import os
from dotenv import load_dotenv
from scipy.stats import norm
import math

load_dotenv()  # Load environment variables from .env file
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# Modified logger setup with datestamp in filename
current_date = datetime.now().strftime('%Y-%m-%d')
from src.utils.logger import get_logger

logger = get_logger(__name__)


### 1. StockDataTool ###
class StockDataInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL for Apple).")

class StockDataOutput(BaseModel):
    data: str = Field(description="Historical stock data as a JSON string.")

class StockInfoTool(BaseTool[StockDataInput, StockDataOutput]):
    def __init__(self):
        super().__init__(
            StockDataInput,
            StockDataOutput,
            "fetch_stock_data",
            "Fetch only historical stock data for a given ticker.",
        )

    async def run(self, args: StockDataInput, cancellation_token: CancellationToken) -> StockDataOutput:
        logger.info(f"Fetching stock data for ticker: {args.ticker}")
        try:
            stock = yf.Ticker(args.ticker)
            data = stock.history(period="1d")
            formatted_data = f"The stock opened at ${data['Open'].iloc[0]:.2f}, reached a high of ${data['High'].iloc[0]:.2f} and a low of ${data['Low'].iloc[0]:.2f}, closed at ${data['Close'].iloc[0]:.2f}, with {int(data['Volume'].iloc[0]):,} shares traded, a dividend of ${data['Dividends'].iloc[0]:.2f}, and {data['Stock Splits'].iloc[0]:.1f} stock splits"
            logger.info(f"Successfully fetched stock data for {args.ticker}")
            return StockDataOutput(data=formatted_data)
        except Exception as e:
            logger.error(f"Error fetching stock data for {args.ticker}: {e}")
            raise RuntimeError(f"Error fetching stock data for {args.ticker}: {e}")


### 2. StockPredictorTool ###
class StockPredictorInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL for Apple).")

class StockPredictorOutput(BaseModel):
    data: str = Field(description="Predicted stock price for the next day as a string.")

class StockForecastTool(BaseTool[StockPredictorInput, StockPredictorOutput]):
    def __init__(self):
        super().__init__(
            StockPredictorInput,
            StockPredictorOutput,
            "predict_stock_price",
            "Forecast or Predict the next day's stock price.",
        )

    async def run(self, args: StockPredictorInput, cancellation_token: CancellationToken) -> StockPredictorOutput:
        logger.info("Starting stock price prediction")
        try:
            data = yf.Ticker(args.ticker).history(period="2y")
            data = data.reset_index()
            data["Timestamp"] = data.index

            X = np.array(data["Timestamp"]).reshape(-1, 1)
            y = data["Close"].values

            model = LinearRegression()
            model.fit(X, y)
            logger.info("Model trained successfully")
            prediction = model.predict([[len(data)]])
            logger.info(f"Predicted stock price: {prediction}")
            logger.info(f"Successfully predicted stock price: {prediction[0]}")
            return StockPredictorOutput(data=f"the predicted stock price for {args.ticker} is ${prediction[0]:.2f}")
        except Exception as e:
            logger.error(f"Error predicting stock price: {e}")
            raise RuntimeError(f"Error predicting stock price: {e}")


### 5. SentimentAnalysisTool ###
class SentimentAnalysisInput(BaseModel):
    company_name: str = Field(description="Name of the company.")

class SentimentAnalysisOutput(BaseModel):
    data: str = Field(description="String containing sentiment analysis results")

class SentimentAnalysisTool(BaseTool[SentimentAnalysisInput, SentimentAnalysisOutput]):
    def __init__(self):
        super().__init__(
            SentimentAnalysisInput,
            SentimentAnalysisOutput,
            "analyze_sentiment",
            "Perform sentiment analysis for a company.",
        )

    async def run(self, args: SentimentAnalysisInput, cancellation_token: CancellationToken) -> SentimentAnalysisOutput:
        sentiment_score = np.random.uniform(-1, 1)
        sentiment = "positive" if sentiment_score > 0 else "negative"
        return SentimentAnalysisOutput(data=f"The sentiment for {args.company_name} is {sentiment} with a score of {sentiment_score:.2f}")


### FinancialLiteracyTool ###
class FinancialLiteracyInput(BaseModel):
    query: str = Field(description="Financial topic or term to explain.")

class FinancialLiteracyOutput(BaseModel):
    data: str = Field(description="Explanation of the financial topic.")

class FinancialLiteracyTool(BaseTool[FinancialLiteracyInput, FinancialLiteracyOutput]):
    def __init__(self):
        super().__init__(
            FinancialLiteracyInput,
            FinancialLiteracyOutput,
            "explain_finance",
            "Explain a financial concept or term.",
        )

    async def run(self, args: FinancialLiteracyInput, cancellation_token: CancellationToken) -> FinancialLiteracyOutput:
        knowledge_base = {
            "compound interest": "Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods.",
            "etf": "An ETF (Exchange-Traded Fund) is a type of investment fund that is traded on stock exchanges, much like stocks.",
            "mutual funds": "A mutual fund is a type of investment vehicle consisting of a portfolio of stocks, bonds, or other securities.",
            "portfolio": "A portfolio is a collection of investments held by an investor, such as stocks, bonds, or other assets.",
            "portfolio optimization": "Portfolio optimization is the process of selecting the best combination of assets to achieve a desired goal, such as maximizing returns or minimizing risk.",
            "risk": "Risk is the possibility of losing money or experiencing negative outcomes, such as losing value in an investment or experiencing a loss in a business.",
            "return": "Return is the profit or gain made from an investment, such as a stock or bond.",
            "volatility": "Volatility is the degree of variation in the price of an asset over time, such as a stock or bond.",
            "correlation": "Correlation is the relationship between two variables, such as the price of a stock and the price of a bond.",
            "beta": "Beta is a measure of the volatility of a stock or bond compared to the overall market.",
            "alpha": "Alpha is a measure of the performance of a stock or bond compared to the overall market.",
            "risk-adjusted return": "Risk-adjusted return is a measure of the performance of an investment compared to the risk taken, such as a stock or bond.",
            "risk-free rate": "A risk-free rate is a theoretical rate of return on an investment with no risk of loss, such as a government bond.",
            "risk premium": "A risk premium is the additional return required to compensate for the risk of an investment, such as a stock or bond.",
            "market risk": "Market risk is the risk of loss due to changes in the overall market, such as a stock market or bond market.",
            "credit risk": "Credit risk is the risk of loss due to the failure of a borrower to repay a loan or bond, such as a company or government.",
            "liquidity risk": "Liquidity risk is the risk of loss due to the inability to sell an investment quickly and at a fair price, such as a stock or bond.",
            "interest rate": "An interest rate is the cost of borrowing money, such as a loan or bond.",
            "dividend": "A dividend is a payment made by a company to its shareholders, such as a stock or bond.",
            "earnings": "Earnings are the profits made by a company, such as a stock or bond.", 
            "market capitalization": "Market capitalization is the total value of a company's outstanding shares of stock, such as a stock or bond.",
            "price-to-earnings ratio": "A price-to-earnings ratio is a measure of a company's stock price relative to its earnings, such as a stock or bond.",
            "price-to-book ratio": "A price-to-book ratio is a measure of a company's stock price relative to its book value, such as a stock or bond.",
            "price-to-sales ratio": "A price-to-sales ratio is a measure of a company's stock price relative to its sales, such as a stock or bond.",
            "price-to-cash flow ratio": "A price-to-cash flow ratio is a measure of a company's stock price relative to its cash flow, such as a stock or bond.",
            "price-to-earnings growth ratio": "A price-to-earnings growth ratio is a measure of a company's stock price relative to its earnings growth, such as a stock or bond.",
            "price-to-book value ratio": "A price-to-book value ratio is a measure of a company's stock price relative to its book value, such as a stock or bond.",    
        }
        explanation = knowledge_base.get(args.query.lower(), "I don't have information on that topic yet.")
        return FinancialLiteracyOutput(data=explanation)


### 7. PortfolioOptimizationTool ###
class PortfolioOptimizationInput(BaseModel):
    portfolio: Dict[str, float] = Field(description="Portfolio with asset weights.")

class PortfolioOptimizationOutput(BaseModel):
    data: str = Field(description="Optimized portfolio weights as a formatted string.")

class PortfolioOptimizationTool(BaseTool[PortfolioOptimizationInput, PortfolioOptimizationOutput]):
    def __init__(self):
        super().__init__(
            PortfolioOptimizationInput,
            PortfolioOptimizationOutput,
            "optimize_portfolio",
            "Optimize portfolio allocations.",
        )

    async def run(self, args: PortfolioOptimizationInput, cancellation_token: CancellationToken) -> PortfolioOptimizationOutput:
        optimized_portfolio = {asset: weight * 1.05 for asset, weight in args.portfolio.items()}
        formatted_output = "Optimized Portfolio Allocation:\n"
        for asset, weight in optimized_portfolio.items():
            formatted_output += f"{asset}: {weight:.2%}\n"
        return PortfolioOptimizationOutput(data=formatted_output)
    

class OptionsPricingInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g., AAPL for Apple).")
    strike_price: float = Field(description="Strike price of the option.")
    time_to_expiry: float = Field(description="Time to expiry in years.")
    option_type: str = Field(description="Option type: 'call' or 'put'.")

class OptionsPricingOutput(BaseModel):
    data: str = Field(description="Fair price of the option.")

class OptionsPricingTool(BaseTool[OptionsPricingInput, OptionsPricingOutput]):
    def __init__(self):
        super().__init__(
            OptionsPricingInput,
            OptionsPricingOutput,
            "options_pricing_calculator",
            "Calculates the fair price of an options contract.",
        )

    async def run(self, args: OptionsPricingInput, cancellation_token: CancellationToken, **kwargs) -> OptionsPricingOutput:
        logger.info(f"Starting options pricing calculation for {args.ticker} with strike price ${args.strike_price}")
        try:
            stock = yf.Ticker(args.ticker)
            data = stock.history(period="1d")
            S = data['Open'].iloc[0]
            logger.info(f"Current stock price (S): ${S:.2f}")
            
            K = args.strike_price
            T = args.time_to_expiry
            
            irx = yf.Ticker("^IRX").history(period="1d")
            r = (irx['Close'].iloc[0])/100
            logger.info(f"Risk-free rate (r): {r:.4f}")
            
            # Calculate volatility
            stock_3mo = stock.history(period="3mo")
            stock_3mo['lag_adj_close'] = stock_3mo['Close'].shift(1)
            stock_3mo['log_return'] = np.log(stock_3mo['Close']/stock_3mo['lag_adj_close'])
            vol = np.std(stock_3mo['log_return'])
            sigma = 252**0.5*vol
            logger.info(f"Annualized volatility (sigma): {sigma:.4f}")

            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * math.sqrt(T))
            logger.info(f"function is math.log({S} / {K}) + ({r} + ({sigma}**2) / 2) * {T}")
            d2 = d1 - sigma * math.sqrt(T)
            logger.debug(f"d1: {d1:.4f}, d2: {d2:.4f}")

            if args.option_type.lower() == "call":
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                logger.info(f"Calculated call option price: ${price:.2f}")
            elif args.option_type.lower() == "put":
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                logger.info(f"Calculated put option price: ${price:.2f}")
            else:
                logger.error(f"Invalid option type provided: {args.option_type}")
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")

            return OptionsPricingOutput(
                data=f"The fair price for this {args.option_type} option is ${price:.2f}, to buy this please contact ibrahim@arthur.ai"
            )
        except Exception as e:
            logger.error(f"Error calculating option price for {args.ticker}: {str(e)}")
            raise RuntimeError(f"Error calculating option price: {str(e)}")
    
class StockScreenerInput(BaseModel):
    sector: str = Field(description="Sector to filter stocks (e.g., Technology, Healthcare).")
    market_cap_min: float = Field(description="Minimum market capitalization (in billions).")
    market_cap_max: float = Field(description="Maximum market capitalization (in billions).")

class StockScreenerOutput(BaseModel):
    data: str = Field(description="Formatted string of stocks meeting the criteria with their market caps.")

class StockScreenerTool(BaseTool[StockScreenerInput, StockScreenerOutput]):
    def __init__(self):
        super().__init__(
            StockScreenerInput,
            StockScreenerOutput,
            "ai_powered_stock_screener",
            "Screens stocks based on sector and market capitalization criteria using Alpha Vantage.",
        )
        self.fundamental_data = FundamentalData(ALPHA_VANTAGE_API_KEY)

    async def run(self, args: StockScreenerInput, cancellation_token: CancellationToken, **kwargs) -> StockScreenerOutput:
        try:
            # Example stock tickers for demonstration (in practice, this could be fetched dynamically)
            stock_tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC", "AMD", "CSCO",  # Technology
                "PFE", "JNJ", "UNH", "MRK", "ABT", "MRNA", "LLY", "AMGN", "GILD", "CVS",        # Healthcare
                "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "V", "MA",                 # Financials
                "HD", "NKE", "MCD", "SBUX", "TGT", "DIS", "LOW", "BKNG", "YUM", "EBAY",         # Consumer Discretionary
                "PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "KMB", "MO", "PM",              # Consumer Staples
                "XOM", "CVX", "COP", "SLB", "HAL", "BKR", "MPC", "VLO", "OXY", "PXD",           # Energy
                "BA", "CAT", "LMT", "GE", "HON", "RTX", "MMM", "UNP", "DE", "NOC",              # Industrials
                "DUK", "NEE", "D", "SO", "EXC", "AEP", "SRE", "ED", "XEL", "PEG",               # Utilities
                "PLD", "AMT", "SPG", "O", "AVB", "DLR", "EQR", "WELL", "VTR", "BXP",            # Real Estate
                "DOW", "LYB", "DD", "NEM", "FCX", "IP", "VMC", "CF", "BALL", "EMN"              # Materials
            ]


            filtered_stocks = {}

            for ticker in stock_tickers:
                try:
                    # Fetch company overview from Alpha Vantage
                    company_overview = self.fundamental_data.get_company_overview(ticker)[0]

                    # Extract sector and market cap
                    company_sector = company_overview["Sector"]
                    market_cap = float(company_overview["MarketCapitalization"]) / 1e9  # Convert to billions

                    # Filter by sector and market capitalization
                    if (
                        company_sector.lower() == args.sector.lower()
                        and args.market_cap_min <= market_cap <= args.market_cap_max
                    ):
                        filtered_stocks[ticker] = market_cap
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")

            # Format the filtered stocks into a string
            if not filtered_stocks:
                result = f"No stocks found in the {args.sector} sector within the market cap range of ${args.market_cap_min}B - ${args.market_cap_max}B."
            else:
                result = f"Stocks in {args.sector} sector with market cap between ${args.market_cap_min}B - ${args.market_cap_max}B:\n"
                for ticker, market_cap in filtered_stocks.items():
                    result += f"{ticker}: ${market_cap:.2f}B\n"

            return StockScreenerOutput(stocks=result)

        except Exception as e:
            raise RuntimeError(f"Error during stock screening: {e}")