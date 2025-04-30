import pandas as pd
from yahooquery import Ticker
import yfinance as yf



def get_name(ticker):
    company = Ticker(ticker)
    return company.price.get(ticker, {}).get('longName')


def get_change(ticker):
    company = Ticker(ticker)
    return round(company.price.get(ticker, {}).get('regularMarketChangePercent'),4) * 100


def get_div(ticker):
    company = Ticker(ticker)
    return company.summary_detail.get(ticker, {}).get('dividendYield') * 100


def get_beta(ticker):
    company = Ticker(ticker)
    return company.summary_detail.get(ticker, {}).get('beta')

def get_price(ticker):
    company = Ticker(ticker)
    return company.price[ticker]['regularMarketPrice']


def get_market_cap(ticker):
    company = Ticker(ticker)
    return company.price.get(ticker, {}).get('marketCap')


def get_data(ticker):
    company = Ticker(ticker)
    return company.history(period='5y', interval='1d')['adjclose']


def get_income_statement(ticker):
    company = Ticker(ticker)
    income_statement = company.income_statement()
    income_statement = income_statement[income_statement['periodType'] =='12M']
    income_statement = income_statement[['asOfDate', 'periodType', 'NetIncome','TotalRevenue', 'BasicEPS']].dropna(how='any', axis = 0).drop_duplicates()
    return income_statement


def get_cash_flow_statement(ticker):
    company = Ticker(ticker)
    cash_flow = company.cash_flow()
    cash_flow = cash_flow[cash_flow['periodType'] =='12M']
    cash_flow = cash_flow[['asOfDate','CapitalExpenditure','FreeCashFlow']].dropna(how='any', axis = 0).drop_duplicates()

    return cash_flow

def get_balance_sheet(ticker):
    company = Ticker(ticker)
    balance_sheet=company.balance_sheet()
    balance_sheet = balance_sheet[balance_sheet['periodType'] =='12M']
    balance_sheet = balance_sheet[['asOfDate', 'TotalAssets', 'TotalDebt']].dropna(how='any', axis = 0).drop_duplicates()
    return balance_sheet



def get_shares_outstanding(ticker):
    company = Ticker(ticker)
    return company.key_stats.get(ticker, {}).get('sharesOutstanding')




def get_pe_ratio(ticker):
    company = Ticker(ticker)
    return round(company.summary_detail.get(ticker, {}).get('trailingPE'),2)


def get_wacc(ticker, risk_free_rate, market_return):
        stock = Ticker(ticker)
        # Step 2: Get market value of equity
        price = stock.price[ticker]['regularMarketPrice']
        shares_outstanding = stock.key_stats[ticker]['sharesOutstanding']
        market_equity = price * shares_outstanding
        # Step 3: Get book value of debt (proxy for market debt)
        balance_sheet = stock.balance_sheet(trailing=False)
        latest_bs = balance_sheet.iloc[-1]
        total_debt = latest_bs['TotalDebt']


        beta = stock.key_stats[ticker]['beta']
        risk_free_rate = risk_free_rate 
        market_return = market_return 
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

        income_statement = stock.income_statement(trailing=True)

        latest_is = income_statement.iloc[-1]

        interest_expense = latest_is['InterestExpense']

        cost_of_debt = interest_expense / total_debt if total_debt != 0 else 0

        income_tax_expense = latest_is['EBIT']
        ebt = latest_is['EBIT']
        tax_rate = latest_is['TaxRateForCalcs']


        # Step 7: WACC Calculation
        total_value = market_equity + total_debt
        equity_weight = market_equity / total_value
        debt_weight = total_debt / total_value

        wacc = (
            equity_weight * cost_of_equity
            + debt_weight * cost_of_debt * (1 - tax_rate)
        )

        return wacc


def get_intrinsic_value(ticker, con, mod, opt, wacc):
    stock = Ticker(ticker)

    try:
        eps_ttm = stock.key_stats[ticker]['trailingEps']
        price = stock.price[ticker]['regularMarketPrice']
        shares_outstanding = stock.key_stats[ticker]['sharesOutstanding']

        # Free Cash Flow
        cash_flow = stock.cash_flow(trailing=False)
        fcf_data = cash_flow['FreeCashFlow']
        latest_fcf = fcf_data.values[-1]

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

    scenarios = {
        "Conservative": {"eps_growth": con, "fcf_growth": con, "pe": 18, "terminal": 0.02, "wacc": wacc},
        "Moderate": {"eps_growth": mod, "fcf_growth": mod, "pe": 21, "terminal": 0.025, "wacc": wacc},
        "Optimistic": {"eps_growth": opt, "fcf_growth": opt, "pe": 25, "terminal": 0.03, "wacc": wacc}
    }

    years = 5
    results = []

    for label, params in scenarios.items():
        eps_growth = params['eps_growth']
        fcf_growth = params['fcf_growth']
        target_pe = params['pe']
        terminal_growth = params['terminal']
        wacc = params['wacc']

        # === P/E Model
        future_eps = eps_ttm * (1 + eps_growth) ** years
        future_price_pe = future_eps * target_pe
        intrinsic_pe = future_price_pe / (1 + wacc) ** years

        # === DCF Model
        forecast_fcfs = [latest_fcf * (1 + fcf_growth) ** i for i in range(1, years + 1)]
        terminal_value = forecast_fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        discounted_fcfs = [fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(forecast_fcfs)]
        discounted_terminal = terminal_value / (1 + wacc) ** years
        intrinsic_dcf_total = sum(discounted_fcfs) + discounted_terminal
        intrinsic_dcf_per_share = intrinsic_dcf_total / shares_outstanding

        results.append({
            "Scenario": label,
            "EPS Growth": f"{eps_growth:.0%}",
            "FCF Growth": f"{fcf_growth:.0%}",
            "Target P/E": target_pe,
            "Terminal Growth": f"{terminal_growth:.1%}",
            "WACC": f"{wacc:.1%}",
            "Intrinsic Value (DCF)": f"${intrinsic_dcf_per_share:.2f}",
            "Intrinsic Value (P/E)": f"${intrinsic_pe:.2f}",
            "Current Price": f"${price:.2f}"
        })

    return pd.DataFrame(results)



def get_valuation_metrics(ticker):
    company = Ticker(ticker)
    valuation = ['asOfDate', 'PsRatio', 'ForwardPeRatio', 'PeRatio', 'PsRatio', 'EnterpriseValue']
    value_df = company.valuation_measures[valuation].dropna()
    return value_df



def get_recommendations(ticker):
    company = Ticker(ticker)
    df = pd.DataFrame(company.recommendation_trend)

    # Define a custom sorting order for the 'period' column
    period_order = ['-3m', '-2m','-1m','0m']
    df['period'] = pd.Categorical(df['period'], categories=period_order, ordered=True)

    # Sort the DataFrame by the 'period' column
    df = df.sort_values('period')
    return df


