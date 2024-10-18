# Financial System

This is a Django-based financial system that fetches daily stock prices, allows backtesting of trading strategies, and integrates machine learning models for stock price prediction.

## Features
- Fetches stock data using Alpha Vantage API.
- Implements a stock prediction model using linear regression.
- Backtesting of trading strategies with customizable parameters.
- Generates reports comparing predicted and actual stock prices.

## Technologies Used
- Django
- PostgreSQL
- Matplotlib, Seaborn for data visualization
- Scikit-learn for machine learning

## Getting Started

1. Clone the repository:
git clone <repository-url>


2. Install dependencies:
pip install -r requirements.txt


3. Set up the PostgreSQL database.

4. Run migrations:
python manage.py migrate


5. Start the development server:
python manage.py runserver

