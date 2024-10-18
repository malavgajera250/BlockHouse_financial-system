from decimal import Decimal
from io import BytesIO
from django.shortcuts import render
from django.http import FileResponse, JsonResponse
from matplotlib import pyplot as plt
from .models import StockData, Prediction #Models
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from django.views.decorators.http import require_http_methods
from datetime import timedelta
from django.utils import timezone 
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


API_KEY = 'VKVIAUHBRVAYJW6M'

def fetch_stock_data(request, symbol='AAPL'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        for date, price_data in time_series.items():
            StockData.objects.update_or_create(
                stock_symbol=symbol,
                date=datetime.strptime(date, '%Y-%m-%d'),
                defaults={
                    'open_price': price_data['1. open'],
                    'close_price': price_data['4. close'],
                    'high_price': price_data['2. high'],
                    'low_price': price_data['3. low'],
                    'volume': price_data['5. volume']
                }
            )
        return JsonResponse({'status': 'success', 'message': f'Stock data for {symbol} fetched and stored.'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Error fetching stock data.'})

def calculate_max_drawdown(df, column):
    cumulative_returns = df[column].expanding(min_periods=1).max()
    drawdown = (df[column] - cumulative_returns) / cumulative_returns
    max_drawdown = drawdown.min() * 100 
    return max_drawdown

def backtest_strategy_view(request, symbol='AAPL'):
    #Get user input for parameters
    initial_investment = float(request.GET.get('initial_investment', 10000))
    short_window = int(request.GET.get('short_window', 50))
    long_window = int(request.GET.get('long_window', 102))

    #Fetch stock data
    stock_data = StockData.objects.filter(stock_symbol=symbol).order_by('date')
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df.set_index('date', inplace=True)

    #Check the data length
    available_data_length = len(df)

    #Check for user input against available data
    if short_window > available_data_length or long_window > available_data_length:
        return render(request, 'finance/backtest.html', {
            'error': f"Please ensure both the short window and long window are less than or equal to {available_data_length} days.",
            'symbol': symbol,
            'initial_investment': initial_investment,
            'short_window': short_window,
            'long_window': long_window,
        })

    #Check for enough data for long_window
    if available_data_length < long_window:
        return render(request, 'finance/backtest.html', {
            'error': "Not enough data to compute moving averages.",
            'symbol': symbol,
            'initial_investment': initial_investment,
            'short_window': short_window,
            'long_window': long_window,
        })

    #Ensure close_price=float
    df['close_price'] = df['close_price'].astype(float)

    #Calculate moving averages
    df['short_mavg'] = df['close_price'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close_price'].rolling(window=long_window).mean()

    #buy/sell signals
    df['signal'] = 0.0
    df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1.0, 0.0)
    df['positions'] = df['signal'].diff()

    cash = initial_investment
    stock_held = 0.0
    trades = 0
    df['portfolio_value'] = cash

    #Track portfolio value over time
    for i, row in df.iterrows():
        if row['positions'] == 1.0:  # Buy signal
            stock_held = cash / row['close_price']
            cash = 0.0
            trades += 1
            df.at[i, 'portfolio_value'] = stock_held * row['close_price']
        elif row['positions'] == -1.0:  # Sell signal
            cash = stock_held * row['close_price']
            stock_held = 0.0
            trades += 1
            df.at[i, 'portfolio_value'] = cash
        else:
            df.at[i, 'portfolio_value'] = stock_held * row['close_price'] + cash

    #Final value
    final_value = cash + stock_held * df['close_price'].iloc[-1]
    total_return = ((final_value - initial_investment) / initial_investment) * 100

    #max drawdown
    df['rolling_max'] = df['portfolio_value'].cummax()
    df['drawdown'] = df['portfolio_value'] / df['rolling_max'] - 1
    max_drawdown = df['drawdown'].min() * 100

    context = {
        'symbol': symbol,
        'initial_investment': initial_investment,
        'short_window': short_window,
        'long_window': long_window,
        'total_return': total_return,
        'final_value': final_value,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'df': df.reset_index().to_dict('records')  #reset index
    }

    return render(request, 'finance/backtest.html', context)


def calculate_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def predict_stock_price(request, symbol='AAPL'):
    #Fetch historical stock data
    stock_data = StockData.objects.filter(stock_symbol=symbol).order_by('date')

    if not stock_data.exists():
        return JsonResponse({'error': 'No stock data available for this symbol.'}, status=404)

    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df.set_index('date', inplace=True)

    #Prepare data for the ML model (X as date, y as close price)
    df['date_num'] = np.arange(len(df))  # Convert dates to numerical values
    X = df[['date_num']]
    y = df['close_price']

    #Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Predict next 30 days
    future_date_num = [[len(df) + i] for i in range(1, 31)]  # Next 30 days
    predicted_prices = model.predict(future_date_num)

    #Store predictions in the database
    predictions = []

    for i, predicted_price in enumerate(predicted_prices):
        pred_date = (timezone.now() + timedelta(days=i + 1)).date()  # Get the next date
        
        # Save the predictions in the database
        Prediction.objects.update_or_create(
            stock_symbol=symbol,
            date=pred_date,
            defaults={'predicted_price': predicted_price}
        )

        # Prepare data for displaying in the table
        predictions.append({
            'date': pred_date,
            'predicted_price': predicted_price
        })

    # Pass predictions to the template
    context = {
        'symbol': symbol,
        'predictions': predictions
    }

    return render(request, 'finance/predictions.html', context)

def generate_report(request, symbol='AAPL'):
    try:
        # Load stock data from the database
        stock_data = StockData.objects.filter(stock_symbol=symbol).order_by('date')

        # Convert to DataFrame
        df = pd.DataFrame(list(stock_data.values('date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume')))

        # Calculate summary stats
        average_close_price = df['close_price'].mean()
        average_open_price = df['open_price'].mean()


        # Convert the last close price to float for prediction
        last_close_price = float(df['close_price'].iloc[-1])
        predicted_price = last_close_price * 1.05

        # Pass the relevant data to the template
        context = {
            'symbol': symbol,
            'stock_data': df.to_dict('records'),
            'average_close_price': average_close_price,
            'predicted_price': predicted_price,
            'average_open_price' : average_open_price,
        }

        return render(request, 'finance/report.html', context)

    except KeyError as e:
        return render(request, 'finance/error.html', {'error_message': f"Column not found: {e}"})
    except FileNotFoundError:
        return render(request, 'finance/error.html', {'error_message': f"Data file for {symbol} not found."})
    


def generate_comparison_report(request, symbol='AAPL'):
    try:
        #Fetch stock data
        stock_data = StockData.objects.filter(stock_symbol=symbol).order_by('date')
        
        #Fetch predicted stock data
        predicted_data = Prediction.objects.filter(stock_symbol=symbol).order_by('date')

        #Convert stock and prediction data to DataFrames
        actual_df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
        predicted_df = pd.DataFrame(list(predicted_data.values('date', 'predicted_price')))

        actual_df['date'] = pd.to_datetime(actual_df['date'])
        predicted_df['date'] = pd.to_datetime(predicted_df['date'])

        #Convert predicted_price to Decimal
        predicted_df['predicted_price'] = predicted_df['predicted_price'].apply(Decimal)

        merged_df = pd.merge(actual_df, predicted_df, on='date', how='outer')

        #summary statistics
        average_actual_price = actual_df['close_price'].mean()
        average_predicted_price = predicted_df['predicted_price'].mean()
        mse = calculate_mse(actual_df['close_price'], predicted_df['predicted_price'])
        
        #actual vs predicted prices
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=merged_df, x='date', y='close_price', label='Actual Price', marker='o')
        sns.lineplot(data=merged_df, x='date', y='predicted_price', label='Predicted Price', marker='x')
        plt.title(f'Actual vs Predicted Stock Prices for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        #PDF report
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:

            pdf.savefig()  #saving into the pdf
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.text(0.1, 0.8, f'Symbol: {symbol}', fontsize=12)
            plt.text(0.1, 0.7, f'Average Actual Price: {average_actual_price:.2f}', fontsize=12)
            plt.text(0.1, 0.6, f'Average Predicted Price: {average_predicted_price:.2f}', fontsize=12)
            plt.text(0.1, 0.5, f'Mean Squared Error: {mse:.2f}', fontsize=12)
            plt.axis('off')
            pdf.savefig()  # saves this text page

        pdf_buffer.seek(0)

        #return PDF as a downloadable file
        response = FileResponse(pdf_buffer, as_attachment=True, filename=f'{symbol}_stock_report.pdf')
        return response

    except KeyError as e:
        return render(request, 'finance/error.html', {'error_message': f"Column not found: {e}"})
    except FileNotFoundError:
        return render(request, 'finance/error.html', {'error_message': f"Data file for {symbol} not found."})

def calculate_mse(y_true, y_pred):
    # Align lengths if y_pred is shorter
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    y_pred = y_pred.apply(Decimal)

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

    

def Backtest_report(request):
    symbol = request.GET.get('symbol', 'AAPL')
    initial_investment = float(request.GET.get('initial_investment', 10000))
    short_window = int(request.GET.get('short_window', 50))
    long_window = int(request.GET.get('long_window', 80))

    stock_data = StockData.objects.filter(stock_symbol=symbol).order_by('date')
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df.set_index('date', inplace=True)

    df['close_price'] = df['close_price'].astype(float)

    df['short_mavg'] = df['close_price'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close_price'].rolling(window=long_window).mean()
    df['signal'] = np.where(df['short_mavg'] > df['long_mavg'], 1.0, 0.0)
    df['positions'] = df['signal'].diff()

    #Backtest logic
    cash = initial_investment
    stock_held = 0.0
    trades = 0
    df['portfolio_value'] = cash
    for i, row in df.iterrows():
        if row['positions'] == 1.0:  #Buy signal
            stock_held = cash / row['close_price']
            cash = 0.0
            trades += 1
        elif row['positions'] == -1.0:  #sell signal
            cash = stock_held * row['close_price']
            stock_held = 0.0
            trades += 1
        df.at[i, 'portfolio_value'] = stock_held * row['close_price'] + cash

    final_value = cash + stock_held * df['close_price'].iloc[-1]
    total_return = ((final_value - initial_investment) / initial_investment) * 100
    max_drawdown = calculate_max_drawdown(df, 'portfolio_value')

    # Generate the PDF
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        #actual vs portfolio value
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='date', y='close_price', label='Close Price', marker='o')
        sns.lineplot(data=df, x='date', y='portfolio_value', label='Portfolio Value', marker='x')
        plt.title(f'Stock vs Portfolio Value for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        pdf.savefig()  #saving into the pdf
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.8, f'Symbol: {symbol}', fontsize=12)
        plt.text(0.1, 0.7, f'Total Return: {total_return:.2f}%', fontsize=12)
        plt.text(0.1, 0.6, f'Final Portfolio Value: {final_value:.2f}', fontsize=12)
        plt.text(0.1, 0.5, f'Max Drawdown: {max_drawdown:.2f}%', fontsize=12)
        plt.text(0.1, 0.4, f'Trades Executed: {trades}', fontsize=12)
        plt.axis('off')
        
        plt.figure(figsize=(10, 6))

        df['rolling_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = df['portfolio_value'] / df['rolling_max'] - 1
        sns.lineplot(data=df, x='date', y='drawdown', label='Drawdown', marker='o')
        plt.title(f'Drawdown over Time for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid()
        pdf.savefig()  # Save this plot to the PDF
        plt.close()
        pdf.savefig()

    pdf_buffer.seek(0)

    # Return a downloadable file
    response = FileResponse(pdf_buffer, as_attachment=True, filename=f'{symbol}_backtest_report.pdf')
    return response
