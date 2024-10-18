
from django.urls import path
from .views import fetch_stock_data, backtest_strategy_view, generate_report, predict_stock_price, generate_comparison_report, Backtest_report

urlpatterns = [
    path('', generate_report, name='index'),
    path('fetch-stock-data/<str:symbol>/', fetch_stock_data, name='fetch_stock_data'),
    path('predict-stock-price/<str:symbol>/', predict_stock_price, name='predict_stock_price'),
    path('backtest/<str:symbol>/', backtest_strategy_view, name='backtest_strategy'),
    path('predict/<str:symbol>/', predict_stock_price, name='predict_stock_price_page'),
    path('generate-comparison-report/<str:symbol>/', generate_comparison_report, name='generate_comparison_report'),
    path('backtest_report/', Backtest_report, name='Backtest_report'),

]
