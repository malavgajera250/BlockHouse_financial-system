from django.db import models

# Create your models here.
class StockData(models.Model):
    stock_symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()

    def __str__(self):
        return f"{self.stock_symbol} - {self.date}"
    
class Prediction(models.Model):
    date = models.DateField()
    stock_symbol = models.CharField(max_length=10)
    predicted_price = models.FloatField()

    def __str__(self):
        return f"Prediction for {self.stock_symbol} on {self.date}"
