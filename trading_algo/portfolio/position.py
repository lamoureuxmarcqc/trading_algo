class Position:
    # Représente une ligne dans le portefeuille (ex: 10 actions Apple)
    def __init__(self, ticker, quantity, average_price):
        self.ticker = ticker
        self.quantity = quantity
        self.average_price = average_price

    def current_value(self, current_price):
        return self.quantity * current_price

    def unrealized_pnl(self, current_price):
        return self.quantity * (current_price - self.average_price)

class Order:
    # Représente un ordre à exécuter
    def __init__(self, ticker, order_type, quantity, limit_price=None):
        self.ticker = ticker
        self.order_type = order_type  # 'buy' ou 'sell'
        self.quantity = quantity
        self.limit_price = limit_price
        self.status = 'pending'

