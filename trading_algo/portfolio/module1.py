# 1. MODÈLES DE DONNÉES (ex: portfolio/models.py)
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

# 2. LOGIQUE MÉTIER (ex: portfolio/portfolio.py)
class Portfolio:
    # Le cœur du système
    def __init__(self, cash, positions=None):
        self.cash = cash
        self.positions = positions if positions else {}  # Dict ticker -> Position
        self.transaction_history = []

    def total_value(self, market_prices):
        # Calcule la valeur totale du portefeuille (cash + positions)
        positions_value = sum(pos.current_value(market_prices[ticker]) 
                              for ticker, pos in self.positions.items())
        return self.cash + positions_value

    def generate_rebalance_orders(self, target_allocations, market_prices):
        # Compare l'allocation actuelle à l'allocation cible et génère des ordres
        orders = []
        total_val = self.total_value(market_prices)
        for ticker, target_pct in target_allocations.items():
            target_value = total_val * target_pct
            current_position = self.positions.get(ticker)
            current_value = current_position.current_value(market_prices[ticker]) if current_position else 0
            diff_value = target_value - current_value

            if abs(diff_value) > 0.01 * total_val:  # Seuil de tolérance de 1%
                # Calculer la quantité à trader (simplifié, sans tenir compte du cash)
                # ... (logique de calcul)
                pass
        return orders

