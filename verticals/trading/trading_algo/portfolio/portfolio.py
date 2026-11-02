import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

# Gestion des imports optionnels pour l'affichage console riche
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Order:
    ticker: str
    order_type: str  # 'buy' or 'sell'
    quantity: float
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: "")
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = 'pending'
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_time: Optional[datetime] = None
    fees: float = 0.0
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ['timestamp', 'execution_time']:
            if data[key] and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data

@dataclass
class Position:
    ticker: str
    quantity: float
    average_price: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    def current_value(self, current_price: float) -> float:
        if current_price is None:
            return 0.0
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        if current_price is None:
            return 0.0
        return (current_price - self.average_price) * self.quantity
    
    def unrealized_pnl_percent(self, current_price: float) -> float:
        if current_price is None or self.average_price == 0:
            return 0.0
        return (current_price - self.average_price) / self.average_price

    def update_position(self, qty: float, price: float, fees: float, is_buy: bool):
        """Logique de calcul du prix moyen pondéré (WAP)"""
        if is_buy:
            total_cost = (self.quantity * self.average_price) + (qty * price) + fees
            self.quantity += qty
            self.average_price = total_cost / self.quantity if self.quantity > 0 else 0
        else:
            if qty > self.quantity + 1e-9:  # Marge d'erreur flottante
                raise ValueError(f"Vente de {qty} {self.ticker} impossible (Avoir: {self.quantity})")
            self.quantity -= qty
        self.last_updated = datetime.now()

class Portfolio:
    def __init__(self, cash: float, name: str = "Main Portfolio"):
        self.name = name
        self.initial_capital = cash
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.transaction_history: List[Order] = []
        self.performance_history: List[Dict] = []  # Liste pour la courbe d'équité
        self.last_updated = datetime.now()

    # --- PERSISTENCE ---

    def save_to_file(self, filepath: str):
        """Sauvegarde l'état complet en JSON."""
        data = {
            "name": self.name,
            "cash": float(self.cash),
            "initial_capital": float(self.initial_capital),
            "positions": {t: asdict(pos) for t, pos in self.positions.items()},
            "history": [o.to_dict() for o in self.transaction_history],
            "performance_history": self.performance_history
        }
        
        # Nettoyage des dates dans les positions
        for t in data["positions"]:
            if isinstance(data["positions"][t]["last_updated"], datetime):
                data["positions"][t]["last_updated"] = data["positions"][t]["last_updated"].isoformat()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['Portfolio']:
        """Reconstruit le Portfolio depuis un JSON."""
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        portfolio = cls(cash=float(data['cash']), name=data['name'])
        portfolio.initial_capital = float(data.get('initial_capital', data['cash']))
        portfolio.performance_history = data.get('performance_history', [])
        
        # Reconstruction Positions
        for ticker, pos_data in data.get('positions', {}).items():
            if isinstance(pos_data.get('last_updated'), str):
                pos_data['last_updated'] = datetime.fromisoformat(pos_data['last_updated'])
            portfolio.positions[ticker] = Position(**pos_data)
            
        # Reconstruction Historique Transactions
        for h in data.get('history', []):
            for k in ['timestamp', 'execution_time']:
                if h.get(k) and isinstance(h[k], str):
                    h[k] = datetime.fromisoformat(h[k])
            
            order = Order(**{k: v for k, v in h.items() if k in Order.__dataclass_fields__})
            portfolio.transaction_history.append(order)
            
        return portfolio

    # --- LOGIQUE MÉTIER ---

    def execute_order(self, order: Order) -> bool:
        """Exécute l'ordre et met à jour le cash et les positions."""
        try:
            is_buy = order.order_type.lower() == 'buy'
            price = order.executed_price or order.limit_price or 0
            qty = order.executed_quantity or order.quantity
            total_cost = (qty * price) + order.fees
            
            if is_buy and total_cost > self.cash:
                logger.error(f"Cash insuffisant: {self.cash:.2f} < {total_cost:.2f}")
                return False

            if order.ticker not in self.positions:
                if not is_buy:
                    return False
                self.positions[order.ticker] = Position(order.ticker, 0, 0)
            
            self.positions[order.ticker].update_position(qty, price, order.fees, is_buy)
            
            # Nettoyage si position fermée
            if self.positions[order.ticker].quantity <= 1e-10:
                del self.positions[order.ticker]

            self.cash += (qty * price * (-1 if is_buy else 1)) - order.fees
            
            order.status = 'executed'
            order.execution_time = datetime.now()
            if order not in self.transaction_history:
                self.transaction_history.append(order)
            
            self.last_updated = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Erreur exécution {order.ticker}: {e}")
            return False

    def update_history(self, market_prices: Dict[str, float]):
        """Point d'entrée pour le Dash : enregistre un point sur la courbe d'équité."""
        perf = self.calculate_performance(market_prices)
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': float(perf['total_value']),
            'cash': float(self.cash)
        })
        # Garder les 1000 derniers points pour limiter la taille du JSON
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

    def calculate_performance(self, market_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calcul complet pour le Dashboard."""
        total_market_value = self.cash
        pos_performance = {}

        for t, pos in self.positions.items():
            price = market_prices.get(t, pos.average_price)
            val = pos.current_value(price)
            total_market_value += val
            
            pos_performance[t] = {
                'qty': pos.quantity,
                'avg_price': pos.average_price,
                'current_price': price,
                'value': val,
                'pnl': pos.unrealized_pnl(price),
                'pnl_pct': pos.unrealized_pnl_percent(price) * 100
            }

        total_pnl = total_market_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

        return {
            'total_value': total_market_value,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions': pos_performance
        }

    def get_allocation(self, market_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Retourne l'allocation en % pour les graphiques."""
        prices = market_prices or {}
        perf = self.calculate_performance(prices)
        total = perf['total_value']
        
        if total <= 0:
            return {"Cash": 100.0}

        alloc = {t: (data['value'] / total) for t, data in perf['positions'].items()}
        alloc['Cash'] = (self.cash / total)
        return alloc

    def get_summary(self, market_prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Génère le DataFrame utilisé par dash_table."""
        prices = market_prices or {}
        if not self.positions:
            return pd.DataFrame()

        data = []
        for t, pos in self.positions.items():
            price = prices.get(t, pos.average_price)
            data.append({
                'Ticker': t,
                'Qty': pos.quantity,
                'Prix moyen': pos.average_price,
                'Prix actuel': price,
                'Valeur': pos.current_value(price),
                'P&L': pos.unrealized_pnl(price),
                'P&L %': pos.unrealized_pnl_percent(price)
            })

        df = pd.DataFrame(data)
        total_mkt_val = df['Valeur'].sum() + self.cash
        df['Poids %'] = df['Valeur'] / total_mkt_val if total_mkt_val > 0 else 0
        return df.sort_values('Valeur', ascending=False)

    def display(self, market_prices: Dict[str, float]):
        """Affichage console formaté."""
        df = self.get_summary(market_prices)
        total_val = df['Valeur'].sum() + self.cash if not df.empty else self.cash
        
        logger.info(f"\n{'='*50}\nPORTFOLIO: {self.name.upper()}\n{'='*50}")
        logger.info(f"VALEUR TOTALE: {total_val:,.2f} $ | CASH: {self.cash:,.2f} $")
        
        if RICH_AVAILABLE and not df.empty:
            console = Console()
            table = Table(show_header=True, header_style="bold cyan", box=None)
            for col in df.columns:
                table.add_column(col)
            
            for _, row in df.iterrows():
                pnl_style = "green" if row['P&L'] >= 0 else "red"
                table.add_row(
                    row['Ticker'],
                    f"{row['Qty']:.2f}",
                    f"{row['Prix moyen']:,.2f}",
                    f"{row['Prix actuel']:,.2f}",
                    f"{row['Valeur']:,.2f}",
                    Text(f"{row['P&L']:+,.2f}", style=pnl_style),
                    Text(f"{row['P&L %']:.2%}", style=pnl_style),
                    f"{row['Poids %']:.1%}"
                )
            console.print(table)
        else:
            if not df.empty:
                logger.info("\n" + df.to_string(index=False))
            else:
                logger.info("Aucune position.")