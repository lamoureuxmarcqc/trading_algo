"""
Module de gestion de portefeuille d'actions
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class Order:
    """Représente un ordre à exécuter"""
    ticker: str
    order_type: str  # 'buy' ou 'sell'
    quantity: float
    limit_price: Optional[float] = None
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: str = 'pending'  # pending, executed, cancelled, rejected
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_time: Optional[datetime] = None
    fees: float = 0.0
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def value(self) -> float:
        """Valeur de l'ordre"""
        price = self.limit_price if self.limit_price else 0
        return self.quantity * price
    
    @property
    def executed_value(self) -> float:
        """Valeur exécutée"""
        if self.executed_price and self.executed_quantity:
            return self.executed_quantity * self.executed_price
        return 0.0
    
    def to_dict(self) -> dict:
        """Convertit l'ordre en dictionnaire"""
        data = asdict(self)
        # Conversion des datetime en string pour la sérialisation JSON
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        if self.execution_time:
            data['execution_time'] = self.execution_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        """Crée un ordre à partir d'un dictionnaire"""
        # Conversion des strings en datetime
        if data.get('timestamp'):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('execution_time'):
            data['execution_time'] = datetime.fromisoformat(data['execution_time'])
        return cls(**data)


@dataclass
class Position:
    """Représente une ligne dans le portefeuille"""
    ticker: str
    quantity: float
    average_price: float
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def current_value(self, current_price: float) -> float:
        """Calcule la valeur actuelle de la position"""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calcule le profit/perte non réalisé"""
        return self.quantity * (current_price - self.average_price)
    
    def unrealized_pnl_percent(self, current_price: float) -> float:
        """Calcule le pourcentage de profit/perte non réalisé"""
        if self.average_price == 0:
            return 0.0
        return ((current_price - self.average_price) / self.average_price) * 100
    
    def cost_basis(self) -> float:
        """Coût total d'acquisition"""
        return self.quantity * self.average_price
    
    def update_average_price(self, new_quantity: float, new_price: float, is_buy: bool):
        """Met à jour le prix moyen après un achat/vente"""
        if is_buy:
            # Nouveau prix moyen pondéré
            total_cost = self.cost_basis() + (new_quantity * new_price)
            total_quantity = self.quantity + new_quantity
            self.average_price = total_cost / total_quantity if total_quantity > 0 else 0
            self.quantity = total_quantity
        else:
            # Vente: on réduit la quantité sans changer le prix moyen
            self.quantity -= new_quantity
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> dict:
        """Convertit la position en dictionnaire"""
        data = asdict(self)
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Crée une position à partir d'un dictionnaire"""
        if data.get('last_updated'):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


class Portfolio:
    """Gestionnaire de portefeuille principal"""
    
    def __init__(self, cash: float, positions: Optional[Dict[str, Position]] = None, name: str = "Mon Portefeuille"):
        self.name = name
        self.cash = cash
        self.positions = positions if positions else {}
        self.transaction_history: List[Order] = []
        self.performance_history: List[Dict] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def total_value(self, market_prices: Dict[str, float]) -> float:
        """Calcule la valeur totale du portefeuille"""
        positions_value = sum(
            pos.current_value(market_prices.get(ticker, 0))
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Récupère une position par son ticker"""
        return self.positions.get(ticker)
    
    def add_position(self, ticker: str, quantity: float, price: float) -> Order:
        """Ajoute une nouvelle position (achat)"""
        if ticker in self.positions:
            # Mise à jour d'une position existante
            self.positions[ticker].update_average_price(quantity, price, is_buy=True)
        else:
            # Nouvelle position
            self.positions[ticker] = Position(ticker, quantity, price)
        
        # Mise à jour du cash
        cost = quantity * price
        self.cash -= cost
        
        # Création de l'ordre
        order = Order(
            ticker=ticker,
            order_type='buy',
            quantity=quantity,
            limit_price=price,
            status='executed',
            executed_price=price,
            executed_quantity=quantity,
            execution_time=datetime.now()
        )
        
        self.transaction_history.append(order)
        self.last_updated = datetime.now()
        
        return order
    
    def remove_position(self, ticker: str, quantity: float, price: float) -> Order:
        """Vend une partie ou la totalité d'une position"""
        if ticker not in self.positions:
            raise ValueError(f"Position {ticker} non trouvée")
        
        position = self.positions[ticker]
        
        if quantity > position.quantity:
            raise ValueError(f"Quantité insuffisante: {position.quantity} disponibles")
        
        # Mise à jour de la position
        position.update_average_price(quantity, price, is_buy=False)
        
        # Si quantité devient 0, supprimer la position
        if position.quantity <= 0:
            del self.positions[ticker]
        
        # Mise à jour du cash
        revenue = quantity * price
        self.cash += revenue
        
        # Création de l'ordre
        order = Order(
            ticker=ticker,
            order_type='sell',
            quantity=quantity,
            limit_price=price,
            status='executed',
            executed_price=price,
            executed_quantity=quantity,
            execution_time=datetime.now()
        )
        
        self.transaction_history.append(order)
        self.last_updated = datetime.now()
        
        return order
    
    def get_allocation(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """Calcule l'allocation actuelle du portefeuille"""
        total = self.total_value(market_prices)
        if total == 0:
            return {}
        
        allocation = {'cash': self.cash / total}
        
        for ticker, position in self.positions.items():
            if ticker in market_prices:
                value = position.current_value(market_prices[ticker])
                allocation[ticker] = value / total
        
        return allocation
    
    def generate_rebalance_orders(
        self, 
        target_allocations: Dict[str, float], 
        market_prices: Dict[str, float],
        min_trade_pct: float = 0.01,
        min_trade_value: float = 100.0
    ) -> List[Order]:
        """
        Génère des ordres pour rééquilibrer le portefeuille
        
        Args:
            target_allocations: Allocation cible (ticker -> pourcentage)
            market_prices: Prix actuels du marché
            min_trade_pct: Seuil minimum en % de la valeur totale pour trader
            min_trade_value: Valeur minimum en $ pour trader
        
        Returns:
            Liste d'ordres à exécuter
        """
        orders = []
        total_val = self.total_value(market_prices)
        
        if total_val == 0:
            return orders
        
        # S'assurer que cash est dans les allocations
        if 'cash' not in target_allocations:
            target_allocations['cash'] = 0.0
        
        # Calculer les valeurs cibles
        target_values = {}
        for ticker, target_pct in target_allocations.items():
            target_values[ticker] = total_val * target_pct
        
        current_values = {'cash': self.cash}
        for ticker, position in self.positions.items():
            if ticker in market_prices:
                current_values[ticker] = position.current_value(market_prices[ticker])
            else:
                current_values[ticker] = 0.0
        
        # Générer les ordres
        for ticker, target_value in target_values.items():
            current_value = current_values.get(ticker, 0)
            diff_value = target_value - current_value
            
            # Vérifier le seuil
            if abs(diff_value) / total_val < min_trade_pct or abs(diff_value) < min_trade_value:
                continue
            
            if ticker == 'cash':
                # Pour le cash, pas d'ordre, juste ajuster
                continue
            
            # Calculer la quantité à trader
            price = market_prices.get(ticker)
            if price is None or price <= 0:
                continue
            
            quantity = diff_value / price
            
            if diff_value > 0:
                # Achat
                orders.append(Order(
                    ticker=ticker,
                    order_type='buy',
                    quantity=abs(quantity),
                    limit_price=price
                ))
            else:
                # Vente
                orders.append(Order(
                    ticker=ticker,
                    order_type='sell',
                    quantity=abs(quantity),
                    limit_price=price
                ))
        
        return orders
    
    def calculate_performance(self, market_prices: Dict[str, float], initial_cash: float = None) -> Dict:
        """Calcule les métriques de performance du portefeuille"""
        if initial_cash is None:
            initial_cash = self.cash + sum(
                pos.cost_basis() for pos in self.positions.values()
            )
        
        total_value = self.total_value(market_prices)
        
        # Profit/Perte total
        total_pnl = total_value - initial_cash
        total_pnl_pct = (total_pnl / initial_cash * 100) if initial_cash > 0 else 0
        
        # Profit/Perte par position
        positions_pnl = {}
        for ticker, position in self.positions.items():
            if ticker in market_prices:
                positions_pnl[ticker] = {
                    'unrealized_pnl': position.unrealized_pnl(market_prices[ticker]),
                    'unrealized_pnl_pct': position.unrealized_pnl_percent(market_prices[ticker]),
                    'current_value': position.current_value(market_prices[ticker]),
                    'cost_basis': position.cost_basis()
                }
        
        # Calcul du rendement pondéré
        if self.performance_history:
            # Logique de calcul du rendement (simplifié)
            returns = [perf.get('return', 0) for perf in self.performance_history[-30:]]
            if returns:
                avg_return = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0
                sharpe = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0
            else:
                avg_return = volatility = sharpe = 0
        else:
            avg_return = volatility = sharpe = 0
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'invested': total_value - self.cash,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions': positions_pnl,
            'avg_daily_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'num_positions': len(self.positions)
        }
    
    def save_to_file(self, filepath: str):
        """Sauvegarde le portefeuille dans un fichier JSON"""
        data = {
            'name': self.name,
            'cash': self.cash,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'positions': {ticker: pos.to_dict() for ticker, pos in self.positions.items()},
            'transaction_history': [order.to_dict() for order in self.transaction_history]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Portefeuille sauvegardé: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Portfolio':
        """Charge un portefeuille depuis un fichier JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruire les positions
        positions = {}
        for ticker, pos_data in data.get('positions', {}).items():
            positions[ticker] = Position.from_dict(pos_data)
        
        # Créer le portefeuille
        portfolio = cls(
            cash=data['cash'],
            positions=positions,
            name=data.get('name', 'Mon Portefeuille')
        )
        
        # Reconstruire l'historique
        portfolio.created_at = datetime.fromisoformat(data['created_at'])
        portfolio.last_updated = datetime.fromisoformat(data['last_updated'])
        portfolio.transaction_history = [
            Order.from_dict(order_data) 
            for order_data in data.get('transaction_history', [])
        ]
        
        return portfolio
    
    def get_transaction_history(self, ticker: Optional[str] = None) -> List[Order]:
        """Récupère l'historique des transactions"""
        if ticker:
            return [order for order in self.transaction_history if order.ticker == ticker]
        return self.transaction_history
    
    def get_summary(self, market_prices: Dict[str, float]) -> pd.DataFrame:
        """Retourne un résumé du portefeuille sous forme de DataFrame"""
        rows = []
        
        for ticker, position in self.positions.items():
            price = market_prices.get(ticker, 0)
            current_value = position.current_value(price)
            pnl = position.unrealized_pnl(price)
            pnl_pct = position.unrealized_pnl_percent(price)
            
            rows.append({
                'Ticker': ticker,
                'Quantité': position.quantity,
                'Prix moyen': position.average_price,
                'Prix actuel': price,
                'Valeur': current_value,
                'Coût': position.cost_basis(),
                'P&L': pnl,
                'P&L %': pnl_pct
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Valeur', ascending=False)
        
        return df