# trading_algo/data/data_extraction.py
"""
Module d'extraction de données boursières refactorisé en classes spécialisées.

APIs supportées :
- Yahoo Finance (via yfinance) : prix historiques, overview, bulk prices
- Alpha Vantage (optionnel) : time series, indicateurs
- Financial Modeling Prep (FMP) : fondamentaux
- Twitter/X API v2 : sentiment social
- NY Times API : actualités financières
- FRED (Federal Reserve) : indicateurs macroéconomiques US
- Statistique Canada (WDS) : données macro Canada

Améliorations :
- Découpage par responsabilité
- Cache LRU pour les appels fréquents
- Gestion d'erreur uniforme via décorateur
- Logging centralisé
- Support de fallback robuste pour yfinance
"""

import os
import sys
import json
import time
import logging
import functools
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Configuration des chemins (pour les imports relatifs)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

warnings.filterwarnings('ignore')

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# DÉCORATEUR DE GESTION D'ERREUR
# ============================================================
def log_exceptions(default_return=None, log_level=logging.ERROR):
    """Décorateur pour capturer et logger les exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"Erreur dans {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator

# ============================================================
# CONFIGURATION API
# ============================================================
API_CONFIG = {
    'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'FINANCIAL_MODELING_PREP': os.getenv('FMP_API_KEY'),
    'POLYGON': os.getenv('POLYGON_API_KEY'),
    'TWITTER_X_BEARER': os.getenv('TWITTER_X_BEARER'),
    'NY_TIMES': os.getenv('NY_TIMES_API_KEY'),
    'NY_TIMES_API_KEY_SECRET': os.getenv('NY_TIMES_API_KEY_SECRET'),
    'FRED_API_KEY': os.getenv('FRED_API_KEY')
}

# Vérification basique des clés
for key_name, key_value in API_CONFIG.items():
    if not key_value:
        logger.warning(f"Clé API manquante: {key_name}")
    elif 'demo' in str(key_value).lower() or len(str(key_value)) < 10:
        logger.warning(f"Clé API suspecte ou démo: {key_name}")
        API_CONFIG[key_name] = 'demo'

# ============================================================
# CACHES POUR LA RÉSOLUTION DE SYMBOLES (YFINANCE)
# ============================================================
_SYMBOL_RESOLUTION_CACHE_FILE = os.path.join(os.getcwd(), "cache", "symbol_resolution.json")
_SYMBOL_RESOLUTION_CACHE: Dict[str, str] = {}
_SYMBOL_NEGATIVE_CACHE_FILE = os.path.join(os.getcwd(), "cache", "symbol_resolution_negative.json")
_SYMBOL_NEGATIVE_CACHE: Dict[str, float] = {}
_NEGATIVE_CACHE_TTL = 24 * 3600  # secondes

def _load_symbol_resolution_cache():
    global _SYMBOL_RESOLUTION_CACHE
    try:
        os.makedirs(os.path.dirname(_SYMBOL_RESOLUTION_CACHE_FILE), exist_ok=True)
        if os.path.exists(_SYMBOL_RESOLUTION_CACHE_FILE):
            with open(_SYMBOL_RESOLUTION_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _SYMBOL_RESOLUTION_CACHE = data
    except Exception as e:
        logger.debug(f"Impossible de charger le cache de résolution: {e}")

def _save_symbol_resolution_cache():
    try:
        os.makedirs(os.path.dirname(_SYMBOL_RESOLUTION_CACHE_FILE), exist_ok=True)
        with open(_SYMBOL_RESOLUTION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_SYMBOL_RESOLUTION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Impossible de sauvegarder le cache de résolution: {e}")

def _load_negative_cache():
    global _SYMBOL_NEGATIVE_CACHE
    try:
        os.makedirs(os.path.dirname(_SYMBOL_NEGATIVE_CACHE_FILE), exist_ok=True)
        if os.path.exists(_SYMBOL_NEGATIVE_CACHE_FILE):
            with open(_SYMBOL_NEGATIVE_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _SYMBOL_NEGATIVE_CACHE = data
    except Exception as e:
        logger.debug(f"Impossible de charger le cache négatif: {e}")

def _save_negative_cache():
    try:
        os.makedirs(os.path.dirname(_SYMBOL_NEGATIVE_CACHE_FILE), exist_ok=True)
        with open(_SYMBOL_NEGATIVE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_SYMBOL_NEGATIVE_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Impossible de sauvegarder le cache négatif: {e}")

def _is_recently_negative(symbol: str) -> bool:
    ts = _SYMBOL_NEGATIVE_CACHE.get(symbol)
    if not ts:
        return False
    return (time.time() - float(ts)) < _NEGATIVE_CACHE_TTL

def _mark_negative(symbol: str) -> None:
    _SYMBOL_NEGATIVE_CACHE[symbol] = time.time()
    _save_negative_cache()

# Chargement initial
_load_symbol_resolution_cache()
_load_negative_cache()

# ============================================================
# HELPER ROBUSTE POUR YFINANCE
# ============================================================
def _yf_history_with_fallback(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Récupère l'historique Yahoo Finance avec résolution automatique des symboles,
    retries sur erreurs transitoires, et cache négatif.
    """
    from requests.exceptions import ReadTimeout, ConnectionError as ReqConnError

    tried = []
    permanent_failure_msgs = ("Not Found", "Quote not found", "No data found, symbol may be delisted")
    transient_errors = (ReadTimeout, ReqConnError)

    if _is_recently_negative(symbol):
        logger.debug(f"Symbole {symbol} en cache négatif - skip")
        return pd.DataFrame()

    # Essayer le symbole résolu en cache
    cached = _SYMBOL_RESOLUTION_CACHE.get(symbol)
    if cached:
        try:
            logger.debug(f"Essai symbole mis en cache {symbol} -> {cached}")
            df = yf.Ticker(cached).history(period=period, interval=interval, timeout=30)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"Utilisation du cache pour {symbol} -> {cached}")
                return df
            # Invalider le cache si vide
            _SYMBOL_RESOLUTION_CACHE.pop(symbol, None)
            _save_symbol_resolution_cache()
        except Exception as e:
            logger.debug(f"Échec du cache pour {cached}: {e}")
            _SYMBOL_RESOLUTION_CACHE.pop(symbol, None)
            _save_symbol_resolution_cache()

    # Générer les candidats
    candidates = [symbol]
    if '.' in symbol:
        candidates.append(symbol.replace('.', '-'))
    if '-' in symbol:
        candidates.append(symbol.replace('-', '.'))
    if symbol.isalpha() and len(symbol) <= 6 and not symbol.upper().endswith(('.TO', '-TO')):
        candidates.append(symbol + ".TO")
        candidates.append(symbol + "-TO")
    candidates.append(symbol.upper())
    # Déduplication
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    all_permanent = True
    for cand in unique_candidates:
        tried.append(cand)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                timeout = 10 + (attempt - 1) * 10
                df = yf.Ticker(cand).history(period=period, interval=interval, timeout=timeout)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if cand != symbol:
                        _SYMBOL_RESOLUTION_CACHE[symbol] = cand
                        _save_symbol_resolution_cache()
                        logger.info(f"Résolution {symbol} -> {cand} (cached)")
                    return df
                # DataFrame vide -> considéré comme permanent (pas de données)
                break
            except Exception as e:
                msg = str(e)
                if any(m in msg for m in permanent_failure_msgs) or ("404" in msg and "Not Found" in msg):
                    logger.debug(f"Candidat {cand} en échec permanent: {msg}")
                    break
                elif isinstance(e, transient_errors) or "timed out" in msg.lower():
                    backoff = 1.0 * attempt
                    logger.debug(f"Erreur transitoire {cand} (tentative {attempt}/{max_attempts}): {msg} - retry dans {backoff}s")
                    time.sleep(backoff)
                    continue
                else:
                    # Autre erreur, on considère comme transitoire pour ne pas bloquer
                    logger.debug(f"Erreur inconnue {cand}: {msg}")
                    time.sleep(0.5 * attempt)
                    continue
        else:
            # Si on sort de la boucle sans succès, on ne marque pas forcément permanent
            all_permanent = False

    if all_permanent:
        _mark_negative(symbol)
        logger.warning(f"Aucune donnée pour {symbol} (essais: {', '.join(tried)}) - négatif cache")
    else:
        logger.warning(f"Aucune donnée pour {symbol} (essais: {', '.join(tried)}) - échec transitoire")

    return pd.DataFrame()

# ============================================================
# 1. STOCK DATA EXTRACTOR (prix + indicateurs techniques)
# ============================================================
class StockDataExtractor:
    """
    Responsable des données de prix historiques, des indicateurs techniques
    et de la récupération groupée des prix récents (bulk).
    """
    def __init__(self, symbol: str = None, cache_dir: str = 'cache'):
        self.symbol = symbol
        self.cache_dir = cache_dir
        self.data = None
        self.features = None
        self.targets = None
        self._setup_cache()

    def _setup_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache = {}

    @log_exceptions(default_return=pd.DataFrame())
    def get_historical_data(self, symbol: str = None, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
        """Récupère l'historique OHLCV via Yahoo Finance."""
        symbol = symbol or self.symbol
        if not symbol:
            logger.error("Aucun symbole fourni")
            return pd.DataFrame()

        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            logger.info(f"Utilisation du cache pour {symbol}")
            return self.cache[cache_key].copy()

        df = _yf_history_with_fallback(symbol, period, interval)
        if df.empty:
            logger.warning(f"Aucune donnée pour {symbol}")
            return pd.DataFrame()

        # Colonnes standard
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                logger.error(f"Colonne manquante: {col}")
                return pd.DataFrame()

        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        self.cache[cache_key] = df.copy()
        self.data = df.copy()
        logger.info(f"Données historiques pour {symbol}: {len(df)} lignes")
        return df

    @log_exceptions(default_return={})
    def get_bulk_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Récupère les derniers prix pour une liste de tickers en une seule requête groupée.
        Optimisé pour le dashboard.
        """
        if not tickers:
            return {}

        # Préparer la correspondance ticker -> symbole résolu
        request_map = {}
        for t in tickers:
            resolved = _SYMBOL_RESOLUTION_CACHE.get(t)
            if resolved:
                request_map[t] = resolved
            elif "." in t and not t.upper().endswith(".TO"):
                request_map[t] = t.replace(".", "-")
            else:
                request_map[t] = t

        # Téléchargement groupé (yfinance supporte les espaces)
        try:
            data = yf.download(
                tickers=" ".join(request_map.values()),
                period="1d",
                interval="1m",
                group_by='ticker',
                threads=True,
                progress=False
            )
        except Exception as e:
            logger.warning(f"Erreur bulk download: {e}")
            data = None

        prices = {}
        for ticker in tickers:
            req_ticker = request_map[ticker]
            try:
                if data is not None:
                    if len(tickers) > 1:
                        last_price = data[req_ticker]['Close'].dropna().iloc[-1]
                    else:
                        last_price = data['Close'].dropna().iloc[-1]
                    prices[ticker] = float(last_price)
                else:
                    prices[ticker] = 0.0
            except Exception:
                # Fallback individuel
                try:
                    df = self.get_historical_data(ticker, period="5d", interval="1d")
                    if not df.empty:
                        prices[ticker] = float(df["Close"].dropna().iloc[-1])
                except Exception:
                    prices[ticker] = 0.0
        return prices

    @log_exceptions(default_return=pd.DataFrame())
    def calculate_technical_indicators(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calcule une large gamme d'indicateurs techniques.
        N'ajoute PAS les features externes (c'est la responsabilité d'autres classes).
        """
        if data is None:
            data = self.data
        if data is None or data.empty:
            logger.warning("Aucune donnée pour calculer les indicateurs")
            return pd.DataFrame()

        df = data.copy()
        close = df['Close'].astype('float32')
        high = df['High'].astype('float32')
        low = df['Low'].astype('float32')
        volume = df['Volume'].astype('float32')

        indicators = {}

        # Moyennes mobiles
        for p in [5, 10, 20, 50, 200]:
            indicators[f'SMA_{p}'] = close.rolling(window=p, min_periods=1).mean()
            indicators[f'EMA_{p}'] = close.ewm(span=p, adjust=False, min_periods=1).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        indicators['MACD'] = ema12 - ema26
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        indicators['BB_Middle'] = bb_mid
        indicators['BB_Upper'] = bb_mid + 2 * bb_std
        indicators['BB_Lower'] = bb_mid - 2 * bb_std
        indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / bb_mid.replace(0, np.finfo(float).eps)

        # ATR
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = np.maximum.reduce([tr1, tr2, tr3])
        indicators['ATR'] = pd.Series(true_range, index=df.index).rolling(14).mean()

        # Stochastic
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        denom = (high14 - low14).replace(0, np.finfo(float).eps)
        indicators['Stoch_%K'] = 100 * ((close - low14) / denom)
        indicators['Stoch_%D'] = indicators['Stoch_%K'].rolling(3).mean()

        # OBV
        price_change = np.sign(close.diff())
        indicators['OBV'] = (price_change * volume).fillna(0).cumsum().astype('float32')

        # VWAP
        typical = (high + low + close) / 3
        indicators['VWAP'] = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Returns & Volatilité
        ret = close.pct_change()
        indicators['Returns'] = ret
        for w in [5, 10, 20]:
            indicators[f'Volatility_{w}d'] = ret.rolling(w).std() * np.sqrt(252)

        # Momentum
        for p in [5, 10, 20]:
            indicators[f'Momentum_{p}'] = close.pct_change(p)

        # Support / Resistance
        lookback = 20
        indicators['Support'] = low.rolling(lookback).min()
        indicators['Resistance'] = high.rolling(lookback).max()
        diff_range = indicators['Resistance'] - indicators['Support']
        diff_range = diff_range.replace(0, np.finfo(float).eps)
        indicators['Price_Position'] = (close - indicators['Support']) / diff_range

        # ADX
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr14 = indicators['ATR'] * 14
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / tr14.replace(0, np.finfo(float).eps)
        minus_di = 100 * pd.Series(np.abs(minus_dm), index=df.index).ewm(alpha=1/14).mean() / tr14.replace(0, np.finfo(float).eps)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
        indicators['ADX'] = dx.ewm(alpha=1/14).mean()

        # Chaikin Oscillator
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.finfo(float).eps)
        adl = mfm * volume
        indicators['Chaikin_Oscillator'] = adl.ewm(span=3).mean() - adl.ewm(span=10).mean()

        # MFI
        typical_price = typical
        money_flow = typical_price * volume
        pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        neg_flow = np.where(typical_price <= typical_price.shift(1), money_flow, 0)
        pos_sum = pd.Series(pos_flow, index=df.index).rolling(14).sum()
        neg_sum = pd.Series(neg_flow, index=df.index).rolling(14).sum()
        mfr = pos_sum / neg_sum.replace(0, np.finfo(float).eps)
        indicators['MFI'] = 100 - (100 / (1 + mfr))

        # CCI
        sma_typical = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        indicators['CCI'] = (typical_price - sma_typical) / (0.015 * mad.replace(0, np.finfo(float).eps))

        # Williams %R
        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        indicators['Williams_%R'] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.finfo(float).eps)

        # Price Channels
        indicators['Price_Channel_High'] = high.rolling(20).max()
        indicators['Price_Channel_Low'] = low.rolling(20).min()

        # Rate of Change
        for p in [10, 20, 50]:
            indicators[f'ROC_{p}'] = close.pct_change(p) * 100

        # Ajout au DataFrame
        for name, values in indicators.items():
            df[name] = values

        # Nettoyage : suppression colonnes trop NaN
        nan_threshold = 0.5
        cols_to_drop = [c for c in df.columns if df[c].isna().sum() / len(df) > nan_threshold]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Forward fill limité puis mean pour les résiduels
        tech_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume']]
        df[tech_cols] = df[tech_cols].ffill(limit=5).bfill(limit=5)
        for col in tech_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        # Troncature pour supprimer les premières lignes instables
        keep_from = max(200, int(len(df) * 0.05))
        df = df.iloc[keep_from:]

        # Conversion float64 -> float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        self.features = df
        logger.info(f"Indicateurs techniques calculés: {df.shape[1]} colonnes, {len(df)} lignes")
        return df
    def create_target_columns(self, 
                              data: pd.DataFrame = None, 
                              forecast_days: List[int] = None) -> pd.DataFrame:
        """
        Crée les colonnes cibles pour l'entraînement du modèle.
        Les prix sont transformés en log (log prix) pour stabiliser la variance.
        """
        if data is None:
            data = self.data

        if data is None or data.empty:
            logger.warning("Aucune donnée pour créer les cibles")
            return pd.DataFrame()

        if forecast_days is None:
            forecast_days = [1, 5, 10, 20, 30, 90]

        try:
            df = data.copy()
            close = df['Close']

            targets = pd.DataFrame(index=df.index)

            for days in forecast_days:
                # Prix futurs (log)
                targets[f'Target_Close_{days}d'] = np.log(close.shift(-days))
                targets[f'Target_Return_{days}d'] = close.shift(-days) / close - 1
                targets[f'Target_Direction_{days}d'] = (close.shift(-days) > close).astype(int)

            # Ajout de la volatilité future (fenêtre glissante)
            returns = close.pct_change()
            for days in [5, 10, 20]:
                if days <= max(forecast_days):
                    targets[f'Target_Volatility_{days}d'] = returns.rolling(days).std().shift(-days)

            self.targets = targets
            logger.info(f"Cibles créées: {targets.shape[1]} colonnes")
            return targets

        except Exception as e:
            logger.error(f"Erreur create_target_columns: {e}", exc_info=True)
            return pd.DataFrame()
# ============================================================
# 2. FUNDAMENTAL EXTRACTOR
# ============================================================
class FundamentalExtractor:
    """Données fondamentales (FMP avec fallback Yahoo Finance) et aperçu rapide."""
    def __init__(self):
        self.fmp_api_key = API_CONFIG.get('FINANCIAL_MODELING_PREP')
        self.fmp_demo = (self.fmp_api_key == 'demo')

    @log_exceptions(default_return={})
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Tente FMP d'abord, puis Yahoo Finance."""
        if not self.fmp_demo:
            try:
                profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
                ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
                params = {'apikey': self.fmp_api_key}

                profile_resp = requests.get(profile_url, params=params, timeout=10)
                if profile_resp.status_code == 403:
                    logger.warning("Clé FMP invalide ou abonnement insuffisant, passage en mode fallback")
                    self.fmp_demo = True
                elif profile_resp.status_code == 200:
                    profile = profile_resp.json()
                    ratios_resp = requests.get(ratios_url, params=params, timeout=10)
                    ratios = ratios_resp.json() if ratios_resp.status_code == 200 else []
                    return {
                        'profile': profile[0] if profile else {},
                        'ratios': ratios[0] if ratios else {},
                        'source': 'Financial Modeling Prep'
                    }
            except Exception as e:
                logger.warning(f"Erreur FMP pour {symbol}: {e}")

        # Fallback Yahoo Finance
        return self._get_fundamental_from_yahoo(symbol)

    @staticmethod
    @log_exceptions(default_return={})
    def _get_fundamental_from_yahoo(symbol: str) -> Dict[str, Any]:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'profile': {
                'companyName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'marketCap': info.get('marketCap', 0),
                'beta': info.get('beta', 0),
            },
            'ratios': {
                'peRatio': info.get('trailingPE', 0),
                'forwardPE': info.get('forwardPE', 0),
                'pegRatio': info.get('pegRatio', 0),
                'priceToSalesRatio': info.get('priceToSalesTrailing12Months', 0),
                'debtToEquity': info.get('debtToEquity', 0),
                'returnOnEquity': info.get('returnOnEquity', 0),
            },
            'source': 'Yahoo Finance'
        }

    @staticmethod
    @log_exceptions(default_return={})
    def get_stock_overview(symbol: str) -> Dict[str, Any]:
        """Aperçu rapide d'une action (nom, prix, market cap, ratios)."""
        stock = yf.Ticker(symbol)
        info = stock.info

        def fmt_market_cap(val):
            if val is None:
                return 'N/A'
            try:
                v = float(val)
                if v >= 1e12:
                    return f"${v/1e12:.2f}T"
                elif v >= 1e9:
                    return f"${v/1e9:.2f}B"
                elif v >= 1e6:
                    return f"${v/1e6:.2f}M"
                return f"${v:,.0f}"
            except:
                return str(val)

        return {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', 'N/A')),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'previous_close': info.get('previousClose', 'N/A'),
            'market_cap': fmt_market_cap(info.get('marketCap')),
            'pe_ratio': f"{info.get('trailingPE', 0):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A',
            'forward_pe': f"{info.get('forwardPE', 0):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A',
            'dividend_yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else '0.00%',
            'beta': info.get('beta', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A'),
            'country': info.get('country', 'N/A')
        }

# ============================================================
# 3. SENTIMENT EXTRACTOR (Twitter/X + NY Times)
# ============================================================
class SentimentExtractor:
    """Sentiment social via Twitter/X et actualités NY Times."""
    def __init__(self):
        self.twitter_bearer = API_CONFIG.get('TWITTER_X_BEARER')
        self.nyt_api_key = API_CONFIG.get('NY_TIMES')
        self.nyt_demo = (self.nyt_api_key == 'demo')

    @log_exceptions(default_return={})
    def get_sentiment_from_x(self, symbol: str) -> Dict[str, Any]:
        """Récupère les tweets récents et analyse le sentiment."""
        if not self.twitter_bearer or self.twitter_bearer == 'demo':
            return self._get_simulated_sentiment(symbol)

        headers = {"Authorization": f"Bearer {self.twitter_bearer}", "User-Agent": "v2RecentSearchPython"}
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': f'${symbol} (stock OR finance OR investing) lang:en -is:retweet -is:reply',
            'max_results': 50,
            'tweet.fields': 'text,created_at,public_metrics',
            'expansions': 'author_id'
        }
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            logger.error(f"Erreur API X: {response.status_code}")
            return self._get_simulated_sentiment(symbol)

        data = response.json()
        tweets = data.get('data', [])
        if not tweets:
            return self._get_simulated_sentiment(symbol)

        positive_words = {'bullish','buy','strong','growth','profit','gain','up','positive','good','great','excellent','amazing','awesome','win','increase','rise','soar','surge','rally','outperform'}
        negative_words = {'bearish','sell','weak','drop','loss','down','negative','crash','bad','poor','terrible','awful','horrible','lose','decrease','fall','plunge','decline','underperform'}

        scores = []
        total_likes = 0
        total_retweets = 0
        sample = []
        for tweet in tweets[:10]:
            text = tweet.get('text', '').lower()
            metrics = tweet.get('public_metrics', {})
            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)
            total = pos + neg
            score = (pos - neg) / total if total > 0 else 0
            scores.append(score)
            total_likes += metrics.get('like_count', 0)
            total_retweets += metrics.get('retweet_count', 0)
            if len(sample) < 3 and len(text) > 20:
                sample.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'sentiment': 'POSITIVE' if score > 0.1 else 'NEGATIVE' if score < -0.1 else 'NEUTRAL'
                })

        avg_sentiment = np.mean(scores) if scores else 0
        if avg_sentiment > 0.2:
            label = 'POSITIVE'
        elif avg_sentiment < -0.2:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        return {
            'tweet_count': len(tweets),
            'avg_sentiment': float(avg_sentiment),
            'sentiment_label': label,
            'total_likes': total_likes,
            'total_retweets': total_retweets,
            'sample_tweets': sample,
            'source': 'Twitter/X API v2'
        }

    def _get_simulated_sentiment(self, symbol: str) -> Dict[str, Any]:
        np.random.seed(hash(symbol) % 1000)
        avg_sent = np.random.uniform(-0.3, 0.6)
        if avg_sent > 0.2:
            label = 'POSITIVE'
        elif avg_sent < -0.2:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        return {
            'tweet_count': np.random.randint(50, 200),
            'avg_sentiment': float(avg_sent),
            'sentiment_label': label,
            'total_likes': np.random.randint(500, 5000),
            'total_retweets': np.random.randint(50, 500),
            'sample_tweets': [
                {'text': f"${symbol} looking strong today #stocks", 'likes': 45, 'retweets': 12, 'sentiment': 'POSITIVE'},
                {'text': f"Watching ${symbol} for potential breakout", 'likes': 23, 'retweets': 5, 'sentiment': 'NEUTRAL'}
            ],
            'source': 'Twitter/X (simulé)'
        }

    @log_exceptions(default_return={})
    def get_news_from_nyt(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Récupère les articles NY Times récents."""
        if self.nyt_demo:
            return self._get_simulated_news(symbol)

        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        params = {
            'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
            'begin_date': from_date,
            'sort': 'newest',
            'api-key': self.nyt_api_key,
            'fl': 'headline,pub_date,web_url,snippet,source'
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                logger.error(f"Erreur NYT: {response.status_code}")
                return self._get_simulated_news(symbol)

            data = response.json()
            articles = data.get('response', {}).get('docs')
            # ✅ Correction : si articles est None, on le transforme en liste vide
            if articles is None:
                articles = []
            news_data = {
                'total_articles': len(articles),
                'articles': [],
                'source': 'New York Times API',
                'last_updated': datetime.now().isoformat()
            }
            positive_keywords = {'gain','rise','surge','profit','growth','beat','increase','soar','rally','outperform','positive','strong','bullish','record','high','success','win'}
            negative_keywords = {'drop','fall','loss','decline','miss','cut','warn','plunge','slide','underperform','negative','weak','bearish','low','failure','lose','risk','crisis'}

            for art in articles[:5]:
                headline = art.get('headline', {}).get('main', '')
                snippet = art.get('snippet', '')
                pub_date = art.get('pub_date', '')
                web_url = art.get('web_url', '')
                source = art.get('source', 'NYT')
                text = (headline + ' ' + snippet).lower()
                pos = sum(1 for w in positive_keywords if w in text)
                neg = sum(1 for w in negative_keywords if w in text)
                if pos > neg:
                    sentiment = 'POSITIVE'
                elif neg > pos:
                    sentiment = 'NEGATIVE'
                else:
                    sentiment = 'NEUTRAL'
                news_data['articles'].append({
                    'headline': headline,
                    'snippet': snippet[:200] + '...' if len(snippet) > 200 else snippet,
                    'date': pub_date[:10] if pub_date else '',
                    'url': web_url,
                    'source': source,
                    'sentiment': sentiment
                })
            return news_data
        except Exception as e:
            logger.error(f"Erreur dans get_news_from_nyt: {e}", exc_info=True)
            return self._get_simulated_news(symbol)
    def _get_simulated_news(self, symbol: str) -> Dict[str, Any]:
        np.random.seed(hash(symbol) % 1000)
        articles = [
            {'headline': f"{symbol} Reports Strong Earnings", 'snippet': "The company beat estimates.", 'sentiment': 'POSITIVE'},
            {'headline': f"Analysts Upgrade {symbol}", 'snippet': "New price target raised.", 'sentiment': 'POSITIVE'},
            {'headline': f"{symbol} Faces Regulatory Scrutiny", 'snippet': "Investigation launched.", 'sentiment': 'NEGATIVE'}
        ]
        selected = np.random.choice(len(articles), size=np.random.randint(2, 4), replace=False)
        return {
            'total_articles': len(selected),
            'articles': [{
                'headline': articles[i]['headline'],
                'snippet': articles[i]['snippet'],
                'date': (datetime.now() - timedelta(days=np.random.randint(0,7))).strftime('%Y-%m-%d'),
                'url': f"https://nytimes.com/simulated/{symbol}",
                'source': 'NYT (simulé)',
                'sentiment': articles[i]['sentiment']
            } for i in selected],
            'source': 'New York Times (simulé)',
            'last_updated': datetime.now().isoformat()
        }

# ============================================================
# 4. MACRO DATA EXTRACTOR (StatCan, FRED, Commodities, FX)
# ============================================================
class MacroDataExtractor:
    """Indicateurs macroéconomiques US et Canada, matières premières, devises."""
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
        self.fred_api_key = API_CONFIG.get('FRED_API_KEY')
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"

    @staticmethod
    def _statcan_sdw(method: str, params: Dict = None) -> Optional[Any]:
        """Appel à l'API Statistique Canada Web Data Service."""
        supported = ['getDataFromCubePidCoordAndLatestNPeriods', 'getDataFromVectorsAndLatestNPeriods',
                     'getBulkVectorDataByRange', 'getCubeMetadata', 'getSeriesInfoFromVector']
        if method not in supported:
            logger.error(f"StatCan: méthode {method} non supportée")
            return None
        url = f"https://www150.statcan.gc.ca/t1/wds/rest/{method}"
        headers = {'Content-Type': 'application/json'}
        payload = []
        if method == 'getDataFromVectorsAndLatestNPeriods':
            vector_ids = params.get('vectorIds', params.get('vectorId', []))
            latest_n = params.get('latestN', 1)
            if not isinstance(vector_ids, (list, tuple)):
                vector_ids = [v.strip() for v in str(vector_ids).split(',') if v.strip()]
            for v in vector_ids:
                v_digits = ''.join(filter(str.isdigit, str(v)))
                if v_digits:
                    payload.append({"vectorId": int(v_digits), "latestN": int(latest_n)})
        else:
            payload = params or {}
        if not payload and method == 'getDataFromVectorsAndLatestNPeriods':
            return None
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Erreur StatCan: {e}")
            return None

    @log_exceptions(default_return={})
    def get_economic_indicators(self, country: str = "US") -> Dict[str, Any]:
        """Indicateurs clés (PIB, taux, etc.) pour US ou Canada."""
        now = datetime.now()
        cache_key = f"macro_{country}_{now.strftime('%Y-%m-%d')}"
        if cache_key in self.cache:
            ts, data = self.cache[cache_key]
            if now - ts < self.cache_duration:
                return data

        indicators = {}
        if country == "CA":
            try:
                res = self._statcan_sdw('getDataFromVectorsAndLatestNPeriods',
                                        {'vectorIds': ['v62305755'], 'latestN': 1})
                if res and isinstance(res, list) and len(res) > 0 and res[0].get('status') == 'SUCCESS':
                    points = res[0].get('object', {}).get('vectorDataPoint', [])
                    if points:
                        indicators['GDP Canada'] = {
                            'value': points[0].get('value'),
                            'date': points[0].get('refPer'),
                            'unit': 'Index (CAD)',
                            'source': 'Statistique Canada'
                        }
            except Exception as e:
                logger.warning(f"Erreur PIB Canada: {e}")

        # Indicateurs globaux via Yahoo
        global_map = {
            '10Y Treasury': '^TNX',
            'VIX Volatility': '^VIX',
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F',
            'US Dollar Index': 'DX-Y.NYB'
        }
        for label, sym in global_map.items():
            df = _yf_history_with_fallback(sym, period='5d', interval='1d')
            if not df.empty:
                current = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                indicators[label] = {
                    'value': round(current, 2),
                    'change': round(change, 2),
                    'unit': '%' if 'Treasury' in label else 'USD',
                    'source': 'Yahoo Finance'
                }

        self.cache[cache_key] = (now, indicators)
        return indicators

    @log_exceptions(default_return={})
    def get_commodity_prices(self) -> Dict[str, Any]:
        """Prix des matières premières via Yahoo."""
        commodities = {'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
                       'Crude Oil': 'CL=F', 'Natural Gas': 'NG=F', 'Wheat': 'ZW=F', 'Corn': 'ZC=F'}
        prices = {}
        for name, sym in commodities.items():
            df = _yf_history_with_fallback(sym, period='5d', interval='1d')
            if not df.empty:
                current = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                prices[name] = {'price': current, 'change': change, 'currency': 'USD'}
        return prices

    @log_exceptions(default_return={})
    def get_currency_rates(self) -> Dict[str, Any]:
        """Taux de change via Yahoo (fallback exchangerate.host si nécessaire)."""
        pairs = {'USD/CAD': 'CAD=X', 'USD/EUR': 'EUR=X', 'USD/JPY': 'JPY=X',
                 'USD/GBP': 'GBP=X', 'USD/CHF': 'CHF=X', 'USD/AUD': 'AUD=X'}
        rates = {}
        fallback_needed = []
        for pair, sym in pairs.items():
            df = _yf_history_with_fallback(sym, period='5d', interval='1d')
            if not df.empty:
                current = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                rates[pair] = {'rate': current, 'change': change, 'source': 'yfinance'}
            else:
                fallback_needed.append(pair)

        if fallback_needed:
            try:
                symbols = sorted({p.split('/')[-1] for p in fallback_needed})
                resp = requests.get("https://api.exchangerate.host/latest", params={'base': 'USD', 'symbols': ','.join(symbols)}, timeout=10)
                if resp.ok:
                    data = resp.json()
                    for p in fallback_needed:
                        code = p.split('/')[-1]
                        rate = data.get('rates', {}).get(code)
                        if rate:
                            rates[p] = {'rate': rate, 'change': 0.0, 'source': 'exchangerate.host'}
            except Exception as e:
                logger.warning(f"Fallback exchangerate.host échoué: {e}")
        return rates

    @log_exceptions(default_return={})
    def get_all_macro_data(self) -> Dict[str, Any]:
        """Regroupe toutes les données macro en un seul dictionnaire."""
        return {
            'timestamp': datetime.now().isoformat(),
            'us_indicators': self.get_economic_indicators('US'),
            'canada_indicators': self.get_economic_indicators('CA'),
            'commodities': self.get_commodity_prices(),
            'currencies': self.get_currency_rates()
        }

    @log_exceptions(default_return={})
    def get_fred_series(self, series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Télécharge une série FRED (ex: DGS2, DGS10, etc.)."""
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date
        }
        if not self.fred_api_key or self.fred_api_key == 'demo':
            logger.warning("Clé FRED manquante ou démo")
            return pd.DataFrame()
        resp = requests.get(self.fred_base_url, params=params, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"FRED erreur {resp.status_code} pour {series_id}")
            return pd.DataFrame()
        data = resp.json().get("observations", [])
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]]

# ============================================================
# 5. SECTOR PERFORMANCE EXTRACTOR
# ============================================================
class SectorPerformanceExtractor:
    """Compare la performance d'un secteur par rapport à un indice."""
    @log_exceptions(default_return=pd.DataFrame())
    def get_sector_vs_market(self, sector_symbol: str, market_symbol: str = "^GSPC", period: str = "6mo") -> pd.DataFrame:
        sec = yf.Ticker(sector_symbol).history(period=period)
        mkt = yf.Ticker(market_symbol).history(period=period)
        if sec.empty or mkt.empty:
            logger.warning(f"Données manquantes pour {sector_symbol} ou {market_symbol}")
            return pd.DataFrame()
        df = pd.DataFrame({
            "sector": sec["Close"].pct_change().cumsum(),
            "market": mkt["Close"].pct_change().cumsum()
        })
        df["relative"] = df["sector"] - df["market"]
        return df

# ============================================================
# 6. MARKET HEALTH ANALYZER (score, scénarios Fed)
# ============================================================
class MarketHealthAnalyzer:
    """Calcule un score de santé du marché et analyse l'impact des hausses de taux."""
    def __init__(self, macro_extractor: Optional[MacroDataExtractor] = None):
        self.macro = macro_extractor or MacroDataExtractor()
        self.sector = SectorPerformanceExtractor()

    @log_exceptions(default_return={'score': 50, 'label': 'Neutral', 'description': 'Neutre'})
    def compute_market_health(self, market_data: Dict = None) -> Dict[str, Any]:
        """Score composite (0-100) basé sur VIX, Put/Call, Breadth, Indices, Sectors."""
        if market_data is None:
            # On récupère les indicateurs via StockDataExtractor (pour compatibilité)
            tmp = StockDataExtractor().get_market_indicators()
            market_data = tmp if tmp else {}
        components = []

        # VIX (20%)
        vix = market_data.get('VIX', {}).get('price')
        if vix is not None:
            vix_score = max(0, min(100, (30 - vix) / 15 * 100))
            components.append(vix_score * 0.20)

        # Put/Call (15%)
        put_call = market_data.get('Equity Put/Call Ratio', {}).get('price')
        if put_call is not None:
            pc_score = max(0, min(100, (1.2 - put_call) / 1.2 * 100))
            components.append(pc_score * 0.15)

        # Breadth (15%)
        adv_decl = market_data.get('NYSE Advance/Decline Line', {})
        if adv_decl:
            change_adv = adv_decl.get('change', 0)
            breadth_score = max(0, min(100, 50 + change_adv))
            components.append(breadth_score * 0.15)

        # Indices (30%)
        indices = ['S&P 500', 'NASDAQ', 'Dow Jones']
        perf_scores = []
        for idx in indices:
            change = market_data.get(idx, {}).get('change')
            if change is not None:
                perf_scores.append(max(0, min(100, 50 + change * 25)))
        if perf_scores:
            components.append(np.mean(perf_scores) * 0.30)

        # Sectors (20%)
        sectors = market_data.get('sectors', {})
        if sectors:
            positive = sum(1 for s in sectors.values() if s.get('change', 0) > 0)
            sector_score = (positive / len(sectors)) * 100
            components.append(sector_score * 0.20)

        if not components:
            score = 50.0
        else:
            score = sum(components)

        score = max(0, min(100, score))
        if score > 70:
            label, desc = 'Bullish', 'Haussier'
        elif score < 30:
            label, desc = 'Bearish', 'Baissier'
        else:
            label, desc = 'Neutral', 'Neutre'
        return {'score': score, 'label': label, 'description': desc, 'score_out_of_100': score}

    @log_exceptions(default_return={})
    def analyze_rate_hike_scenario(self) -> Dict[str, Any]:
        """Analyse l'impact d'une hausse de taux sur les secteurs (REITs, financiers, discrétionnaire)."""
        try:
            t2 = self.macro.get_fred_series("DGS2").value.iloc[-1]
            t10 = self.macro.get_fred_series("DGS10").value.iloc[-1]
            spread = t10 - t2 if not (pd.isna(t2) or pd.isna(t10)) else None
            reits = self.sector.get_sector_vs_market("USRT")
            financials = self.sector.get_sector_vs_market("IYG")
            discretionary = self.sector.get_sector_vs_market("XLY")
            return {
                "yield_curve_spread": spread,
                "reits_relative": reits["relative"].iloc[-1] if not reits.empty else None,
                "financials_relative": financials["relative"].iloc[-1] if not financials.empty else None,
                "discretionary_relative": discretionary["relative"].iloc[-1] if not discretionary.empty else None
            }
        except Exception as e:
            logger.error(f"Erreur analyse Fed: {e}")
            return {}

# ============================================================
# 7. FONCTIONS DE HAUT NIVEAU (COMPATIBILITÉ)
# ============================================================
def get_stock_overview(symbol: str) -> Dict[str, Any]:
    """Compatibilité avec l'ancienne API."""
    return FundamentalExtractor.get_stock_overview(symbol)

def get_market_sentiment() -> Dict[str, Any]:
    """Compatibilité - retourne les indicateurs de marché."""
    extractor = StockDataExtractor()
    return extractor.get_market_indicators()

def fetch_stock_data(symbol: str, period: str = "1y", include_technicals: bool = True) -> Dict[str, Any]:
    """Fonction de haut niveau pour récupérer toutes les données d'une action."""
    price_ext = StockDataExtractor(symbol)
    fund_ext = FundamentalExtractor()
    sent_ext = SentimentExtractor()
    macro_ext = MacroDataExtractor()

    historical = price_ext.get_historical_data(period=period)
    technical = price_ext.calculate_technical_indicators(historical) if include_technicals and not historical.empty else pd.DataFrame()
    fundamentals = fund_ext.get_fundamental_data(symbol)
    sentiment = sent_ext.get_sentiment_from_x(symbol)
    news = sent_ext.get_news_from_nyt(symbol)
    market_indicators = price_ext.get_market_indicators()

    result = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'overview': get_stock_overview(symbol),
        'historical': historical,
        'technical': technical,
        'fundamentals': fundamentals,
        'sentiment_x': sentiment,
        'news_nyt': news,
        'market_indicators': market_indicators
    }
    # Pour compatibilité JSON, on peut ajouter une version sérialisable
    if not historical.empty:
        result['history_json'] = {
            'dates': historical.index.strftime('%Y-%m-%d').tolist(),
            'closes': historical['Close'].tolist(),
            'rsi': technical['RSI'].tolist() if 'RSI' in technical.columns else []
        }
    return result

# Patch de la méthode get_market_indicators dans StockDataExtractor pour qu'elle utilise les nouvelles classes
def _get_market_indicators(self) -> Dict[str, Any]:
    """Récupère les indicateurs de marché (indices, breadth, secteurs)."""
    market_symbols = {
        'S&P 500': '^GSPC', 'NASDAQ': '^IXIC', 'Dow Jones': '^DJI',
        'VIX': '^VIX', 'Russell 2000': '^RUT', 'TSX Composite': '^GSPTSE',
        'NYSE Advance/Decline Line': '^NYAD', 'Equity Put/Call Ratio': '^CPCE'
    }
    sector_etfs = {
        'Financials (XLF)': 'XLF', 'Energy (XLE)': 'XLE',
        'Health Care (XLV)': 'XLV', 'Technology (XLK)': 'XLK',
        'Consumer Discretionary (XLY)': 'XLY'
    }
    data = {}
    for name, sym in market_symbols.items():
        df = _yf_history_with_fallback(sym, period='5d', interval='1d')
        if not df.empty:
            current = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
            change = ((current - prev) / prev * 100) if prev != 0 else 0
            data[name] = {'price': current, 'change': change, 'status': '🟢' if change > 0 else '🔴' if change < 0 else '⚪'}
    sectors = {}
    for name, sym in sector_etfs.items():
        df = _yf_history_with_fallback(sym, period='5d', interval='1d')
        if not df.empty:
            current = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2]) if len(df) > 1 else current
            change = ((current - prev) / prev * 100) if prev != 0 else 0
            sectors[name] = {'price': current, 'change': change, 'status': '🟢' if change > 0 else '🔴' if change < 0 else '⚪'}
    data['sectors'] = sectors
    # Ajout du score de santé
    analyzer = MarketHealthAnalyzer()
    health = analyzer.compute_market_health(data)
    data['overall_sentiment'] = health
    return data

# Monter la méthode dans StockDataExtractor
StockDataExtractor.get_market_indicators = _get_market_indicators

# Ajouter également la méthode evaluate_market_health pour compatibilité
def _evaluate_market_health(self) -> Dict[str, Any]:
    health = MarketHealthAnalyzer().compute_market_health(self.get_market_indicators())
    return {
        'score': health['score_out_of_100'],
        'rating': health['label'],
        'description': health['description'],
        'vix': self.get_market_indicators().get('VIX', {}).get('price'),
        'put_call_ratio': self.get_market_indicators().get('Equity Put/Call Ratio', {}).get('price'),
        'advance_decline': self.get_market_indicators().get('NYSE Advance/Decline Line', {}).get('change'),
        'indices': {k: v.get('change') for k, v in self.get_market_indicators().items() if k in ['S&P 500','NASDAQ','Dow Jones']},
        'sectors': {k: v.get('change', 0) for k, v in self.get_market_indicators().get('sectors', {}).items()}
    }
StockDataExtractor.evaluate_market_health = _evaluate_market_health

# ============================================================
# TEST (si exécuté directement)
# ============================================================
if __name__ == "__main__":
    logger.info("Test du module data_extraction refactorisé")
    symbol = "AAPL"
    # Test overview
    ov = get_stock_overview(symbol)
    print(f"Overview: {ov['name']} - ${ov['current_price']}")
    # Test historique
    ext = StockDataExtractor(symbol)
    hist = ext.get_historical_data(period="1mo")
    print(f"Historique: {len(hist)} jours")
    # Test indicateurs
    tech = ext.calculate_technical_indicators(hist)
    print(f"Indicateurs: {tech.shape[1]} colonnes")
    # Test fondamentaux
    fund = FundamentalExtractor().get_fundamental_data(symbol)
    print(f"Fondamentaux: PE = {fund.get('ratios',{}).get('peRatio')}")
    # Test sentiment
    sent = SentimentExtractor().get_sentiment_from_x(symbol)
    print(f"Sentiment X: {sent.get('sentiment_label')}")
    # Test macro
    macro = MacroDataExtractor().get_all_macro_data()
    print(f"Macro US: {len(macro.get('us_indicators', {}))} indicateurs")
    # Test market health
    health = MarketHealthAnalyzer().compute_market_health()
    print(f"Market health: {health['label']} ({health['score']:.1f})")
    logger.info("Test terminé")