'''Module d'extraction de données boursières avec intégration d'API multiples
API disponibles:
- Alpha Vantage: Time series, indicateurs techniques (5 calls/min, 500 calls/day)
- Yahoo Finance: Données en temps réel (via yfinance)
- Twitter API X: Sentiment social
- NY Times API: Actualités financières
- Financial Modeling Prep: Données fondamentales
- Polygon.io: Données boursières premium
- Statistique Canada: Données macroéconomiques Canada

Module d'extraction de données boursières avec intégration d'API multiples
API disponibles:
- Alpha Vantage: Time series, indicateurs techniques (5 calls/min, 500 calls/day)
- Yahoo Finance: Données en temps réel (via yfinance)
- Twitter API X: Sentiment social
- NY Times API: Actualités financières
- Financial Modeling Prep: Données fondamentales
- Polygon.io: Données boursières premium
- Statistique Canada: Données macroéconomiques Canada
- FED
'''

import yfinance as yf
import time
import pandas as pd
import numpy as np
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import lru_cache
import warnings
import json

warnings.filterwarnings('ignore')

# Do not configure logging here. The application entrypoint must call init_logging().
logger = logging.getLogger(__name__)

# persistent mapping: requested_symbol -> resolved_candidate
_SYMBOL_RESOLUTION_CACHE_FILE = os.path.join(os.getcwd(), "cache", "symbol_resolution.json")
_SYMBOL_RESOLUTION_CACHE: Dict[str, str] = {}

# negative cache to avoid retrying repeatedly for symbols that failed resolution
_SYMBOL_NEGATIVE_CACHE_FILE = os.path.join(os.getcwd(), "cache", "symbol_resolution_negative.json")
_SYMBOL_NEGATIVE_CACHE: Dict[str, float] = {}
_NEGATIVE_CACHE_TTL = 24 * 3600  # seconds to keep negative entry (24h)


def _load_symbol_resolution_cache():
    global _SYMBOL_RESOLUTION_CACHE
    try:
        os.makedirs(os.path.dirname(_SYMBOL_RESOLUTION_CACHE_FILE), exist_ok=True)
        if os.path.exists(_SYMBOL_RESOLUTION_CACHE_FILE):
            with open(_SYMBOL_RESOLUTION_CACHE_FILE, "r", encoding="utf-8") as f:
                _SYMBOL_RESOLUTION_CACHE = json.load(f) or {}
                if not isinstance(_SYMBOL_RESOLUTION_CACHE, dict):
                    _SYMBOL_RESOLUTION_CACHE = {}
    except Exception as e:
        logger.debug(f"Could not load symbol resolution cache: {e}")
        _SYMBOL_RESOLUTION_CACHE = {}


def _save_symbol_resolution_cache():
    try:
        os.makedirs(os.path.dirname(_SYMBOL_RESOLUTION_CACHE_FILE), exist_ok=True)
        with open(_SYMBOL_RESOLUTION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_SYMBOL_RESOLUTION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Could not save symbol resolution cache: {e}")


def _load_negative_cache():
    global _SYMBOL_NEGATIVE_CACHE
    try:
        os.makedirs(os.path.dirname(_SYMBOL_NEGATIVE_CACHE_FILE), exist_ok=True)
        if os.path.exists(_SYMBOL_NEGATIVE_CACHE_FILE):
            with open(_SYMBOL_NEGATIVE_CACHE_FILE, "r", encoding="utf-8") as f:
                _SYMBOL_NEGATIVE_CACHE = json.load(f) or {}
                if not isinstance(_SYMBOL_NEGATIVE_CACHE, dict):
                    _SYMBOL_NEGATIVE_CACHE = {}
    except Exception as e:
        logger.debug(f"Could not load negative symbol cache: {e}")
        _SYMBOL_NEGATIVE_CACHE = {}


def _save_negative_cache():
    try:
        os.makedirs(os.path.dirname(_SYMBOL_NEGATIVE_CACHE_FILE), exist_ok=True)
        with open(_SYMBOL_NEGATIVE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_SYMBOL_NEGATIVE_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Could not save negative symbol cache: {e}")


# load caches at module import
_load_symbol_resolution_cache()
_load_negative_cache()


def _is_recently_negative(symbol: str) -> bool:
    ts = _SYMBOL_NEGATIVE_CACHE.get(symbol)
    if not ts:
        return False
    return (time.time() - float(ts)) < _NEGATIVE_CACHE_TTL


def _mark_negative(symbol: str) -> None:
    _SYMBOL_NEGATIVE_CACHE[symbol] = time.time()
    _save_negative_cache()


# -- Helper: robust Yahoo Finance symbol resolution for historic queries ----------
def _yf_history_with_fallback(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Robust yfinance history fetch with:
      - candidate resolution (. <-> -, add .TO / -TO for short tickers)
      - transient retry (exponential backoff) for timeouts / network errors
      - treat HTTP 404 / "Quote not found" as permanent (do not negative-cache as transient)
      - negative-cache only when all candidates definitively have no data (404 or empty after retries)
    """
    import time
    from requests.exceptions import ReadTimeout, ConnectionError as ReqConnError
    from urllib.error import HTTPError

    tried = []
    transient_errors = (ReadTimeout, ReqConnError)
    permanent_failure_msgs = ("Not Found", "Quote not found", "No data found, symbol may be delisted")

    # Skip if recently negative
    if _is_recently_negative(symbol):
        logger.debug(f"Skipping resolution for {symbol}: negative cache hit")
        return pd.DataFrame()

    # Try cached resolved symbol first (if any)
    cached = _SYMBOL_RESOLUTION_CACHE.get(symbol)
    if cached:
        try:
            logger.debug(f"Trying cached resolve for {symbol} -> {cached}")
            df = yf.Ticker(cached).history(period=period, interval=interval, timeout=30)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"yfinance: using cached symbol '{cached}' for '{symbol}'")
                return df
            logger.debug(f"Cached symbol {cached} returned no data; invalidating cache and retrying resolution.")
            _SYMBOL_RESOLUTION_CACHE.pop(symbol, None)
            _save_symbol_resolution_cache()
        except Exception as e:
            logger.debug(f"Cached symbol attempt failed for {cached}: {e}")
            _SYMBOL_RESOLUTION_CACHE.pop(symbol, None)
            _save_symbol_resolution_cache()

    # Build candidate list (original + common variations)
    candidates = [symbol]
    if '.' in symbol:
        candidates.append(symbol.replace('.', '-'))
    if '-' in symbol:
        candidates.append(symbol.replace('-', '.'))
    if symbol.isalpha() and len(symbol) <= 6 and not symbol.upper().endswith(('.TO', '-TO')):
        candidates.append(symbol + ".TO")
        candidates.append(symbol + "-TO")
    candidates.append(symbol.upper())

    # Deduplicate preserving order
    seen = set()
    candidates_unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            candidates_unique.append(c)

    all_candidates_permanent_fail = True

    for cand in candidates_unique:
        tried.append(cand)
        # For each candidate allow a small retry loop for transient errors
        max_attempts = 3
        attempt = 0
        last_exception = None
        while attempt < max_attempts:
            attempt += 1
            try:
                # increase timeout on retries
                timeout = 10 + (attempt - 1) * 10
                df = yf.Ticker(cand).history(period=period, interval=interval, timeout=timeout)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # persist successful mapping (only when different from original)
                    if cand != symbol:
                        _SYMBOL_RESOLUTION_CACHE[symbol] = cand
                        _save_symbol_resolution_cache()
                        logger.info(f"yfinance: resolved '{symbol}' -> '{cand}' and cached mapping")
                    return df
                # empty DataFrame is not necessarily permanent (Yahoo sometimes returns empty on large period / transient)
                # mark as potential permanent but continue trying other candidates
                last_exception = None
                break
            except Exception as e:
                last_exception = e
                msg = str(e)
                # Detect explicit 404 / "Quote not found" errors -> treat as permanent for this candidate
                if any(m in msg for m in permanent_failure_msgs) or ("HTTP Error 404" in msg) or ("404" in msg and "Not Found" in msg):
                    logger.debug(f"Permanent failure for candidate {cand}: {msg}")
                    # Continue to next candidate (this candidate is permanent fail)
                    break
                # Transient network/timeouts -> retry with backoff
                if isinstance(e, transient_errors) or "timed out" in msg.lower() or "curl" in msg.lower():
                    backoff = 1.0 * attempt
                    logger.debug(f"Transient error for {cand} (attempt {attempt}/{max_attempts}): {msg} — retrying after {backoff}s")
                    time.sleep(backoff)
                    continue
                # Unknown exception: log and treat as transient (retry) for safety
                logger.debug(f"Unknown error for {cand} (attempt {attempt}/{max_attempts}): {msg}")
                time.sleep(0.5 * attempt)
                continue

        # If we reach here and candidate returned an explicit permanent failure message, it's not transient.
        # But we only mark global negative after trying all candidates.
        # If a candidate succeeded partially (empty df) we consider it not definitive.
        if last_exception:
            # if last_exception indicates transient, we don't treat as permanent failure
            if any(m in str(last_exception) for m in permanent_failure_msgs):
                logger.debug(f"Candidate {cand} permanent fail: {last_exception}")
            else:
                # a transient issue occurred; we should not assume all candidates are permanent failures
                all_candidates_permanent_fail = False

    # After all candidates tried: if all candidates produced permanent failures, mark negative cache
    if all_candidates_permanent_fail:
        _mark_negative(symbol)
        logger.warning(f"yfinance: no historical data for {symbol} (tried: {', '.join(tried)}) — negative cached")
    else:
        logger.warning(f"yfinance: no historical data for {symbol} (tried: {', '.join(tried)}) — transient or mixed failures")

    return pd.DataFrame()

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# Configuration unique des API Keys
API_CONFIG = {
    'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'FINANCIAL_MODELING_PREP': os.getenv('FMP_API_KEY'),
    'POLYGON': os.getenv('POLYGON_API_KEY'),
    'TWITTER_X_BEARER': os.getenv('TWITTER_X_BEARER'),
    'NY_TIMES': os.getenv('NY_TIMES_API_KEY'),
    'NY_TIMES_API_KEY_SECRET':os.getenv('NY_TIMES_API_KEY_SECRET'),
    'FRED_API_KEY':os.getenv('FRED_API_KEY')
}

# Vérification que les clés sont chargées
for key_name, key_value in API_CONFIG.items():
    if not key_value:
        logger.warning(f"Clé API manquante: {key_name}")
    elif 'demo' in str(key_value).lower() or len(str(key_value)) < 10:
        logger.warning(f"Clé API suspecte ou démo: {key_name}")
        # Utiliser la valeur 'demo' si la clé est invalide
        API_CONFIG[key_name] = 'demo'

class StockDataExtractor:
    """
    Classe principale pour l'extraction de données boursières
    Utilise multiple sources: Yahoo Finance, Alpha Vantage, Financial Modeling Prep
    """
    
    def __init__(self, symbol: str = None, cache_dir: str = 'cache'):
        self.symbol = symbol
        self.cache_dir = cache_dir
        self.data = None
        self.features = None
        self.targets = None
        self._setup_cache()
    
    def _setup_cache(self):
        """Configure le système de cache"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Cache initialisé dans: {self.cache_dir}")
        self.cache = {}

    def get_bulk_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Récupère les derniers prix de clôture pour une liste de tickers en une seule requête.
        Optimise radicalement le temps de chargement du dashboard.
        """
        if not tickers:
            return {}

        try:
            # Téléchargement groupé (yfinance supporte les listes séparées par des espaces)
            data = yf.download(
                tickers=" ".join(tickers), 
                period="1d", 
                interval="1m", # On prend la minute la plus récente
                group_by='ticker',
                threads=True,
                progress=False
            )

            prices = {}
            for ticker in tickers:
                try:
                    # Extraction du dernier prix 'Close' valide
                    if len(tickers) > 1:
                        last_price = data[ticker]['Close'].dropna().iloc[-1]
                    else:
                        # Cas particulier si un seul ticker est demandé, yfinance change le format du DF
                        last_price = data['Close'].dropna().iloc[-1]
                    
                    prices[ticker] = float(last_price)
                except Exception:
                    prices[ticker] = 0.0 # Valeur par défaut si échec
            
            return prices

        except Exception as e:
            print(f"Erreur lors du bulk fetch: {e}")
            return {}
   
    def get_historical_data(self, 
                           symbol: str = None, 
                           period: str = "3y",
                           interval: str = "1d") -> pd.DataFrame:
        """
        Récupère les données historiques depuis Yahoo Finance
        Args:
            symbol: Symbole boursier
            period: Période (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Intervalle (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        Returns:
            DataFrame avec données OHLCV
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            logger.info(f"Récupération données historiques pour {symbol} - période: {period}")
            
            # Utiliser le cache si disponible
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                logger.info("Utilisation des données en cache")
                return self.cache[cache_key].copy()
            
            # Télécharger depuis Yahoo Finance with robust fallback
            df = _yf_history_with_fallback(symbol, period, interval)
            
            if df.empty:
                logger.warning(f"Aucune donnée disponible pour {symbol}")
                return pd.DataFrame()
            
            # Standardiser les colonnes
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Colonne manquante: {col}")
                    return pd.DataFrame()
            
            # Ajouter des colonnes calculées
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Mettre en cache
            self.cache[cache_key] = df.copy()
            self.data = df.copy()
            
            logger.info(f"Données récupérées: {len(df)} périodes, {df.shape[1]} colonnes")
            return df
            
        except Exception as e:
            logger.error(f"Erreur get_historical_data: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str = None) -> Dict[str, Any]:
        """
        Récupère les données fondamentales depuis Financial Modeling Prep
        Gestion améliorée des erreurs 403
        """
        if symbol is None:
            symbol = self.symbol
    
        try:
            api_key = API_CONFIG['FINANCIAL_MODELING_PREP']
            if api_key == 'demo':
                logger.warning("Clé FMP démo - données limitées")
                return {}
        
            # Données de profil
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            profile_params = {'apikey': api_key}
        
            response = requests.get(profile_url, params=profile_params, timeout=10)
        
            if response.status_code == 403:
                logger.error(f"Accès interdit à l'API FMP (403) pour {symbol} - vérifiez votre clé ou abonnement")
                # Marquer la clé comme invalide pour éviter de nouveaux appels
                API_CONFIG['FINANCIAL_MODELING_PREP'] = 'demo'
                return {}
            elif response.status_code != 200:
                logger.warning(f"Erreur API FMP (profile): {response.status_code}")
                profile_data = []
            else:
                profile_data = response.json()
        
            # Ratios financiers
            ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
            ratios_params = {'apikey': api_key}
        
            ratios_response = requests.get(ratios_url, params=ratios_params, timeout=10)
        
            if ratios_response.status_code == 403:
                logger.error(f"Accès interdit à l'API FMP (ratios) - vérifiez votre clé ou abonnement")
                API_CONFIG['FINANCIAL_MODELING_PREP'] = 'demo'
                return {}
            elif ratios_response.status_code != 200:
                logger.warning(f"Erreur API FMP (ratios): {ratios_response.status_code}")
                ratios_data = []
            else:
                ratios_data = ratios_response.json()
        
            fundamental_data = {
                'profile': profile_data[0] if profile_data else {},
                'ratios': ratios_data[0] if ratios_data else {},
                'source': 'Financial Modeling Prep'
            }
        
            logger.info(f"Données fondamentales récupérées pour {symbol}")
            return fundamental_data
        
        except Exception as e:
            logger.error(f"Erreur données fondamentales pour {symbol}: {e}")
            return {}

    def get_fundamental_data_fallback(self, symbol: str) -> Dict[str, Any]:
        """Récupère des données fondamentales via Yahoo Finance"""
        try:
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
        except:
            return {}    
    def get_full_ticker_data(self, symbol: str, period="3y"):
        """Récupère prix + fondamentaux et prépare le format JSON."""
        try:
            # 1. Récupération des prix (ton code actuel)
            history = self.get_historical_data(symbol, period=period) 
            
            # 2. Récupération des fondamentaux
            fundamentals = self.get_fundamental_data_fallback(symbol)
            
            # 3. Fusion pour le cache
            payload = {
                'symbol': symbol,
                'last_updated': datetime.now().isoformat(),
                'profile': fundamentals.get('profile', {}),
                'ratios': fundamentals.get('ratios', {}),
                'history': {
                    'dates': history.index.strftime('%Y-%m-%d').tolist(),
                    'closes': history['Close'].tolist(),
                    'volumes': history['Volume'].tolist()
                }
            }
            return payload
        except Exception as e:
            logger.error(f"Erreur extraction complète pour {symbol}: {e}")
            return None
    def get_sentiment_from_x(self, symbol: str = None) -> Dict[str, Any]:
        """
        Récupère le sentiment depuis X (Twitter)
        Version corrigée avec gestion des erreurs améliorée
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            bearer_token = API_CONFIG['TWITTER_X_BEARER']
            
            # Vérifier si le token est valide (pas le token par défaut)
            if bearer_token == 'demo':
                logger.warning("Utilisation du token X par défaut - résultats limités")
                return self._get_simulated_sentiment(symbol)
            
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "v2RecentSearchPython"
            }
            
            # URL de l'API Twitter v2
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            # Paramètres de recherche
            params = {
                'query': f'${symbol} (stock OR finance OR investing) lang:en -is:retweet -is:reply',
                'max_results': 50,
                'tweet.fields': 'text,created_at,public_metrics',
                'expansions': 'author_id'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                
                if not tweets:
                    logger.info(f"Aucun tweet trouvé pour ${symbol}")
                    return {
                        'tweet_count': 0,
                        'avg_sentiment': 0,
                        'sentiment_label': 'NEUTRAL',
                        'total_likes': 0,
                        'total_retweets': 0,
                        'sample_tweets': [],
                        'source': 'Twitter/X API v2 (simulé)'
                    }
                
                # Analyse de sentiment simple
                positive_words = [
                    'bullish', 'buy', 'strong', 'growth', 'profit', 'gain', 'up', 'positive',
                    'good', 'great', 'excellent', 'amazing', 'awesome', 'profit', 'win',
                    'increase', 'rise', 'soar', 'surge', 'rally', 'outperform'
                ]
                
                negative_words = [
                    'bearish', 'sell', 'weak', 'drop', 'loss', 'down', 'negative', 'crash',
                    'bad', 'poor', 'terrible', 'awful', 'horrible', 'loss', 'lose',
                    'decrease', 'fall', 'plunge', 'drop', 'decline', 'underperform'
                ]
                
                sentiment_scores = []
                total_likes = 0
                total_retweets = 0
                sample_tweets = []
                
                for tweet in tweets[:10]:  # Analyser seulement les 10 premiers pour la performance
                    text = tweet.get('text', '').lower()
                    metrics = tweet.get('public_metrics', {})
                    
                    # Calcul du sentiment
                    positive_count = sum(1 for word in positive_words if word in text)
                    negative_count = sum(1 for word in negative_words if word in text)
                    
                    # Score de sentiment normalisé entre -1 et 1
                    if positive_count + negative_count > 0:
                        score = (positive_count - negative_count) / (positive_count + negative_count)
                    else:
                        score = 0
                    
                    sentiment_scores.append(score)
                    total_likes += metrics.get('like_count', 0)
                    total_retweets += metrics.get('retweet_count', 0)
                    
                    # Garder quelques tweets d'exemple
                    if len(sample_tweets) < 3 and len(text) > 20:
                        sample_tweets.append({
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'likes': metrics.get('like_count', 0),
                            'retweets': metrics.get('retweet_count', 0),
                            'sentiment': 'POSITIVE' if score > 0.1 else 'NEGATIVE' if score < -0.1 else 'NEUTRAL'
                        })
                
                # Calculer la moyenne des scores de sentiment
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                
                # Déterminer le label de sentiment
                if avg_sentiment > 0.2:
                    sentiment_label = 'POSITIVE'
                elif avg_sentiment < -0.2:
                    sentiment_label = 'NEGATIVE'
                else:
                    sentiment_label = 'NEUTRAL'
                
                sentiment_data = {
                    'tweet_count': len(tweets),
                    'avg_sentiment': float(avg_sentiment),
                    'sentiment_label': sentiment_label,
                    'total_likes': total_likes,
                    'total_retweets': total_retweets,
                    'sample_tweets': sample_tweets,
                    'source': 'Twitter/X API v2'
                }
                
                logger.info(f"Sentiment X pour ${symbol}: {avg_sentiment:.2f} ({sentiment_label}, {len(tweets)} tweets)")
                return sentiment_data
                
            else:
                logger.error(f"Erreur API X: {response.status_code}")
                # Retourner des données simulées en cas d'erreur
                return self._get_simulated_sentiment(symbol)
                
        except Exception as e:
            logger.error(f"Erreur get_sentiment_from_x pour {symbol}: {e}")
            # Retourner des données simulées en cas d'exception
            return self._get_simulated_sentiment(symbol)
    
    def _get_simulated_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Retourne des données de sentiment simulées pour les tests
        """
        logger.info(f"Utilisation de données de sentiment simulées pour {symbol}")
        
        # Générer un score aléatoire avec une légère tendance positive
        np.random.seed(hash(symbol) % 1000)
        avg_sentiment = np.random.uniform(-0.3, 0.6)
        
        if avg_sentiment > 0.2:
            sentiment_label = 'POSITIVE'
        elif avg_sentiment < -0.2:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'tweet_count': np.random.randint(50, 200),
            'avg_sentiment': float(avg_sentiment),
            'sentiment_label': sentiment_label,
            'total_likes': np.random.randint(500, 5000),
            'total_retweets': np.random.randint(50, 500),
            'sample_tweets': [
                {
                    'text': f"${symbol} looking strong today with good volume #stocks",
                    'likes': np.random.randint(10, 100),
                    'retweets': np.random.randint(5, 50),
                    'sentiment': 'POSITIVE'
                },
                {
                    'text': f"Watching ${symbol} for potential breakout above resistance",
                    'likes': np.random.randint(5, 50),
                    'retweets': np.random.randint(2, 20),
                    'sentiment': 'NEUTRAL'
                }
            ],
            'source': 'Twitter/X API v2 (simulé)'
        }
    def get_news_from_nyt(self, symbol: str = None, days: int = 7) -> Dict[str, Any]:
        """
        Récupère les articles du New York Times concernant une action
        Version corrigée avec gestion des erreurs améliorée
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            api_key = API_CONFIG['NY_TIMES']
            
            # Vérifier si la clé est valide
            if api_key == 'demo':
                logger.warning("Utilisation de la clé NY Times par défaut - résultats limités")
                return self._get_simulated_news(symbol)
            
            # Calculer la date de début
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
            params = {
                'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                'begin_date': from_date,
                'sort': 'newest',
                'api-key': api_key,
                'fl': 'headline,pub_date,web_url,snippet,source'  # Champs à récupérer
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                response_data = data.get('response', {})
                articles = response_data.get('docs')
                if articles is None:
                    articles = []                
                news_data = {
                    'total_articles': len(articles),
                    'articles': [],
                    'source': 'New York Times API',
                    'last_updated': datetime.now().isoformat()
                }
                
                # Analyser les articles
                for article in articles[:5]:  # Limiter aux 5 premiers articles
                    headline = article.get('headline', {}).get('main', 'No headline')
                    snippet = article.get('snippet', '')
                    pub_date = article.get('pub_date', '')
                    web_url = article.get('web_url', '')
                    source = article.get('source', 'New York Times')
                    
                    # Analyse simple du sentiment du titre
                    title_lower = headline.lower()
                    
                    positive_keywords = [
                        'gain', 'rise', 'surge', 'profit', 'growth', 'beat', 'increase',
                        'soar', 'rally', 'outperform', 'positive', 'strong', 'bullish',
                        'record', 'high', 'success', 'win', 'advantage', 'opportunity'
                    ]
                    
                    negative_keywords = [
                        'drop', 'fall', 'loss', 'decline', 'miss', 'cut', 'warn',
                        'plunge', 'slide', 'underperform', 'negative', 'weak', 'bearish',
                        'low', 'failure', 'lose', 'disadvantage', 'risk', 'crisis'
                    ]
                    
                    positive_count = sum(1 for word in positive_keywords if word in title_lower)
                    negative_count = sum(1 for word in negative_keywords if word in title_lower)
                    
                    if positive_count > negative_count:
                        sentiment = 'POSITIVE'
                    elif negative_count > positive_count:
                        sentiment = 'NEGATIVE'
                    else:
                        sentiment = 'NEUTRAL'
                    
                    news_data['articles'].append({
                        'headline': headline,
                        'snippet': snippet[:200] + '...' if len(snippet) > 200 else snippet,
                        'date': pub_date[:10] if pub_date else '',
                        'url': web_url,
                        'source': source,
                        'sentiment': sentiment,
                        'positive_keywords': positive_count,
                        'negative_keywords': negative_count
                    })
                
                logger.info(f"Actualités NYT pour {symbol}: {len(articles)} articles")
                return news_data
                
            else:
                logger.error(f"Erreur API NY Times: {response.status_code}")
                return self._get_simulated_news(symbol)
                
        except Exception as e:
            logger.error(f"Erreur get_news_from_nyt pour {symbol}: {e}")
            return self._get_simulated_news(symbol)
    
    def _get_simulated_news(self, symbol: str) -> Dict[str, Any]:
        """
        Retourne des données d'actualités simulées pour les tests
        """
        logger.info(f"Utilisation de données d'actualités simulées pour {symbol}")
        
        # Générer quelques articles simulés
        articles = []
        
        # Liste d'articles fictifs avec différents sentiments
        simulated_articles = [
            {
                'headline': f"{symbol} Reports Strong Quarterly Earnings, Beating Estimates",
                'snippet': f"{symbol} announced better-than-expected quarterly results, driven by strong demand for its products.",
                'sentiment': 'POSITIVE'
            },
            {
                'headline': f"Analysts Raise Price Target for {symbol} Amid Market Optimism",
                'snippet': f"Several Wall Street analysts have increased their price targets for {symbol} following positive industry trends.",
                'sentiment': 'POSITIVE'
            },
            {
                'headline': f"{symbol} Faces Regulatory Scrutiny Over Business Practices",
                'snippet': f"Regulatory authorities are investigating certain business practices at {symbol}, according to sources.",
                'sentiment': 'NEGATIVE'
            },
            {
                'headline': f"{symbol} Announces New Product Launch Next Month",
                'snippet': f"{symbol} has scheduled a product launch event for next month, with expectations of new innovations.",
                'sentiment': 'NEUTRAL'
            }
        ]
        
        # Sélectionner aléatoirement 2-3 articles
        np.random.seed(hash(symbol) % 1000)
        selected_indices = np.random.choice(len(simulated_articles), size=np.random.randint(2, 4), replace=False)
        
        for idx in selected_indices:
            article = simulated_articles[idx]
            articles.append({
                'headline': article['headline'],
                'snippet': article['snippet'],
                'date': (datetime.now() - timedelta(days=np.random.randint(0, 7))).strftime('%Y-%m-%d'),
                'url': f"https://www.nytimes.com/simulated/{symbol.lower()}/article{idx}",
                'source': 'New York Times (simulé)',
                'sentiment': article['sentiment']
            })
        
        return {
            'total_articles': len(articles),
            'articles': articles,
            'source': 'New York Times API (simulé)',
            'last_updated': datetime.now().isoformat()
        }
    def add_external_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Ajoute des features externes au DataFrame des indicateurs techniques en conservant toutes les colonnes existantes.
        """
        if symbol is None:
            symbol = self.symbol
        if symbol is None:
            logger.warning("Aucun symbole fourni pour les features externes")
            return df
    
        # Important : copier le DataFrame pour ne pas modifier l'original
        df_ext = df.copy()
        logger.info(f"Avant ajout external features : {df_ext.shape[1]} colonnes")
    
        # 1. Données fondamentales (via fallback Yahoo Finance)
        try:
            fund_data = self.get_fundamental_data_fallback(symbol)
            if fund_data:
                profile = fund_data.get('profile', {})
                ratios = fund_data.get('ratios', {})
                for key, value in profile.items():
                    if isinstance(value, (int, float)):
                        df_ext[f'fund_{key}'] = value
                for key, value in ratios.items():
                    if isinstance(value, (int, float)):
                        df_ext[f'fund_{key}'] = value
                logger.info(f"Ajout de {len(profile)+len(ratios)} features fondamentales")
        except Exception as e:
            logger.warning(f"Erreur ajout features fondamentales: {e}")
    
        # 2. Sentiment Twitter
        try:
            sentiment = self.get_sentiment_from_x(symbol)
            if sentiment:
                avg_sent = sentiment.get('avg_sentiment', 0)
                tweet_count = sentiment.get('tweet_count', 0)
                total_likes = sentiment.get('total_likes', 0)
                df_ext['sentiment_score'] = avg_sent
                df_ext['sentiment_tweet_count'] = tweet_count
                df_ext['sentiment_total_likes'] = total_likes
                logger.info("Ajout de 3 features de sentiment Twitter")
        except Exception as e:
            logger.warning(f"Erreur ajout features sentiment: {e}")
    
        # 3. Actualités NYT
        try:
            news = self.get_news_from_nyt(symbol, days=30)
            if news and 'articles' in news:
                articles_df = pd.DataFrame(news['articles'])
                if not articles_df.empty:
                    articles_df['date'] = pd.to_datetime(articles_df['date'])
                    articles_df = articles_df.set_index('date')
                    daily_articles = articles_df.resample('D').size()
                    daily_articles = daily_articles.reindex(df_ext.index, method='ffill').fillna(0)
                    df_ext['news_count'] = daily_articles
                    sentiment_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
                    articles_df['sentiment_value'] = articles_df['sentiment'].map(sentiment_map).fillna(0)
                    daily_sentiment = articles_df['sentiment_value'].resample('D').mean()
                    daily_sentiment = daily_sentiment.reindex(df_ext.index, method='ffill').fillna(0)
                    df_ext['news_sentiment'] = daily_sentiment
                    logger.info("Ajout de 2 features d'actualités NYT")
        except Exception as e:
            logger.warning(f"Erreur ajout features news: {e}")
    
        # 4. Indicateurs macroéconomiques
        try:
            macro = MacroDataExtractor()
            econ = macro.get_economic_indicators('US')
            for name, data in econ.items():
                if isinstance(data, dict) and 'value' in data:
                    df_ext[f'macro_{name}'] = data['value']
            logger.info(f"Ajout de {len(econ)} features macroéconomiques")
        except Exception as e:
            logger.warning(f"Erreur ajout features macro: {e}")
    
        # 5. Taux de change
        try:
            macro = MacroDataExtractor()
            currencies = macro.get_currency_rates()
            for pair, data in currencies.items():
                if isinstance(data, dict) and 'rate' in data:
                    df_ext[f'forex_{pair.replace("/", "_")}'] = data['rate']
            logger.info(f"Ajout de {len(currencies)} features de change")
        except Exception as e:
            logger.warning(f"Erreur ajout features change: {e}")
    
        logger.info(f"Après ajout external features : {df_ext.shape[1]} colonnes (soit {df_ext.shape[1] - df.shape[1]} nouvelles)")
        return df_ext
    def calculate_technical_indicators(self, data: pd.DataFrame = None, include_external: bool = True) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques sur les données OHLCV de manière optimisée
        Args:
            data: DataFrame avec colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
        Returns:
            DataFrame avec indicateurs techniques ajoutés
        """
        if data is None:
            data = self.data
    
        if data is None or data.empty:
            logger.warning("Aucune donnée pour calculer les indicateurs")
            return pd.DataFrame()
    
        try:
            df = data.copy()
        
            # Stocker les séries pour éviter les appels répétés
            close = df['Close'].astype('float32')
            high = df['High'].astype('float32')
            low = df['Low'].astype('float32')
            volume = df['Volume'].astype('float32')
        
            # Dictionnaire pour stocker les indicateurs temporaires
            indicators = {}
        
            # 1. Moyennes mobiles (calcul groupé)
            sma_periods = [5, 10, 20, 50, 200]
            for period in sma_periods:
                indicators[f'SMA_{period}'] = close.rolling(window=period, min_periods=1).mean()
                indicators[f'EMA_{period}'] = close.ewm(span=period, adjust=False, min_periods=1).mean()
        
            # 2. RSI optimisé
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
        
            # Utiliser EWM pour un RSI plus réactif
            avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            indicators['RSI'] = 100 - (100 / (1 + rs))
        
            # 3. MACD
            ema_12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
            ema_26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
            indicators['MACD'] = ema_12 - ema_26
            indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
            # 4. Bandes de Bollinger
            bb_middle = close.rolling(window=20, min_periods=1).mean()
            bb_std = close.rolling(window=20, min_periods=1).std()
            indicators['BB_Middle'] = bb_middle
            indicators['BB_Upper'] = bb_middle + (bb_std * 2)
            indicators['BB_Lower'] = bb_middle - (bb_std * 2)
            indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / bb_middle.replace(0, np.finfo(float).eps)
        
            # 5. ATR optimisé (avec alignement d'index)
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            true_range = np.maximum.reduce([tr1, tr2, tr3])
            true_range_series = pd.Series(true_range, index=df.index)
            indicators['ATR'] = true_range_series.rolling(window=14, min_periods=1).mean()

            # 6. Stochastic Oscillator optimisé
            low_14 = low.rolling(window=14, min_periods=1).min()
            high_14 = high.rolling(window=14, min_periods=1).max()
            denominator = (high_14 - low_14).replace(0, np.finfo(float).eps)
            indicators['Stoch_%K'] = 100 * ((close - low_14) / denominator)
            indicators['Stoch_%D'] = indicators['Stoch_%K'].rolling(window=3, min_periods=1).mean()
        
            # 7. OBV vectorisé
            price_change = np.sign(close.diff())
            obv = (price_change * volume).fillna(0).cumsum()
            indicators['OBV'] = obv.astype('float32')
        
            # 8. VWAP correct (sur rolling window pour être réaliste)
            typical_price = (high + low + close) / 3
            vwap_window = 20  # Window pour VWAP
            indicators['VWAP'] = (typical_price * volume).rolling(window=vwap_window).sum() / volume.rolling(window=vwap_window).sum()
        
            # 9. Returns and Volatility optimisés
            returns = close.pct_change()
            indicators['Returns'] = returns
            volatility_windows = [5, 10, 20]
            for window in volatility_windows:
                indicators[f'Volatility_{window}d'] = returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        
            # 10. Momentum optimisé
            for period in [5, 10, 20]:
                indicators[f'Momentum_{period}'] = close.pct_change(periods=period)
        
            # 11. Support and Resistance avec lookback raisonnable
            lookback = 20
            indicators['Support'] = low.rolling(window=lookback, min_periods=1).min()
            indicators['Resistance'] = high.rolling(window=lookback, min_periods=1).max()
        
            # 12. Price Position normalisée
            range_diff = indicators['Resistance'] - indicators['Support']
            range_diff = range_diff.replace(0, np.finfo(float).eps)
            indicators['Price_Position'] = (close - indicators['Support']) / range_diff
        
            # 13. ADX optimisé
            tr = indicators['ATR'] * 14  # Approx
            up_move = high.diff()
            down_move = low.diff()
        
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
            plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr.replace(0, np.finfo(float).eps)
            minus_di = 100 * pd.Series(np.abs(minus_dm), index=df.index).ewm(alpha=1/14, adjust=False).mean() / tr.replace(0, np.finfo(float).eps)
        
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
            indicators['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
            # 14. Chaikin Oscillator optimisé
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.finfo(float).eps)
            adl = money_flow_multiplier * volume
            indicators['Chaikin_Oscillator'] = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
        
            # 15. Money Flow Index optimisé
            typical_price_mfi = typical_price
            money_flow = typical_price_mfi * volume
        
            positive_mask = typical_price_mfi > typical_price_mfi.shift(1)
            positive_flow = np.where(positive_mask, money_flow, 0)
            negative_flow = np.where(~positive_mask, money_flow, 0)
        
            positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=14, min_periods=1).sum()
            negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=14, min_periods=1).sum()
        
            mfr = positive_mf / negative_mf.replace(0, np.finfo(float).eps)
            indicators['MFI'] = 100 - (100 / (1 + mfr))
        
            # 16. Ajout de nouveaux indicateurs utiles
            typical_price_cci = typical_price
            sma_typical = typical_price_cci.rolling(window=20, min_periods=1).mean()
            mad = typical_price_cci.rolling(window=20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            indicators['CCI'] = (typical_price_cci - sma_typical) / (0.015 * mad.replace(0, np.finfo(float).eps))
        
            # Williams %R
            williams_lookback = 14
            highest_high = high.rolling(window=williams_lookback, min_periods=1).max()
            lowest_low = low.rolling(window=williams_lookback, min_periods=1).min()
            indicators['Williams_%R'] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.finfo(float).eps)
        
            # Price Channels
            indicators['Price_Channel_High'] = high.rolling(window=20, min_periods=1).max()
            indicators['Price_Channel_Low'] = low.rolling(window=20, min_periods=1).min()
        
            # Rate of Change
            for period in [10, 20, 50]:
                indicators[f'ROC_{period}'] = close.pct_change(periods=period) * 100
        
            # Ajouter tous les indicateurs au DataFrame
            for name, values in indicators.items():
                df[name] = values
        
            # 17. ajout des indicateurs externe
            logger.info(f"Nombre de colonnes techniques avant ajout externe : {df.shape[1]}")
            if include_external and self.symbol:
                df = self.add_external_features(df, self.symbol)
    
            # 18. Nettoyage intelligent des données
            nan_threshold = 0.5  # 50% maximum de NaN
            cols_to_drop = []
            for col in df.columns:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio > nan_threshold:
                    cols_to_drop.append(col)
                    logger.info(f"Colonne {col} supprimée ({nan_ratio:.1%} de NaN)")
        
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        
            indicator_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            df[indicator_cols] = df[indicator_cols].ffill(limit=5).bfill(limit=5)
            for col in indicator_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        
            keep_from = max(200, int(len(df) * 0.05))
            initial_length = len(df)
        
            if len(df) > keep_from:
                df = df.iloc[keep_from:]
        
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
        
            self.features = df
            logger.info(f"Indicateurs techniques calculés: {df.shape[1]} colonnes, {len(df)} lignes (réduction de {initial_length} à {len(df)})")
        
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} valeurs NaN restantes après nettoyage")
        
            return df
        
        except Exception as e:
            logger.error(f"Erreur calculate_technical_indicators: {e}", exc_info=True)
            return data.copy() if data is not None else pd.DataFrame()  
        
    def create_target_columns(self, 
                             data: pd.DataFrame = None, 
                             forecast_days: List[int] = None) -> pd.DataFrame:
        """
        Crée les colonnes cibles pour l'entraînement du modèle
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
                targets[f'Target_Open_{days}d'] = df['Open'].shift(-days)
                targets[f'Target_High_{days}d'] = df['High'].shift(-days)
                targets[f'Target_Low_{days}d'] = df['Low'].shift(-days)
                targets[f'Target_Close_{days}d'] = close.shift(-days)
                targets[f'Target_Volume_{days}d'] = df['Volume'].shift(-days)
                targets[f'Target_Return_{days}d'] = close.shift(-days) / close - 1
        
            for days in forecast_days:
                if days >= 1:
                    targets[f'Target_Direction_{days}d'] = (close.shift(-days) > close).astype(int)
        
            future_returns = close.pct_change()
            volatility_windows = [w for w in [5, 10, 20] if w <= max(forecast_days)]
            for days in volatility_windows:
                targets[f'Target_Volatility_{days}d'] = future_returns.rolling(days).std().shift(-days)
        
            self.targets = targets
            logger.info(f"Cibles créées: {targets.shape[1]} colonnes")
        
            return targets
        
        except KeyError as e:
            logger.error(f"Colonne manquante dans les données: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur create_target_columns: {e}")
            return pd.DataFrame()
    def _compute_market_health_score(self, market_data: Dict) -> float:

        components = []

        # -------------------------
        # 1. VIX (20%)
        # -------------------------
        vix = market_data.get('VIX', {}).get('price')
        if vix is not None:
            vix_score = max(0, min(100, (30 - vix) / 15 * 100))
            components.append(vix_score * 0.20)

        # -------------------------
        # 2. Put/Call (15%)
        # -------------------------
        put_call = market_data.get('Equity Put/Call Ratio', {}).get('price')
        if put_call is not None:
            put_call_score = max(0, min(100, (1.2 - put_call) / 1.2 * 100))
            components.append(put_call_score * 0.15)

        # -------------------------
        # 3. Breadth (15%)
        # -------------------------
        adv_decl = market_data.get('NYSE Advance/Decline Line', {})
        if adv_decl:
            change_adv = adv_decl.get('change', 0)
            breadth_score = max(0, min(100, 50 + change_adv))
            components.append(breadth_score * 0.15)

        # -------------------------
        # 4. Indices (30%)
        # -------------------------
        indices = ['S&P 500', 'NASDAQ', 'Dow Jones']
        perf_scores = []

        for idx in indices:
            change = market_data.get(idx, {}).get('change')
            if change is not None:
                score = max(0, min(100, 50 + change * 25))
                perf_scores.append(score)

        if perf_scores:
            components.append(np.mean(perf_scores) * 0.30)

        # -------------------------
        # 5. Sectors (20%)
        # -------------------------
        sectors = market_data.get('sectors', {})
        if sectors:
            positive = sum(1 for s in sectors.values() if s.get('change', 0) > 0)
            sector_score = (positive / len(sectors)) * 100
            components.append(sector_score * 0.20)

        # -------------------------
        # FINAL SCORE
        # -------------------------
        if not components:
            return 50.0

        score = sum(components)

        return max(0, min(100, score))   
    def get_market_indicators(self) -> Dict[str, Any]:
        """
        Récupère les indicateurs de marché globaux (indices majeurs + breadth + volatilité + sentiment).
        Retourne un dictionnaire contenant les données et un score composite.
        """
        try:
            # Liste des symboles à surveiller
            market_symbols = {
                'S&P 500': '^GSPC',
                'NASDAQ': '^IXIC',
                'Dow Jones': '^DJI',
                'VIX': '^VIX',
                'Russell 2000': '^RUT',
                'TSX Composite': '^GSPTSE',
                # Indices de largeur de marché (breadth) – disponibles chez Yahoo
                'NYSE Advance/Decline Line': '^NYAD',      # ligne A/D
                'Equity Put/Call Ratio': '^CPCE',          # ratio put/call équité
            }

            # ETFs sectoriels pour observer les rotations
            sector_etfs = {
                'Financials (XLF)': 'XLF',
                'Energy (XLE)': 'XLE',
                'Health Care (XLV)': 'XLV',
                'Technology (XLK)': 'XLK',
                'Consumer Discretionary (XLY)': 'XLY'
            }

            market_data = {}
            # Récupération des indices
            for name, symbol in market_symbols.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        logger.warning(f"Aucune donnée pour {name} ({symbol}) - skipping")
                        continue
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change_pct = ((current - previous) / previous * 100) if previous != 0 else 0
                    market_data[name] = {
                        'price': current,
                        'change': change_pct,
                        'status': '🟢' if change_pct > 0 else '🔴' if change_pct < 0 else '⚪'
                    }
                except Exception as e:
                    logger.warning(f"Erreur pour {name}: {e}")
                    continue

            # Récupération des ETFs sectoriels
            sector_data = {}
            for name, symbol in sector_etfs.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        continue
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change_pct = ((current - previous) / previous * 100) if previous != 0 else 0
                    sector_data[name] = {
                        'price': current,
                        'change': change_pct,
                        'status': '🟢' if change_pct > 0 else '🔴' if change_pct < 0 else '⚪'
                    }
                except Exception:
                    pass

            market_data['sectors'] = sector_data

            # Calcul du score de santé du marché
            health_score = self._compute_market_health_score(market_data)
            market_data['overall_sentiment'] = {
                'score': health_score,
                'label': 'Bullish' if health_score > 70 else 'Bearish' if health_score < 30 else 'Neutral',
                'description': 'Haussier' if health_score > 70 else 'Baissier' if health_score < 30 else 'Neutre',
                'score_out_of_100': health_score
            }

            return market_data

        except Exception as e:
            logger.error(f"Erreur get_market_indicators: {e}", exc_info=True)
            return {}
    def get_all_data(self,
                     symbol: str = None,
                     period: str = "1y") -> Dict[str, Any]:
        """
        Récupère un ensemble complet de données pour un symbole donné.

        Retourne un dict contenant au minimum les clefs :
          - 'symbol'
          - 'timestamp'
          - 'overview'
          - 'historical'   -> pd.DataFrame (peut être vide)
          - 'technical'    -> pd.DataFrame ou None
          - 'targets'      -> pd.DataFrame ou None
          - 'fundamentals'
          - 'sentiment_x'
          - 'news_nyt'
          - 'market_indicators'
        Cette méthode est utilisée par `StockModelTrain.fetch_data()` et doit toujours
        renvoyer un dict (même en cas d'erreur) pour éviter les AttributeError.
        """
        if symbol is None:
            symbol = self.symbol

        try:
            logger.info(f"Récupération complète des données pour {symbol} (period={period})")

            # Overview (quick meta)
            overview = get_stock_overview(symbol)

            # Historical OHLCV
            historical = self.get_historical_data(symbol=symbol, period=period, interval="1d")
            if historical is None:
                historical = pd.DataFrame()

            # Fundamentals (primary) with fallback handled inside method
            fundamentals = self.get_fundamental_data(symbol) or {}
            # Sentiment X (Twitter)
            sentiment_x = self.get_sentiment_from_x(symbol) or {}
            # NYT news
            news_nyt = self.get_news_from_nyt(symbol) or {}

            # Market indicators (global context)
            market_indicators = self.get_market_indicators() or {}

            technical = None
            targets = None

            # If we have historical data, compute technicals and targets
            if isinstance(historical, pd.DataFrame) and not historical.empty:
                try:
                    technical = self.calculate_technical_indicators(historical)
                except Exception as e:
                    logger.warning(f"Erreur calcul technical pour {symbol}: {e}")
                    technical = pd.DataFrame()

                try:
                    targets = self.create_target_columns(historical)
                except Exception as e:
                    logger.warning(f"Erreur création targets pour {symbol}: {e}")
                    targets = pd.DataFrame()
            else:
                technical = pd.DataFrame()
                targets = pd.DataFrame()

            all_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'overview': overview,
                'historical': historical,
                'technical': technical,
                'targets': targets,
                'fundamentals': fundamentals,
                'sentiment_x': sentiment_x,
                'news_nyt': news_nyt,
                'market_indicators': market_indicators
            }

            logger.info(f"Données complètes récupérées pour {symbol}")
            return all_data

        except Exception as e:
            logger.error(f"Erreur get_all_data pour {symbol}: {e}", exc_info=True)
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'overview': {},
                'historical': pd.DataFrame(),
                'technical': pd.DataFrame(),
                'targets': pd.DataFrame(),
                'fundamentals': {},
                'sentiment_x': {},
                'news_nyt': {},
                'market_indicators': {},
                'error': str(e)
            }
    
    def get_commodity_prices(self) -> Dict[str, Any]:
        """Récupère les prix des matières premières (robuste)"""
        try:
            commodities = {
                'Gold': 'GC=F',
                'Silver': 'SI=F',
                'Copper': 'HG=F',
                'Crude Oil': 'CL=F',
                'Natural Gas': 'NG=F',
                'Wheat': 'ZW=F',
                'Corn': 'ZC=F'
            }
            
            prices = {}
            
            for name, symbol in commodities.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        logger.warning(f"Aucune donnée pour matière première {name} ({symbol})")
                        continue
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change = ((current - previous) / previous * 100) if previous != 0 else 0
                    prices[name] = {
                        'price': float(current),
                        'change': float(change),
                        'currency': 'USD' if 'Oil' in name or 'Gas' in name else 'USD/oz' if name in ['Gold', 'Silver'] else 'USD/lb',
                        'trend': '🟢' if change > 0 else '🔴' if change < 0 else '⚪'
                    }
                except Exception as e:
                    logger.warning(f"Erreur matière première {name}: {e}")
                    continue
            
            return prices
            
        except Exception as e:
            logger.error(f"Erreur get_commodity_prices: {e}")
            return {}

    def get_currency_rates(self) -> Dict[str, Any]:
        """
        Récupère les taux de change (robuste) :
          - Essaye d'abord via yfinance pour symboles courants (ex: CAD=X, EUR=X, JPY=X).
          - Si yfinance ne retourne rien (timeouts / pas de données), fallback vers l'API publique
            gratuite `exchangerate.host` pour obtenir le taux spot.
        Retourne dict:
          { 'USD/CAD': {'rate': 1.23, 'change': 0.12, 'trend': '🟢', 'source': 'yfinance'|'exchangerate.host'}, ... }
        """
        try:
            currencies = {
                'USD/CAD': 'CAD=X',
                'USD/EUR': 'EUR=X',
                'USD/JPY': 'JPY=X',
                'USD/GBP': 'GBP=X',
                'USD/CHF': 'CHF=X',
                'USD/AUD': 'AUD=X'
            }

            rates: Dict[str, Any] = {}
            fallback_pairs = []

            # First attempt: yfinance per pair
            for pair, symbol in currencies.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        logger.warning(f"Aucune donnée pour devise {pair} ({symbol}) via yfinance")
                        fallback_pairs.append(pair)
                        continue

                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change = ((current - previous) / previous * 100) if previous != 0 else 0.0
                    # Trend heuristic for display
                    trend = '🟢' if change > 0 else '🔴' if change < 0 else '⚪'

                    rates[pair] = {
                        'rate': float(current),
                        'change': float(change),
                        'trend': trend,
                        'source': 'yfinance'
                    }
                except Exception as e:
                    logger.debug(f"yfinance error for {pair} ({symbol}): {e}")
                    fallback_pairs.append(pair)

            # If some pairs failed with yfinance, use exchangerate.host as fallback (single call)
            if fallback_pairs:
                try:
                    # Build unique symbol list to request (right side of pair, e.g. 'EUR', 'JPY', 'CAD' ...)
                    symbols_to_request = sorted({p.split('/')[-1] for p in fallback_pairs})
                    if symbols_to_request:
                        params = {'base': 'USD', 'symbols': ','.join(symbols_to_request)}
                        resp = requests.get("https://api.exchangerate.host/latest", params=params, timeout=10)
                        if resp.ok:
                            data = resp.json()
                            rates_data = data.get('rates', {})
                            for p in fallback_pairs:
                                code = p.split('/')[-1]
                                rate = rates_data.get(code)
                                if rate is None:
                                    logger.warning(f"exchangerate.host: pas de taux pour {code}")
                                    continue
                                # No previous rate available from exchangerate.host simple endpoint -> change = 0
                                rates[p] = {
                                    'rate': float(rate),
                                    'change': 0.0,
                                    'trend': '⚪',
                                    'source': 'exchangerate.host'
                                }
                        else:
                            logger.warning(f"exchangerate.host returned status {resp.status_code}")
                            # leave those pairs missing (they will not be included)
                except Exception as e:
                    logger.warning(f"Erreur fallback exchangerate.host: {e}")

            return rates

        except Exception as e:
            logger.error(f"Erreur get_currency_rates: {e}", exc_info=True)
            return {}
    
    def get_all_macro_data(self) -> Dict[str, Any]:
        """Récupère toutes les données macroéconomiques"""
        try:
            macro_data = {
                'timestamp': datetime.now().isoformat(),
                'us_indicators': self.get_economic_indicators('US'),
                'canada_indicators': self.get_economic_indicators('CA'),
                'commodities': self.get_commodity_prices(),
                'currencies': self.get_currency_rates()
            }
            
            logger.info("Données macroéconomiques complètes récupérées")
            return macro_data
            
        except Exception as e:
            logger.error(f"Erreur get_all_macro_data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Fonction utilitaire pour obtenir le sentiment de marché
        """
        extractor = StockDataExtractor()
        return extractor.get_market_indicators()

    def get_market_health_score(self, market_data: Dict) -> float:
        return self._compute_market_health_score(market_data)

    def evaluate_market_health(self) -> Dict[str, Any]:
        """
        Évalue la santé du marché et retourne un résumé avec score et analyse.
        """
        try:
            indicators = self.get_market_indicators()
            if not indicators:
                return {'error': 'Impossible de récupérer les indicateurs de marché'}

            health_score = indicators.get('overall_sentiment', {}).get('score_out_of_100', 50)
            analysis = {
                'score': health_score,
                'rating': indicators['overall_sentiment']['label'],
                'description': indicators['overall_sentiment']['description'],
                'vix': indicators.get('VIX', {}).get('price'),
                'put_call_ratio': indicators.get('Equity Put/Call Ratio', {}).get('price'),
                'advance_decline': indicators.get('NYSE Advance/Decline Line', {}).get('change'),
                'indices': {
                    'sp500': indicators.get('S&P 500', {}).get('change'),
                    'nasdaq': indicators.get('NASDAQ', {}).get('change'),
                    'dow': indicators.get('Dow Jones', {}).get('change')
                },
                'sectors': {name: data.get('change', 0) for name, data in indicators.get('sectors', {}).items()}
            }
            return analysis
        except Exception as e:
            logger.error(f"Erreur evaluate_market_health: {e}")
            return {'error': str(e)}

# Insert this class definition above the existing `def get_market_sentiment(self) -> Dict[str, Any]:`
# (i.e. before line that currently reads `def get_market_sentiment(self) -> Dict[str, Any]:`)
class MacroDataExtractor:
    """
    Extracteur de données macroéconomiques intégrant Statistique Canada (WDS) 
    et Yahoo Finance pour les indicateurs globaux.
    Extracteur de données macroéconomiques.
    Fournit des wrappers robustes autour de yfinance et StatCan.
    class MacroDataExtractor:
    
    Extraction de données macro pertinentes pour les scénarios Fed / secteurs.
    Inclut FRED + Statistique Canada.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
        self.FRED_API_KEY = os.getenv("FRED_API_KEY")
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.logger = logging.getLogger(__name__)
        logger.info("MacroDataExtractor initialisé")
        self.cache_duration = timedelta(hours=6)

    @staticmethod
    def statistique_canada_sdw(method: str, params: Dict = None) -> Optional[Any]:
        """
        Accède aux données de Statistique Canada Web Data Service (WDS).
        IMPORTANT: Méthode statique sans 'self' pour éviter les erreurs de signature.
        """
        supported_methods = [
            'getDataFromCubePidCoordAndLatestNPeriods',
            'getDataFromVectorsAndLatestNPeriods',
            'getBulkVectorDataByRange',
            'getCubeMetadata',
            'getSeriesInfoFromVector'
        ]
    
        if method not in supported_methods:
            logging.error(f"StatCan: Méthode '{method}' non supportée.")
            return None
    
        base_url = "https://www150.statcan.gc.ca/t1/wds/rest"
        url = f"{base_url}/{method}"
    
        try:
            headers = {'Content-Type': 'application/json'}
            payload = []

            # Préparation spécifique pour la méthode de récupération par vecteurs
            if method == 'getDataFromVectorsAndLatestNPeriods':
                vector_ids = params.get('vectorIds') or params.get('vectorId') or []
                latest_n = params.get('latestN') or 1
                
                if not isinstance(vector_ids, (list, tuple)):
                    vector_ids = [v.strip() for v in str(vector_ids).split(',') if v.strip()]
                
                for v in vector_ids:
                    # Extraction du numérique uniquement (ex: v62305755 -> 62305755)
                    v_digits = ''.join(filter(str.isdigit, str(v)))
                    if v_digits:
                        payload.append({
                            "vectorId": int(v_digits), 
                            "latestN": int(latest_n)
                        })
            else:
                payload = params or {}

            if not payload and method == 'getDataFromVectorsAndLatestNPeriods':
                return None

            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logging.error(f"Erreur API Statistique Canada ({method}): {e}")
            return None

    def get_economic_indicators(self, country: str = "US") -> Dict[str, Any]:
        """
        Récupère les indicateurs économiques (PIB, Taux, Volatilité, etc.)
        """
        now = datetime.now()
        cache_key = f"macro_{country}_{now.strftime('%Y-%m-%d')}"
        
        # Gestion du cache interne
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if now - cache_time < self.cache_duration:
                return data

        indicators: Dict[str, Any] = {}

        # --- CAS CANADA (Statistique Canada) ---
        if country == "CA":
            try:
                # v62305755 : PIB aux prix de base (Indice)
                params = {'vectorIds': ['v62305755'], 'latestN': 1}
                res = self.statistique_canada_sdw('getDataFromVectorsAndLatestNPeriods', params)
                
                # StatCan renvoie une liste d'objets (un par vecteur)
                if res and isinstance(res, list) and len(res) > 0:
                    vec_obj = res[0]
                    if vec_obj.get('status') == 'SUCCESS':
                        data_points = vec_obj.get('object', {}).get('vectorDataPoint', [])
                        if data_points:
                            latest = data_points[0]
                            indicators['GDP Canada'] = {
                                'value': latest.get('value'),
                                'date': latest.get('refPer'),
                                'unit': 'Index (CAD)',
                                'source': 'Statistique Canada'
                            }
            except Exception as e:
                self.logger.warning(f"Échec extraction PIB Canada: {e}")

        # --- INDICATEURS US / GLOBAUX (Yahoo Finance) ---
        # On utilise ta fonction helper _yf_history_with_fallback pour la robustesse
        global_map = {
            '10Y Treasury': '^TNX',
            'VIX Volatility': '^VIX',
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F',
            'US Dollar Index': 'DX-Y.NYB'
        }

        for label, sym in global_map.items():
            try:
                # Utilisation de ton helper robuste
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
            except Exception as e:
                self.logger.debug(f"Indicateur {label} non disponible: {e}")

        if indicators:
            self.cache[cache_key] = (now, indicators)
            
        return indicators

    def get_all_macro_data(self) -> Dict[str, Any]:
        """Agrège les données macro US et Canada pour le dashboard."""
        data_us = self.get_economic_indicators("US")
        data_ca = self.get_economic_indicators("CA")
        return {**data_us, **data_ca}


    def get_commodity_prices(self) -> Dict[str, Any]:
        """Récupère les prix des matières premières en robust manner."""
        try:
            commodities = {
                'Gold': 'GC=F',
                'Silver': 'SI=F',
                'Copper': 'HG=F',
                'Crude Oil': 'CL=F',
                'Natural Gas': 'NG=F',
            }
            prices = {}
            for name, symbol in commodities.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        logger.warning(f"Aucune donnée pour matière première {name} ({symbol})")
                        continue
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change = ((current - previous) / previous * 100) if previous != 0 else 0
                    prices[name] = {'price': current, 'change': change, 'currency': 'USD'}
                except Exception as e:
                    self.logger.warning(f"Erreur matière première {name}: {e}")
                    continue
            return prices
        except Exception as e:
            self.logger.error(f"Erreur get_commodity_prices: {e}")
            return {}

    def get_currency_rates(self) -> Dict[str, Any]:
        """Récupère des taux de change via symbols FX de Yahoo."""
        try:
            currencies = {
                'USD/CAD': 'CAD=X',
                'USD/EUR': 'EUR=X',
                'USD/JPY': 'JPY=X',
            }
            rates = {}
            for pair, symbol in currencies.items():
                try:
                    hist = _yf_history_with_fallback(symbol, period='5d', interval='1d')
                    if not isinstance(hist, pd.DataFrame) or hist.empty or 'Close' not in hist.columns:
                        logger.warning(f"Aucune donnée pour devise {pair} ({symbol})")
                        continue
                    current = float(hist['Close'].iloc[-1])
                    previous = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
                    change = ((current - previous) / previous * 100) if previous != 0 else 0
                    rates[pair] = {'rate': current, 'change': change}
                except Exception as e:
                    self.logger.warning(f"Erreur devise {pair}: {e}")
                    continue
            return rates
        except Exception as e:
            self.logger.error(f"Erreur get_currency_rates: {e}")
            return {}

    def get_fred_series(self, series_id: str, start_date="2000-01-01") -> pd.DataFrame:
        """Télécharge une série FRED (taux, inflation, sentiment, etc.)"""
        try:
            params = {
                "series_id": series_id,
                "api_key": self.FRED_API_KEY,
                "file_type": "json",
                "observation_start": start_date
            }
            r = requests.get(self.fred_base_url, params=params, timeout=10)

            if r.status_code != 200:
                self.logger.warning(f"FRED error {r.status_code} for {series_id}")
                return pd.DataFrame()

            data = r.json().get("observations", [])
            df = pd.DataFrame(data)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["date", "value"]]

        except Exception as e:
            self.logger.error(f"Erreur FRED pour {series_id}: {e}")
            return pd.DataFrame()

    def get_all_macro_data(self) -> Dict[str, Any]:
        """Regroupe toutes les données macro en un seul dict."""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'us_indicators': self.get_economic_indicators('US'),
                'canada_indicators': self.get_economic_indicators('CA'),
                'commodities': self.get_commodity_prices(),
                'currencies': self.get_currency_rates()
            }
            self.logger.info("Données macroéconomiques complètes récupérées")
            return data
        except Exception as e:
            self.logger.error(f"Erreur get_all_macro_data: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}

    def get_market_health(self) -> Dict[str, Any]:
        """
        Récupère la santé du marché via le StockDataExtractor.
        """
        extractor = StockDataExtractor()
        return extractor.evaluate_market_health()

class SectorPerformanceExtractor:
    """
    Compare la performance d’un secteur vs un indice large (ex: S&P500).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_sector_vs_market(self, sector_symbol: str, market_symbol="^GSPC", period="6mo"):
        try:
            sec = yf.Ticker(sector_symbol).history(period=period)
            mkt = yf.Ticker(market_symbol).history(period=period)

            if sec.empty or mkt.empty:
                self.logger.warning(f"Données manquantes pour {sector_symbol} ou {market_symbol}")
                return pd.DataFrame()

            df = pd.DataFrame({
                "sector": sec["Close"].pct_change().cumsum(),
                "market": mkt["Close"].pct_change().cumsum()
            })
            df["relative"] = df["sector"] - df["market"]
            return df

        except Exception as e:
            self.logger.error(f"Erreur sector vs market ({sector_symbol}): {e}")
            return pd.DataFrame()

class FedScenarioAnalyzer:
    """
    Analyse l’impact d’une hausse de taux sur les secteurs clés :
    - Services financiers (IYG)
    - REITs (USRT)
    - Consommation discrétionnaire (XLY)
    """

    def __init__(self, macro: MacroDataExtractor, sector: SectorPerformanceExtractor):
        self.macro = macro
        self.sector = sector
        self.logger = logging.getLogger(__name__)

    def analyze_rate_hike_scenario(self):
        try:
            t2 = self.macro.get_fred_series("DGS2").value.iloc[-1]
            t10 = self.macro.get_fred_series("DGS10").value.iloc[-1]
            spread = t10 - t2

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
            self.logger.error(f"Erreur analyse Fed: {e}")
            return {}



# Fonctions utilitaires
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction utilitaire pour ajouter des indicateurs techniques à un DataFrame
    Args:
        df: DataFrame avec colonnes OHLCV
    Returns:
        DataFrame avec indicateurs ajoutés
    """
    extractor = StockDataExtractor()
    return extractor.calculate_technical_indicators(df)




def get_market_sentiment() -> Dict[str, Any]:
    """
    Fonction utilitaire pour obtenir le sentiment de marché
    """
    extractor = StockDataExtractor()
    return extractor.get_market_indicators()

def get_stock_overview(symbol: str) -> Dict[str, Any]:
    """
    Obtient un aperçu rapide d'une action en utilisant Yahoo Finance
    Compatible avec la fonction demandée dans l'interface
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Formater les valeurs monétaires
        def format_market_cap(value):
            if value is None or value == 'N/A':
                return 'N/A'
            try:
                value = float(value)
                if value >= 1e12:
                    return f"${value/1e12:.2f}T"
                elif value >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    return f"${value/1e6:.2f}M"
                else:
                    return f"${value:,.2f}"
            except:
                return str(value)
        
        overview = {
            'symbol': symbol,
            'name': info.get('longName', info.get('shortName', 'N/A')),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'previous_close': info.get('previousClose', 'N/A'),
            'market_cap': format_market_cap(info.get('marketCap', 'N/A')),
            'pe_ratio': f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A',
            'forward_pe': f"{info.get('forwardPE', 'N/A'):.2f}" if isinstance(info.get('forwardPE'), (int, float)) else 'N/A',
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
        
        logger.info(f"Aperçu récupéré pour {symbol}")

        return overview
        
    except Exception as e:
        logger.error(f"Erreur get_stock_overview pour {symbol}: {e}")
        return {
            'symbol': symbol,
            'name': 'N/A',
            'sector': 'N/A',
            'industry': 'N/A',
            'current_price': 'N/A',
            'previous_close': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A',
            'forward_pe': 'N/A',
            'dividend_yield': 'N/A',
            'beta': 'N/A',
            '52_week_high': 'N/A',
            '52_week_low': 'N/A',
            'volume': 'N/A',
            'avg_volume': 'N/A',
            'currency': 'USD',
            'exchange': 'N/A',
            'country': 'N/A'
        }


def fetch_stock_data(symbol: str, period: str = "1y", include_technicals: bool = True) -> Dict[str, Any]:
    extractor = StockDataExtractor(symbol)
    data = extractor.get_all_data(period=period)
    
    # Si on a des données historiques
    if 'historical' in data and isinstance(data['historical'], pd.DataFrame):
        df = data['historical']
        
        # Calcul des indicateurs techniques si demandé
        if include_technicals:
            df = extractor.calculate_technical_indicators(df)
        
        # --- CRUCIAL POUR LE JSON ---
        # On convertit le DataFrame en dictionnaire pour le cache
        data['history_json'] = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'closes': df['Close'].tolist(),
            'rsi': df['RSI'].tolist() if 'RSI' in df.columns else [],
            'macd': df['MACD'].tolist() if 'MACD' in df.columns else []
        }
        # On peut supprimer le DataFrame original pour alléger le cache JSON
        del data['historical'] 
        
    return data


if __name__ == "__main__":
    # Exemple d'utilisation
    symbol = "AAPL"
    
    logger.info("🔍 Test du module data_extraction")
    logger.info("=" * 50)
    
    # Test de l'aperçu
    logger.info("\n1. Aperçu de l'action:")
    overview = get_stock_overview(symbol)
    logger.info(f"   Nom: {overview.get('name')}")
    logger.info(f"   Prix: {overview.get('current_price')}")
    logger.info(f"   Market Cap: {overview.get('market_cap')}")
    
    # Test de l'extracteur de données
    logger.info("\n2. Extraction des données:")
    extractor = StockDataExtractor(symbol)
    data = extractor.get_historical_data(period="1y")
    logger.info(f"   Données récupérées: {len(data)} jours")
    
    # Test des indicateurs techniques
    if not data.empty:
        logger.info("\n3. Indicateurs techniques:")
        technicals = extractor.calculate_technical_indicators(data)
        logger.info(f"   Colonnes techniques: {len(technicals.columns)}")
        if 'RSI' in technicals.columns:
            logger.info(f"   Dernier RSI: {technicals['RSI'].iloc[-1]:.2f}")
        else:
            logger.info("   RSI: Non calculé")
    
    # Test de l'API X (Twitter)
    logger.info("\n4. Sentiment X (Twitter):")
    sentiment = extractor.get_sentiment_from_x(symbol)
    logger.info(f"   Score de sentiment: {sentiment.get('avg_sentiment', 0):.2f}")
    logger.info(f"   Label: {sentiment.get('sentiment_label', 'N/A')}")
    logger.info(f"   Nombre de tweets: {sentiment.get('tweet_count', 0)}")
    
    # Test de l'API NY Times
    logger.info("\n5. Actualités NY Times:")
    news = extractor.get_news_from_nyt(symbol, days=7)
    logger.info(f"   Nombre d'articles: {news.get('total_articles', 0)}")
    if news.get('articles'):
        logger.info(f"   Premier article: {news['articles'][0].get('headline', 'N/A')[:50]}...")
    
    # Test des données macro
    logger.info("\n6. Données macroéconomiques:")
    macro = MacroDataExtractor()
    us_indicators = macro.get_economic_indicators("US")
    logger.info(f"   Indicateurs US: {len(us_indicators)}")
    sector = SectorPerformanceExtractor()
    fed = FedScenarioAnalyzer(macro, sector)

    diagnostic = fed.analyze_rate_hike_scenario()
    logger.info(diagnostic)

    logger.info("\n✅ Tests terminés!")