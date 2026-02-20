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
'''

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# Configuration unique des API Keys
API_CONFIG = {
    'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'FINANCIAL_MODELING_PREP': os.getenv('FMP_API_KEY'),
    'POLYGON': os.getenv('POLYGON_API_KEY'),
    'TWITTER_X_BEARER': os.getenv('TWITTER_X_BEARER'),
    'NY_TIMES': os.getenv('NY_TIMES_API_KEY')
}

# Vérification que les clés sont chargées
for key_name, key_value in API_CONFIG.items():
    if not key_value:
        logger.warning(f"Clé API manquante: {key_name}")
    elif 'demo' in str(key_value).lower() or len(str(key_value)) < 10:
        logger.warning(f"Clé API suspecte ou démo: {key_name}")
        # Utiliser la valeur 'demo' si la clé est invalide
        API_CONFIG[key_name] = 'demo'

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
        self.cache = {}
        logger.info(f"Cache initialisé dans: {self.cache_dir}")
    
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
            
            # Télécharger depuis Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            
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
    def calculate_technical_indicators(self, data: pd.DataFrame = None) -> pd.DataFrame:
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
            '''
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())

            # On calcule le True Range (numpy gère bien les NaNs de shift())
            true_range_raw = np.maximum.reduce([tr1, tr2, tr3])

            # Crucial : On réinjecte l'index de 'high' (ou 'close') pour l'alignement temporel
            true_range = pd.Series(true_range_raw, index=high.index) 

            # Le calcul du rolling fonctionnera maintenant parfaitement
            indicators['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            '''
            # 5. ATR optimisé (sans concat)
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            true_range = np.maximum.reduce([tr1, tr2, tr3])
            # indicators['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
            true_range_series = pd.Series(true_range, index=df.index)  # ou utilisez l'index existant
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
            # True Range déjà calculé
            tr = indicators['ATR'] * 14  # Approx
        
            # Directional Movement
            up_move = high.diff()
            down_move = low.diff()
        
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
            # Smooth DM
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
        
            # Utiliser shift pour éviter les boucles
            positive_mask = typical_price_mfi > typical_price_mfi.shift(1)
            positive_flow = np.where(positive_mask, money_flow, 0)
            negative_flow = np.where(~positive_mask, money_flow, 0)
        
            positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=14, min_periods=1).sum()
            negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=14, min_periods=1).sum()
        
            mfr = positive_mf / negative_mf.replace(0, np.finfo(float).eps)
            indicators['MFI'] = 100 - (100 / (1 + mfr))
        
            # 16. Ajout de nouveaux indicateurs utiles
            # CCI (Commodity Channel Index)
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
        
            # 17. Nettoyage intelligent des données
            # Supprimer les colonnes avec trop de NaN
            nan_threshold = 0.5  # 50% maximum de NaN
            cols_to_drop = []
            for col in df.columns:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio > nan_threshold:
                    cols_to_drop.append(col)
                    logger.info(f"Colonne {col} supprimée ({nan_ratio:.1%} de NaN)")
        
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        
            # Imputation limitée aux indicateurs techniques uniquement
            indicator_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
            # Forward fill puis backward fill pour les indicateurs
            # df[indicator_cols] = df[indicator_cols].fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
            df[indicator_cols] = df[indicator_cols].ffill(limit=5).bfill(limit=5)
            # Pour les valeurs restantes, remplir avec la moyenne de la colonne
            for col in indicator_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        
            # 18. Supprimer les premières lignes avec trop de NaN pour les indicateurs à long terme
            # Garder au moins 200 périodes si disponible
            keep_from = max(200, int(len(df) * 0.05))  # Au moins 200 lignes ou 5% des données
            initial_length = len(df)
        
            if len(df) > keep_from:
                df = df.iloc[keep_from:]
        
            # 19. Normalisation optionnelle (commentée par défaut)
            # if self.normalize_features:
            #     df = self._normalize_features(df)
        
            # Convertir les types de données pour économiser de la mémoire
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
        
            self.features = df
            logger.info(f"Indicateurs techniques calculés: {df.shape[1]} colonnes, {len(df)} lignes (réduction de {initial_length} à {len(df)})")
        
            # Enregistrer les métriques de qualité
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} valeurs NaN restantes après nettoyage")
        
            return df
        
        except Exception as e:
            logger.error(f"Erreur calculate_technical_indicators: {e}", exc_info=True)
            # Retourner les données originales si possible
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
        
            # Cibles de prix futur
            for days in forecast_days:
                targets[f'Target_Open_{days}d'] = df['Open'].shift(-days)
                targets[f'Target_High_{days}d'] = df['High'].shift(-days)
                targets[f'Target_Low_{days}d'] = df['Low'].shift(-days)
                targets[f'Target_Close_{days}d'] = close.shift(-days)
                targets[f'Target_Volume_{days}d'] = df['Volume'].shift(-days)
                targets[f'Target_Return_{days}d'] = close.shift(-days) / close - 1
        
            # Direction binaire pour tous les horizons
            for days in forecast_days:
                if days >= 1:  # Éviter les horizons négatifs ou nuls
                    targets[f'Target_Direction_{days}d'] = (close.shift(-days) > close).astype(int)
        
            # Volatilité future pour des horizons spécifiques
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
    
    def get_market_indicators(self) -> Dict[str, Any]:
        """
        Récupère les indicateurs de marché globaux
        """
        try:
            # Indices majeurs
            indices = {
                'S&P 500': '^GSPC',
                'NASDAQ': '^IXIC',
                'Dow Jones': '^DJI',
                'VIX': '^VIX',  # Indice de volatilité
                'Russell 2000': '^RUT',
                'TSX Composite': '^GSPTSE'
            }
            
            market_data = {}
            
            for name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change_pct = ((current - previous) / previous * 100) if previous != 0 else 0
                        
                        market_data[name] = {
                            'price': float(current),
                            'change': float(change_pct),
                            'status': '🟢' if change_pct > 0 else '🔴' if change_pct < 0 else '⚪'
                        }
                except Exception as e:
                    logger.warning(f"Erreur pour {name}: {e}")
                    continue
            
            # Calculer le sentiment global
            if market_data:
                positive = sum(1 for data in market_data.values() if data['change'] > 0)
                total = len(market_data)
                sentiment_score = (positive / total * 100) if total > 0 else 50
                
                market_data['overall_sentiment'] = {
                    'score': sentiment_score,
                    'label': 'Bullish' if sentiment_score > 60 else 'Bearish' if sentiment_score < 40 else 'Neutral',
                    'description': 'Haussier' if sentiment_score > 60 else 'Baissier' if sentiment_score < 40 else 'Neutre'
                }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Erreur get_market_indicators: {e}")
            return {}
    
    def get_all_data(self, 
                    symbol: str = None,
                    period: str = "1y") -> Dict[str, Any]:
        """
        Récupère toutes les données disponibles pour un symbole
        """
        if symbol is None:
            symbol = self.symbol
        
        try:
            logger.info(f"Récupération complète des données pour {symbol}")
            
            all_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'overview': get_stock_overview(symbol),
                'historical': self.get_historical_data(symbol, period=period),
                'fundamentals': self.get_fundamental_data(symbol),
                'sentiment_x': self.get_sentiment_from_x(symbol),
                'news_nyt': self.get_news_from_nyt(symbol),
                'market_indicators': self.get_market_indicators()
            }
            
            # Calculer les indicateurs techniques si données historiques disponibles
            if not all_data['historical'].empty:
                all_data['technical'] = self.calculate_technical_indicators(all_data['historical'])
                all_data['targets'] = self.create_target_columns(all_data['historical'])
            
            logger.info(f"Données complètes récupérées pour {symbol}")
            return all_data
            
        except Exception as e:
            logger.error(f"Erreur get_all_data pour {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class MacroDataExtractor:
    """
    Extracteur de données macroéconomiques
    Sources: Statistique Canada, FRED, Yahoo Finance
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
        logger.info("MacroDataExtractor initialisé")
    
    @staticmethod
    def statistique_canada_sdw(method: str, params: Dict = None) -> Optional[Dict]:
        """
        Accède aux données de Statistique Canada Web Data Service
        Args:
            method: Méthode API (voir liste supportée)
            params: Paramètres de la requête
        Returns:
            Données JSON ou None en cas d'erreur
        """
        supported_methods = [
            'getDataFromCubePidCoordAndLatestNPeriods',
            'getDataFromVectorsAndLatestNPeriods',
            'getBulkVectorDataByRange',
            'getCubeMetadata',
            'getSeriesInfoFromVector'
        ]
        
        if method not in supported_methods:
            raise ValueError(f"Méthode non supportée. Méthodes: {supported_methods}")
        
        base_url = "https://www150.statcan.gc.ca/t1/wds/rest"
        url = f"{base_url}/{method}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erreur Statistique Canada API ({method}): {e}")
            return None
    
    def get_economic_indicators(self, country: str = "US") -> Dict[str, Any]:
        """
        Récupère les indicateurs économiques principaux
        Args:
            country: Pays (US, CA, etc.)
        """
        cache_key = f"economic_{country}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Vérifier le cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                logger.info(f"Utilisation cache indicateurs économiques {country}")
                return data
        
        try:
            indicators = {}
            
            if country == "CA":
                # Canada via Statistique Canada
                try:
                    # PIB réel
                    params = {
                        'vectorIds': 'v65201210',  # PIB réel
                        'startDate': '2020-01-01',
                        'endDate': datetime.now().strftime('%Y-%m-%d')
                    }
                    gdp_data = self.statistique_canada_sdw('getDataFromVectorsAndLatestNPeriods', params)
                    
                    if gdp_data and 'object' in gdp_data and gdp_data['object']:
                        latest = gdp_data['object'][0]['vectorDataPoint'][0]
                        indicators['GDP Canada'] = {
                            'value': latest.get('value'),
                            'date': latest.get('refPer'),
                            'growth': latest.get('growthRate'),
                            'unit': 'CAD Billions',
                            'source': 'Statistique Canada'
                        }
                except Exception as e:
                    logger.warning(f"Erreur données Canada: {e}")
            
            # Indicateurs US - utiliser des symboles valides de Yahoo Finance
            us_indicators = {
                '10Y Treasury': '^TNX',
                '30Y Treasury': '^TYX',
                'VIX Volatility': '^VIX',
                'Gold': 'GC=F',
                'Crude Oil': 'CL=F'
            }
            
            for name, symbol in us_indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1mo')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change = ((current - previous) / previous * 100) if previous != 0 else 0
                        
                        indicators[name] = {
                            'value': float(current),
                            'change': float(change),
                            'unit': '%' if 'Treasury' in name else 'points' if name == 'VIX Volatility' else 'USD',
                            'source': 'Yahoo Finance'
                        }
                except Exception as e:
                    logger.warning(f"Erreur indicateur {name}: {e}")
                    continue
            
            # Mettre en cache
            self.cache[cache_key] = (datetime.now(), indicators)
            
            logger.info(f"Indicateurs économiques {country}: {len(indicators)} récupérés")
            return indicators
            
        except Exception as e:
            logger.error(f"Erreur get_economic_indicators: {e}")
            return {}
    
    def get_commodity_prices(self) -> Dict[str, Any]:
        """Récupère les prix des matières premières"""
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
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
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
        """Récupère les taux de change"""
        try:
            currencies = {
                'USD/CAD': 'CAD=X',
                'USD/EUR': 'EUR=X',
                'USD/JPY': 'JPY=X',
                'USD/GBP': 'GBP=X',
                'USD/CHF': 'CHF=X',
                'USD/AUD': 'AUD=X'
            }
            
            rates = {}
            
            for pair, symbol in currencies.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change = ((current - previous) / previous * 100) if previous != 0 else 0
                        
                        rates[pair] = {
                            'rate': float(current),
                            'change': float(change),
                            'trend': '🟢' if 'USD' in pair and change > 0 else '🔴' if 'USD' in pair and change < 0 else '⚪'
                        }
                except Exception as e:
                    logger.warning(f"Erreur devise {pair}: {e}")
                    continue
            
            return rates
            
        except Exception as e:
            logger.error(f"Erreur get_currency_rates: {e}")
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


def fetch_stock_data(symbol: str, 
                    period: str = "1y", 
                    include_technicals: bool = True) -> Dict[str, Any]:
    """
    Fonction simplifiée pour récupérer les données d'une action
    """
    extractor = StockDataExtractor(symbol)
    data = extractor.get_all_data(period=period)
    
    if include_technicals and not data['historical'].empty:
        data['technical'] = extractor.calculate_technical_indicators(data['historical'])
    
    return data


if __name__ == "__main__":
    # Exemple d'utilisation
    symbol = "AAPL"
    
    print("🔍 Test du module data_extraction")
    print("=" * 50)
    
    # Test de l'aperçu
    print("\n1. Aperçu de l'action:")
    overview = get_stock_overview(symbol)
    print(f"   Nom: {overview.get('name')}")
    print(f"   Prix: {overview.get('current_price')}")
    print(f"   Market Cap: {overview.get('market_cap')}")
    
    # Test de l'extracteur de données
    print("\n2. Extraction des données:")
    extractor = StockDataExtractor(symbol)
    data = extractor.get_historical_data(period="1y")
    print(f"   Données récupérées: {len(data)} jours")
    
    # Test des indicateurs techniques
    if not data.empty:
        print("\n3. Indicateurs techniques:")
        technicals = extractor.calculate_technical_indicators(data)
        print(f"   Colonnes techniques: {len(technicals.columns)}")
        print(f"   Dernier RSI: {technicals['RSI'].iloc[-1]:.2f}" if 'RSI' in technicals.columns else "   RSI: Non calculé")
    
    # Test de l'API X (Twitter)
    print("\n4. Sentiment X (Twitter):")
    sentiment = extractor.get_sentiment_from_x(symbol)
    print(f"   Score de sentiment: {sentiment.get('avg_sentiment', 0):.2f}")
    print(f"   Label: {sentiment.get('sentiment_label', 'N/A')}")
    print(f"   Nombre de tweets: {sentiment.get('tweet_count', 0)}")
    
    # Test de l'API NY Times
    print("\n5. Actualités NY Times:")
    news = extractor.get_news_from_nyt(symbol, days=7)
    print(f"   Nombre d'articles: {news.get('total_articles', 0)}")
    if news.get('articles'):
        print(f"   Premier article: {news['articles'][0].get('headline', 'N/A')[:50]}...")
    
    # Test des données macro
    print("\n6. Données macroéconomiques:")
    macro = MacroDataExtractor()
    us_indicators = macro.get_economic_indicators("US")
    print(f"   Indicateurs US: {len(us_indicators)}")
    
    print("\n✅ Tests terminés!")