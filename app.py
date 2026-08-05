from flask import Flask, jsonify, render_template_string, request, send_file
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import requests
import io
from dotenv import load_dotenv
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import feedparser
from bs4 import BeautifulSoup
import random
import logging
import sqlite3
from pathlib import Path
import numpy as np
from collections import deque

# ============================================================
# SUPPRESS WARNINGS
# ============================================================
warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ============================================================
# SIMPLE SENTIMENT ANALYSIS (No TextBlob)
# ============================================================
def get_sentiment_analysis(text):
    """Simple sentiment analysis without TextBlob"""
    if not text:
        return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
    
    # Keyword lists
    positive_words = ['bullish', 'surge', 'soar', 'rocket', 'breakout', 'record', 
                     'outperform', 'strong', 'buy', 'upgrade', 'beating', 'exceed',
                     'up', 'gain', 'positive', 'good', 'great', 'excellent', 'growth', 
                     'profit', 'beat', 'rally', 'strong', 'opportunity', 'win', 'success',
                     'higher', 'rise', 'jump', 'climb', 'advance', 'improve', 'boost']
    
    negative_words = ['bearish', 'crash', 'plunge', 'tumble', 'collapse', 'downgrade',
                     'underperform', 'strong sell', 'miss', 'disaster', 'terrible',
                     'down', 'loss', 'negative', 'bad', 'poor', 'decline', 'drop', 
                     'fall', 'weak', 'risk', 'danger', 'failure', 'problem', 'low',
                     'slip', 'slide', 'struggle', 'worry', 'concern']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
    
    score = (pos_count - neg_count) / total
    confidence = min(100, (total / 5) * 100)
    
    if score > 0.3:
        label = "BULLISH"
    elif score > 0.1:
        label = "POSITIVE"
    elif score < -0.3:
        label = "BEARISH"
    elif score < -0.1:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    return {
        'label': label,
        'score': round(score, 2),
        'confidence': round(confidence, 2)
    }

# ============================================================
# DATABASE FOR PAPER TRADING
# ============================================================
class PaperTradingDB:
    def __init__(self):
        # Use Railway volume if available
        if os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'):
            db_dir = Path(os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'))
            db_dir.mkdir(exist_ok=True)
            self.db_path = str(db_dir / 'paper_trading.db')
        else:
            self.db_path = 'paper_trading.db'
        
        self.init_db()
        self.cache = {}
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                cash REAL DEFAULT 10000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ticker TEXT,
                shares REAL,
                avg_price REAL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ticker TEXT,
                type TEXT,
                shares REAL,
                price REAL,
                total REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                total_value REAL,
                cash REAL,
                holdings_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, username='default'):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, cash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if result:
            user_id, cash = result
        else:
            cursor.execute('INSERT INTO users (username, cash) VALUES (?, ?)', (username, 10000))
            conn.commit()
            user_id = cursor.lastrowid
            cash = 10000
        
        conn.close()
        return user_id, cash
    
    def get_portfolio(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT ticker, shares, avg_price FROM portfolio WHERE user_id = ?', (user_id,))
        results = cursor.fetchall()
        conn.close()
        return [{'ticker': r[0], 'shares': r[1], 'avg_price': r[2]} for r in results]
    
    def get_transactions(self, user_id, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ticker, type, shares, price, total, timestamp 
            FROM transactions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        results = cursor.fetchall()
        conn.close()
        return [{'ticker': r[0], 'type': r[1], 'shares': r[2], 'price': r[3], 'total': r[4], 'timestamp': r[5]} for r in results]
    
    def buy_stock(self, user_id, ticker, shares, price):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_cost = shares * price
        
        cursor.execute('SELECT cash FROM users WHERE user_id = ?', (user_id,))
        cash = cursor.fetchone()[0]
        
        if cash < total_cost:
            conn.close()
            return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${cash:.2f}"
        
        cursor.execute('UPDATE users SET cash = cash - ? WHERE user_id = ?', (total_cost, user_id))
        
        cursor.execute('SELECT shares, avg_price FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
        holding = cursor.fetchone()
        
        if holding:
            existing_shares, avg_price = holding
            new_shares = existing_shares + shares
            new_avg_price = ((existing_shares * avg_price) + (shares * price)) / new_shares
            cursor.execute('''
                UPDATE portfolio 
                SET shares = ?, avg_price = ? 
                WHERE user_id = ? AND ticker = ?
            ''', (new_shares, new_avg_price, user_id, ticker))
        else:
            cursor.execute('''
                INSERT INTO portfolio (user_id, ticker, shares, avg_price)
                VALUES (?, ?, ?, ?)
            ''', (user_id, ticker, shares, price))
        
        cursor.execute('''
            INSERT INTO transactions (user_id, ticker, type, shares, price, total)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'BUY', shares, price, total_cost))
        
        conn.commit()
        conn.close()
        
        if user_id in self.cache:
            del self.cache[user_id]
        
        return True, f"Bought {shares} shares of {ticker} at ${price:.2f}"
    
    def sell_stock(self, user_id, ticker, shares, price):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT shares, avg_price FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
        holding = cursor.fetchone()
        
        if not holding:
            conn.close()
            return False, f"You don't own any {ticker} shares"
        
        existing_shares, avg_price = holding
        
        if existing_shares < shares:
            conn.close()
            return False, f"Insufficient shares. You have {existing_shares}, trying to sell {shares}"
        
        total_value = shares * price
        
        cursor.execute('UPDATE users SET cash = cash + ? WHERE user_id = ?', (total_value, user_id))
        
        new_shares = existing_shares - shares
        if new_shares == 0:
            cursor.execute('DELETE FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
        else:
            cursor.execute('UPDATE portfolio SET shares = ? WHERE user_id = ? AND ticker = ?', (new_shares, user_id, ticker))
        
        cursor.execute('''
            INSERT INTO transactions (user_id, ticker, type, shares, price, total)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'SELL', shares, price, total_value))
        
        conn.commit()
        conn.close()
        
        if user_id in self.cache:
            del self.cache[user_id]
        
        profit_loss = (price - avg_price) * shares
        return True, f"Sold {shares} shares of {ticker} at ${price:.2f} (P/L: ${profit_loss:.2f})"
    
    def get_portfolio_value(self, user_id):
        if user_id in self.cache:
            cache_time, data = self.cache[user_id]
            if (datetime.now() - cache_time).seconds < 5:
                return data
        
        portfolio = self.get_portfolio(user_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash FROM users WHERE user_id = ?', (user_id,))
        cash = cursor.fetchone()[0]
        conn.close()
        
        holdings = []
        total_holdings_value = 0
        total_cost_basis = 0
        
        for item in portfolio:
            try:
                stock = yf.Ticker(item['ticker'])
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    current_value = current_price * item['shares']
                    cost_basis = item['avg_price'] * item['shares']
                    
                    total_holdings_value += current_value
                    total_cost_basis += cost_basis
                    
                    holdings.append({
                        'ticker': item['ticker'],
                        'shares': item['shares'],
                        'avg_price': item['avg_price'],
                        'current_price': current_price,
                        'value': current_value,
                        'cost_basis': cost_basis,
                        'profit_loss': current_value - cost_basis,
                        'profit_loss_pct': ((current_price / item['avg_price']) - 1) * 100
                    })
            except:
                pass
        
        total_value = cash + total_holdings_value
        total_profit_loss = total_holdings_value - total_cost_basis
        
        result = {
            'cash': cash,
            'holdings': holdings,
            'total_holdings_value': total_holdings_value,
            'total_cost_basis': total_cost_basis,
            'total_value': total_value,
            'total_profit_loss': total_profit_loss,
            'total_profit_loss_pct': (total_profit_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0
        }
        
        self.cache[user_id] = (datetime.now(), result)
        return result

# ============================================================
# NEWS SCRAPER
# ============================================================
class EnhancedNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.timeout = 3
        self.scrape_cache = {}
        self.cache_ttl = 180
        self.ai_engine = None
        self.news_history = deque(maxlen=500)
    
    def set_ai_engine(self, ai_engine):
        self.ai_engine = ai_engine
    
    def _safe_scrape(self, url):
        try:
            response = self.session.get(url, timeout=self.timeout, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            })
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None
    
    def scrape_finviz(self, ticker):
        results = {'news': []}
        cache_key = f"finviz_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://finviz.com/quote.ashx?t={ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                news_table = soup.find('table', {'class': 'fullview-news-outer'})
                if news_table:
                    for row in news_table.find_all('tr')[:5]:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            headline = cells[1].text.strip()
                            if headline and len(headline) > 5:
                                news_item = {
                                    'headline': headline,
                                    'time': cells[0].text.strip() if cells[0] else '',
                                    'source': 'Finviz',
                                    'sentiment': self.ai_engine.get_ai_sentiment(headline) if self.ai_engine else {'label': 'NEUTRAL'}
                                }
                                results['news'].append(news_item)
                                self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_google_news(self, ticker):
        results = {'news': []}
        cache_key = f"google_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                if entry.get('title'):
                    text = entry.title
                    if len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': entry.get('published', ''),
                            'source': 'Google News',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def fetch_all_news(self, ticker):
        all_news = {}
        
        sources = {
            'finviz': self.scrape_finviz,
            'google_news': self.scrape_google_news,
        }
        
        for source_name, scraper_func in sources.items():
            try:
                result = scraper_func(ticker)
                if result and result.get('news'):
                    all_news[source_name] = result['news']
            except:
                pass
        
        return all_news
    
    def get_news_feed(self, limit=200):
        return list(self.news_history)[-limit:]

# ============================================================
# AI ANALYSIS ENGINE
# ============================================================
class AIAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 120
    
    def get_ai_sentiment(self, text):
        return get_sentiment_analysis(text)
    
    def get_ai_analysis(self, ticker, company, yahoo_data, sentiment_score, news_data):
        cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.analysis_cache:
            cache_time, data = self.analysis_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        result = self._get_enhanced_fallback_analysis(ticker, company, yahoo_data, sentiment_score)
        if result:
            self.analysis_cache[cache_key] = (datetime.now(), result)
        return result
    
    def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment_score):
        score = 50
        change = yahoo_data['change_1d']
        rsi = yahoo_data['rsi']
        trend = yahoo_data.get('trend', 'NEUTRAL')
        price_vs_sma20 = yahoo_data.get('price_vs_sma20', 'BELOW')
        consecutive_down = yahoo_data.get('consecutive_down_days', 0)
        volume_ratio = yahoo_data.get('volume_ratio', 1)
        
        # Price direction
        if change > 0:
            if change > 3:
                score += 25
            elif change > 1:
                score += 18
            else:
                score += 10
        else:
            if change < -3:
                score -= 15
            elif change < -1:
                score -= 8
            else:
                score -= 2
        
        # Consecutive down days
        if consecutive_down >= 5:
            score -= 15
        elif consecutive_down >= 3:
            score -= 8
        elif consecutive_down >= 2:
            score -= 3
        else:
            score += 10
        
        # Moving averages
        if price_vs_sma20 == 'ABOVE':
            score += 15
        else:
            score -= 8
        
        # Trend
        if 'BULLISH' in trend:
            score += 15
        elif 'BEARISH' in trend:
            score -= 10
        
        # RSI
        if 40 < rsi < 70:
            score += 10
        elif rsi < 30 and change > 0:
            score += 10
        elif rsi > 70 and change < 0:
            score += 5
        
        # Volume
        if volume_ratio > 1.5 and change > 0:
            score += 15
        
        # Sentiment
        if sentiment_score > 0.3:
            score += 10
        elif sentiment_score < -0.3:
            score -= 8
        
        score = max(10, min(90, score))
        
        if score >= 75:
            rec = "STRONG BUY"
            summary = f"📈 {ticker} strong upward momentum"
        elif score >= 62:
            rec = "BUY"
            summary = f"✅ {ticker} positive price action"
        elif score >= 48:
            rec = "WATCH"
            summary = f"👀 {ticker} consolidating"
        elif score >= 35:
            rec = "AVOID"
            summary = f"⚖️ {ticker} weak signals"
        else:
            rec = "SELL"
            summary = f"🚨 {ticker} bearish signals"
        
        return {
            "rec": rec,
            "conf": score,
            "summary": summary,
            "technical_score": score,
            "sentiment_score": max(0, min(100, 50 + sentiment_score * 10)),
            "risk_level": "HIGH" if consecutive_down >= 4 else "MEDIUM" if score < 55 else "LOW",
            "key_factors": ["Price direction", "Technical analysis", "Trend strength"],
            "price_target": f"${yahoo_data['price'] * (1 + (score - 50)/250):.2f}",
            "ai_insight": f"Price moving {change:+.1f}%",
            "momentum_score": score,
            "_source": "Technical Analysis"
        }

# ============================================================
# INITIALIZE COMPONENTS
# ============================================================
ai_engine = AIAnalysisEngine()
news_scraper = EnhancedNewsScraper()
news_scraper.set_ai_engine(ai_engine)
paper_trading = PaperTradingDB()

# ============================================================
# NEWS SOURCES CONFIG
# ============================================================
NEWS_SOURCES = {
    "finviz": {"name": "Finviz", "enabled": True, "icon": "📊", "category": "equities"},
    "google_news": {"name": "Google News", "enabled": True, "icon": "🔍", "category": "aggregator"},
}

CATEGORIES = {
    "equities": {"name": "Equities", "icon": "📈", "count": 1},
    "aggregator": {"name": "Aggregators", "icon": "🔍", "count": 1},
}

# ============================================================
# STOCKS DATABASE
# ============================================================
ALL_STOCKS = {
    # Major Stocks
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc", "sector": "Technology"},
    "AMZN": {"name": "Amazon.com", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corp", "sector": "Technology"},
    "META": {"name": "Meta Platforms", "sector": "Technology"},
    "TSLA": {"name": "Tesla Inc", "sector": "Technology"},
    "INTC": {"name": "Intel Corp", "sector": "Technology"},
    "AMD": {"name": "AMD Corp", "sector": "Technology"},
    "NFLX": {"name": "Netflix Inc", "sector": "Technology"},
    "ADBE": {"name": "Adobe Inc", "sector": "Technology"},
    "CRM": {"name": "Salesforce Inc", "sector": "Technology"},
    "ORCL": {"name": "Oracle Corp", "sector": "Technology"},
    "IBM": {"name": "IBM Corp", "sector": "Technology"},
    "CSCO": {"name": "Cisco Systems", "sector": "Technology"},
    "QCOM": {"name": "Qualcomm Inc", "sector": "Technology"},
    "TXN": {"name": "Texas Instruments", "sector": "Technology"},
    "AVGO": {"name": "Broadcom Inc", "sector": "Technology"},
    "SHOP": {"name": "Shopify Inc", "sector": "Technology"},
    "SNOW": {"name": "Snowflake Inc", "sector": "Technology"},
    "PLTR": {"name": "Palantir Technologies", "sector": "Technology"},
    "UBER": {"name": "Uber Technologies", "sector": "Technology"},
    "PANW": {"name": "Palo Alto Networks", "sector": "Technology"},
    "CRWD": {"name": "CrowdStrike Holdings", "sector": "Technology"},
    
    # Financial
    "JPM": {"name": "JPMorgan Chase", "sector": "Financial"},
    "BAC": {"name": "Bank of America", "sector": "Financial"},
    "WFC": {"name": "Wells Fargo", "sector": "Financial"},
    "C": {"name": "Citigroup Inc", "sector": "Financial"},
    "GS": {"name": "Goldman Sachs", "sector": "Financial"},
    "MS": {"name": "Morgan Stanley", "sector": "Financial"},
    "V": {"name": "Visa Inc", "sector": "Financial"},
    "MA": {"name": "Mastercard Inc", "sector": "Financial"},
    "PYPL": {"name": "PayPal Holdings", "sector": "Financial"},
    "AXP": {"name": "American Express", "sector": "Financial"},
    
    # Healthcare
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc", "sector": "Healthcare"},
    "MRK": {"name": "Merck & Co", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie Inc", "sector": "Healthcare"},
    "TMO": {"name": "Thermo Fisher", "sector": "Healthcare"},
    "ABT": {"name": "Abbott Labs", "sector": "Healthcare"},
    "BMY": {"name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare"},
    "AMGN": {"name": "Amgen Inc", "sector": "Healthcare"},
    "CVS": {"name": "CVS Health Corp", "sector": "Healthcare"},
    
    # Consumer
    "KO": {"name": "Coca-Cola Co", "sector": "Consumer"},
    "PEP": {"name": "PepsiCo Inc", "sector": "Consumer"},
    "COST": {"name": "Costco Wholesale", "sector": "Consumer"},
    "WMT": {"name": "Walmart Inc", "sector": "Consumer"},
    "TGT": {"name": "Target Corp", "sector": "Consumer"},
    "HD": {"name": "Home Depot", "sector": "Consumer"},
    "MCD": {"name": "McDonald's Corp", "sector": "Consumer"},
    "SBUX": {"name": "Starbucks Corp", "sector": "Consumer"},
    "NKE": {"name": "Nike Inc", "sector": "Consumer"},
    "DIS": {"name": "Walt Disney", "sector": "Consumer"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer"},
    
    # Energy
    "MPC": {"name": "Marathon Petroleum Corp", "sector": "Energy"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "EOG": {"name": "EOG Resources", "sector": "Energy"},
    "SLB": {"name": "Schlumberger", "sector": "Energy"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy"},
    "PSX": {"name": "Phillips 66", "sector": "Energy"},
    "VLO": {"name": "Valero Energy", "sector": "Energy"},
    
    # Industrial
    "GE": {"name": "General Electric", "sector": "Industrial"},
    "CAT": {"name": "Caterpillar Inc", "sector": "Industrial"},
    "BA": {"name": "Boeing Co", "sector": "Industrial"},
    "RTX": {"name": "Raytheon Technologies", "sector": "Industrial"},
    "HON": {"name": "Honeywell International", "sector": "Industrial"},
    "DE": {"name": "Deere & Co", "sector": "Industrial"},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrial"},
    
    # Communications
    "T": {"name": "AT&T Inc", "sector": "Communications"},
    "VZ": {"name": "Verizon Communications", "sector": "Communications"},
    "TMUS": {"name": "T-Mobile US", "sector": "Communications"},
    "CMCSA": {"name": "Comcast Corp", "sector": "Communications"},
    "CHTR": {"name": "Charter Communications", "sector": "Communications"},
    
    # Real Estate
    "AMT": {"name": "American Tower", "sector": "Real Estate"},
    "PLD": {"name": "Prologis Inc", "sector": "Real Estate"},
    "SPG": {"name": "Simon Property Group", "sector": "Real Estate"},
    "CCI": {"name": "Crown Castle", "sector": "Real Estate"},
    "EQIX": {"name": "Equinix Inc", "sector": "Real Estate"},
}

# ============================================================
# ENHANCED STOCK ANALYZER
# ============================================================
class EnhancedStockAnalyzer:
    def __init__(self):
        self.stock_cache = {}
        self.cache_ttl = 60
        self.ai_engine = ai_engine
        self.news_scraper = news_scraper
        self.loaded_tickers = set()
        self.filters = {
            'min_price': 0,
            'max_price': 10000,
            'min_rsi': 0,
            'max_rsi': 100,
            'min_change': -100,
            'max_change': 100,
            'min_volume_ratio': 0,
            'sentiment_filter': 'all',
            'trend_filter': 'all'
        }
    
    def set_filters(self, filters):
        self.filters.update(filters)
    
    def apply_filters(self, stock_data):
        if not stock_data:
            return False
        
        price = stock_data.get('price', 0)
        volume_ratio = stock_data.get('volume_ratio', 0)
        rsi = stock_data.get('rsi', 50)
        change = stock_data.get('change_1d', 0)
        sentiment = stock_data.get('sentiment_aggregate', 0)
        
        if price < self.filters.get('min_price', 0) or price > self.filters.get('max_price', 10000):
            return False
        if volume_ratio < self.filters.get('min_volume_ratio', 0):
            return False
        if rsi < self.filters.get('min_rsi', 0) or rsi > self.filters.get('max_rsi', 100):
            return False
        if change < self.filters.get('min_change', -100) or change > self.filters.get('max_change', 100):
            return False
        
        sentiment_filter = self.filters.get('sentiment_filter', 'all')
        if sentiment_filter == 'positive' and sentiment <= 0:
            return False
        elif sentiment_filter == 'negative' and sentiment >= 0:
            return False
        
        trend_filter = self.filters.get('trend_filter', 'all')
        trend = stock_data.get('trend', 'NEUTRAL')
        if trend_filter == 'uptrend' and not ('BULLISH' in trend or 'UPTREND' in trend):
            return False
        elif trend_filter == 'downtrend' and not ('BEARISH' in trend or 'DOWNTREND' in trend):
            return False
        
        return True
    
    def get_stock_data(self, ticker):
        if ticker in self.stock_cache:
            cache_time, data = self.stock_cache[ticker]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2mo", timeout=2)
            if hist.empty:
                return None
            
            info = stock.info
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            loss = loss.replace(0, 0.001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            sma20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            sma50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
            
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            price_vs_sma20 = 'ABOVE' if current_price > sma20 else 'BELOW'
            price_vs_sma50 = 'ABOVE' if current_price > sma50 else 'BELOW'
            
            consecutive_down = 0
            close_prices = hist['Close'].tolist()
            for i in range(len(close_prices)-1, 0, -1):
                if close_prices[i] < close_prices[i-1]:
                    consecutive_down += 1
                else:
                    break
            
            if current_price > sma20 > sma50:
                trend = "STRONG BULLISH"
                trend_strength = "STRONG_BULLISH"
            elif current_price > sma20 and current_price > sma50:
                trend = "BULLISH"
                trend_strength = "BULLISH"
            elif current_price > sma20 and current_price < sma50:
                trend = "CONSOLIDATING (above SMA20)"
                trend_strength = "NEUTRAL_BULLISH"
            elif current_price < sma20 < sma50:
                trend = "STRONG BEARISH"
                trend_strength = "STRONG_BEARISH"
            elif current_price < sma20 and current_price < sma50:
                trend = "BEARISH"
                trend_strength = "BEARISH"
            elif current_price < sma20 and current_price > sma50:
                trend = "CONSOLIDATING (below SMA20)"
                trend_strength = "NEUTRAL_BEARISH"
            else:
                trend = "NEUTRAL"
                trend_strength = "NEUTRAL"
            
            change_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) >= 2 else 0
            
            sector_info = ALL_STOCKS.get(ticker, {})
            result = {
                "ticker": ticker,
                "company": sector_info.get('name', ticker),
                "sector": sector_info.get('sector', 'Unknown'),
                "price": round(float(current_price), 2),
                "change_1d": round(float(change_1d), 2),
                "rsi": round(float(current_rsi), 1),
                "volume_ratio": round(float(volume_ratio), 2),
                "trend": trend,
                "trend_strength": trend_strength,
                "trend_icon": "📈" if "BULLISH" in trend else "📉" if "BEARISH" in trend else "➡️",
                "sma20": round(float(sma20), 2),
                "sma50": round(float(sma50), 2),
                "price_vs_sma20": price_vs_sma20,
                "price_vs_sma50": price_vs_sma50,
                "consecutive_down_days": consecutive_down,
                "historical": {
                    "dates": hist.index.strftime('%Y-%m-%d').tolist()[-30:],
                    "prices": hist['Close'].tolist()[-30:],
                    "volumes": hist['Volume'].tolist()[-30:]
                },
                "pe_ratio": round(float(info.get('trailingPE', 0)), 2) if info.get('trailingPE') else None,
                "target_price": round(float(info.get('targetMeanPrice', 0)), 2) if info.get('targetMeanPrice') else None,
                "current_volume": int(current_volume)
            }
            self.stock_cache[ticker] = (datetime.now(), result)
            return result
        except Exception as e:
            print(f"⚠️ Error getting data for {ticker}: {e}")
            return None
    
    def get_news_sentiment(self, ticker):
        news_data = self.news_scraper.fetch_all_news(ticker)
        sentiment_scores = {'BULLISH': 0, 'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0, 'BEARISH': 0}
        total_news = 0
        sentiment_sum = 0
        news_items = []
        
        for source, items in news_data.items():
            if items:
                for item in items:
                    sentiment = item.get('sentiment', {})
                    label = sentiment.get('label', 'NEUTRAL')
                    if label in sentiment_scores:
                        sentiment_scores[label] += 1
                        total_news += 1
                    score = sentiment.get('score', 0)
                    sentiment_sum += score
                    news_items.append({
                        'headline': item.get('headline', ''),
                        'source': source,
                        'sentiment': label,
                        'score': score
                    })
        
        if total_news > 0:
            avg_sentiment = sentiment_sum / total_news
        else:
            avg_sentiment = 0
        
        sentiment_score = (sentiment_scores.get('BULLISH', 0) * 2 + 
                          sentiment_scores.get('POSITIVE', 0) * 1 -
                          sentiment_scores.get('NEGATIVE', 0) * 1 -
                          sentiment_scores.get('BEARISH', 0) * 2) / (total_news + 1)
        
        return {
            'news_data': news_data,
            'sentiment_scores': sentiment_scores,
            'total_news': total_news,
            'sentiment_score': round(sentiment_score, 2),
            'avg_sentiment': round(avg_sentiment, 2),
            'news_items': news_items
        }

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================
def generate_recommendation_enhanced(data, sentiment_score, ai_analysis):
    score = 50
    change = data['change_1d']
    rsi = data['rsi']
    consecutive_down = data.get('consecutive_down_days', 0)
    price_vs_sma20 = data.get('price_vs_sma20', 'BELOW')
    trend = data.get('trend', 'NEUTRAL')
    volume_ratio = data.get('volume_ratio', 1)
    
    # Price direction
    if change > 0:
        if change > 3:
            score += 28
        elif change > 1:
            score += 20
        else:
            score += 12
    else:
        if change < -3:
            score -= 12
        elif change < -1:
            score -= 6
        else:
            score -= 2
    
    # Consecutive down days
    if consecutive_down >= 5:
        score -= 12
    elif consecutive_down >= 3:
        score -= 6
    elif consecutive_down >= 2:
        score -= 2
    else:
        score += 12
    
    # Moving averages
    if price_vs_sma20 == 'ABOVE':
        score += 18
    else:
        score -= 8
    
    # Trend strength
    if 'STRONG_BULLISH' in trend:
        score += 25
    elif 'BULLISH' in trend:
        score += 16
    elif 'STRONG_BEARISH' in trend:
        score -= 10
    elif 'BEARISH' in trend:
        score -= 5
    
    # RSI
    if 40 < rsi < 70:
        score += 12
    elif 30 < rsi < 40:
        score += 10
    elif rsi > 70:
        if change > 0:
            score -= 4
        else:
            score += 4
    elif rsi < 30:
        if change < 0 and consecutive_down >= 3:
            score -= 4
        else:
            score += 12
    
    # Volume
    if volume_ratio > 1.5:
        if change > 0:
            score += 20
        else:
            score -= 6
    elif volume_ratio > 1.0:
        if change > 0:
            score += 10
        else:
            score += 2
    
    # Sentiment
    if sentiment_score > 0.3:
        if change > 0:
            score += 20
        else:
            score += 10
    elif sentiment_score < -0.3:
        if change < 0:
            score -= 10
        else:
            score -= 4
    
    # AI analysis
    if ai_analysis:
        ai_rec = ai_analysis.get('rec', 'WATCH')
        ai_conf = ai_analysis.get('conf', 50)
        technical_score = ai_analysis.get('technical_score', 50)
        
        if ai_rec in ['STRONG BUY']:
            score += 20
        elif ai_rec in ['BUY']:
            score += 12
        elif ai_rec in ['SELL']:
            score -= 12
        
        score = (score * 0.5) + (technical_score * 0.3) + (ai_conf * 0.2)
    
    score = max(10, min(90, round(score)))
    
    if score >= 75:
        rec = "STRONG BUY"
        summary = f"📈 {data['ticker']} strong upward momentum"
    elif score >= 62:
        rec = "BUY"
        summary = f"✅ {data['ticker']} positive price action"
    elif score >= 48:
        rec = "WATCH"
        summary = f"👀 {data['ticker']} consolidating"
    elif score >= 35:
        rec = "AVOID"
        summary = f"⚖️ {data['ticker']} weak signals"
    else:
        rec = "SELL"
        summary = f"🚨 {data['ticker']} bearish signals"
    
    confidence = min(100, round(score * 0.8 + 10))
    momentum_score = score
    
    return rec, confidence, summary, momentum_score, score

# ============================================================
# MAIN ANALYSIS
# ============================================================
scan_stats = {"technical": 0, "total": 0}
stock_analyzer = EnhancedStockAnalyzer()
loaded_tickers = set()

def get_tickers_by_sector(sector=None):
    if sector and sector != 'all':
        return [t for t, info in ALL_STOCKS.items() if info['sector'] == sector]
    return list(ALL_STOCKS.keys())

def get_next_batch(sector=None, offset=0, batch_size=30, loaded_set=None):
    all_tickers = get_tickers_by_sector(sector)
    
    if loaded_set:
        all_tickers = [t for t in all_tickers if t not in loaded_set]
    
    start = offset
    end = min(offset + batch_size, len(all_tickers))
    if start >= len(all_tickers):
        return []
    return all_tickers[start:end]

def analyze_stock_complete(ticker, use_ai=True):
    global scan_stats
    
    yahoo_data = stock_analyzer.get_stock_data(ticker)
    if not yahoo_data:
        return None
    
    news_analysis = stock_analyzer.get_news_sentiment(ticker)
    news_data = news_analysis['news_data']
    sentiment_score = news_analysis['sentiment_score']
    sentiment_scores = news_analysis['sentiment_scores']
    total_news = news_analysis['total_news']
    news_items = news_analysis.get('news_items', [])
    
    ai_analysis = None
    ai_source = "Technical"
    if use_ai:
        ai_analysis = ai_engine.get_ai_analysis(
            ticker, yahoo_data['company'], yahoo_data, sentiment_score, news_data
        )
        if ai_analysis:
            ai_source = ai_analysis.get('_source', 'AI')
    
    rec, confidence, summary, momentum_score, score = generate_recommendation_enhanced(
        yahoo_data, sentiment_score, ai_analysis
    )
    
    if ai_analysis:
        source = ai_source
    else:
        source = "Technical Fallback"
        scan_stats["technical"] += 1
    
    scan_stats["total"] += 1
    
    rank_score = 0
    if rec in ['STRONG BUY']:
        rank_score += 35
    elif rec in ['BUY']:
        rank_score += 22
    elif rec in ['WATCH']:
        rank_score += 10
    elif rec in ['AVOID']:
        rank_score -= 8
    elif rec in ['SELL']:
        rank_score -= 20
    
    rank_score += confidence * 0.25
    rank_score += (yahoo_data['change_1d'] * 0.6)
    rsi_score = 50 - abs(yahoo_data['rsi'] - 55) * 0.4
    rank_score += rsi_score * 0.2
    if yahoo_data['volume_ratio'] > 1.5:
        rank_score += 8
    rank_score += sentiment_score * 4
    rank_score += momentum_score * 0.2
    
    filtered_data = {
        'price': yahoo_data['price'],
        'volume_ratio': yahoo_data['volume_ratio'],
        'rsi': yahoo_data['rsi'],
        'change_1d': yahoo_data['change_1d'],
        'sentiment_aggregate': sentiment_score,
        'momentum_score': momentum_score,
        'confidence': confidence,
        'trend': yahoo_data['trend']
    }
    
    passes_filters = stock_analyzer.apply_filters(filtered_data)
    
    result = {
        "ticker": ticker,
        "company": yahoo_data['company'],
        "sector": yahoo_data['sector'],
        "price": yahoo_data['price'],
        "change_1d": yahoo_data['change_1d'],
        "rsi": yahoo_data['rsi'],
        "volume_ratio": yahoo_data['volume_ratio'],
        "trend": yahoo_data['trend'],
        "trend_strength": yahoo_data.get('trend_strength', 'NEUTRAL'),
        "trend_icon": yahoo_data['trend_icon'],
        "sma20": yahoo_data['sma20'],
        "sma50": yahoo_data['sma50'],
        "price_vs_sma20": yahoo_data.get('price_vs_sma20', 'BELOW'),
        "price_vs_sma50": yahoo_data.get('price_vs_sma50', 'BELOW'),
        "consecutive_down_days": yahoo_data.get('consecutive_down_days', 0),
        "pe_ratio": yahoo_data.get('pe_ratio'),
        "target_price": yahoo_data.get('target_price'),
        "recommendation": rec,
        "confidence": confidence,
        "summary": summary,
        "source": source,
        "historical": yahoo_data['historical'],
        "news": news_data,
        "news_count": total_news,
        "sentiment_scores": sentiment_scores,
        "sentiment_aggregate": round(sentiment_score, 2),
        "rank_score": round(rank_score, 2),
        "ai_source": ai_source,
        "momentum_score": momentum_score,
        "technical_score": score,
        "passes_filters": passes_filters,
        "news_items": news_items[:5],
        "price_direction": "UP" if yahoo_data['change_1d'] > 0 else "DOWN"
    }
    
    return result

# ============================================================
# FLASK ROUTES
# ============================================================

# ============================================================
# FULL HTML TEMPLATE - The complete version
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>AI Stock Analyzer Pro</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);min-height:100vh;padding:20px;color:#fff}
        .app-container{display:flex;gap:20px;max-width:1900px;margin:0 auto}
        .main-content{flex:1;min-width:0}
        .header{background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:18px 22px;margin-bottom:18px}
        .header-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
        h1{font-size:1.6em;display:flex;align-items:center;gap:8px}
        .gradient{background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .subtitle{color:#888;font-size:12px}
        .btn{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(102,126,234,0.4)}
        .btn-success{background:linear-gradient(135deg,#4CAF50,#2E7D32);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn-success:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(76,175,80,0.4)}
        .btn-danger{background:linear-gradient(135deg,#f44336,#c62828);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn-danger:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(244,67,54,0.4)}
        .btn-sm{padding:4px 12px;font-size:11px}
        .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:8px;margin:12px 0}
        .stat-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 12px;text-align:center}
        .stat-number{font-size:20px;font-weight:bold}
        .stat-label{color:#888;font-size:9px;margin-top:2px}
        .green{color:#4CAF50}.orange{color:#FF9800}.red{color:#f44336}.blue{color:#64B5F6}.gold{color:#FFD700}
        .controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
        .search-box{padding:6px 12px;border:2px solid rgba(255,255,255,0.1);border-radius:6px;background:rgba(255,255,255,0.05);color:#fff;flex:1;min-width:150px;font-size:12px}
        .search-box:focus{outline:none;border-color:#667eea}
        .filter-btn{padding:4px 10px;border:2px solid rgba(255,255,255,0.1);border-radius:5px;background:transparent;color:#aaa;cursor:pointer;font-size:10px}
        .filter-btn:hover{border-color:#667eea;color:#fff}
        .filter-btn.active{background:#667eea;color:#fff;border-color:#667eea}
        .checkbox-label{font-size:11px;color:#aaa;display:flex;align-items:center;gap:4px;cursor:pointer;background:rgba(255,255,255,0.05);padding:3px 10px;border-radius:5px}
        .tabs{display:flex;gap:4px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;flex-wrap:wrap}
        .tab-btn{padding:6px 16px;border-radius:6px 6px 0 0;background:transparent;border:none;color:#888;cursor:pointer;font-size:12px;transition:all 0.3s;font-weight:600}
        .tab-btn:hover{color:#fff;background:rgba(255,255,255,0.05)}
        .tab-btn.active{color:#fff;background:rgba(102,126,234,0.2);border-bottom:2px solid #667eea}
        .card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:10px}
        .stock-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px;transition:all 0.3s;cursor:pointer;min-height:280px;display:flex;flex-direction:column}
        .stock-card:hover{transform:translateY(-3px);border-color:rgba(102,126,234,0.4);box-shadow:0 12px 35px rgba(0,0,0,0.3)}
        .stock-card.pinned{border-color:#FFD700;background:rgba(255,215,0,0.05)}
        .card-rec{padding:2px 10px;border-radius:14px;font-weight:700;font-size:10px;display:inline-block}
        .card-rec.strong-buy{background:rgba(76,175,80,0.25);color:#4CAF50;border:1px solid rgba(76,175,80,0.25)}
        .card-rec.buy{background:rgba(76,175,80,0.15);color:#81C784;border:1px solid rgba(76,175,80,0.15)}
        .card-rec.watch{background:rgba(255,152,0,0.15);color:#FFB74D;border:1px solid rgba(255,152,0,0.15)}
        .card-rec.avoid{background:rgba(244,67,54,0.15);color:#ef9a9a;border:1px solid rgba(244,67,54,0.15)}
        .card-rec.sell{background:rgba(244,67,54,0.25);color:#f44336;border:1px solid rgba(244,67,54,0.25)}
        .pin-btn{background:none;border:none;color:#888;cursor:pointer;font-size:14px;padding:0 4px;transition:all 0.3s}
        .pin-btn:hover{color:#FFD700;transform:scale(1.2)}
        .pin-btn.pinned{color:#FFD700}
        .loading{text-align:center;padding:30px 20px}
        .spinner{border:4px solid rgba(255,255,255,0.1);border-top:4px solid #667eea;border-radius:50%;width:35px;height:35px;animation:spin 1s linear infinite;margin:0 auto 12px}
        @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
        .load-more-container{text-align:center;padding:20px 0}
        .load-more-btn{background:rgba(102,126,234,0.15);border:2px solid rgba(102,126,234,0.3);color:#667eea;padding:10px 30px;border-radius:10px;font-size:14px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .load-more-btn:hover{background:rgba(102,126,234,0.25)}
        .sort-btn{padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);background:transparent;color:#888;cursor:pointer;font-size:9px;transition:all 0.3s}
        .sort-btn:hover{border-color:#667eea;color:#fff}
        .sort-btn.active{background:#667eea;color:#fff;border-color:#667eea}
        .trend-bullish{color:#4CAF50}
        .trend-bearish{color:#f44336}
        .trend-neutral{color:#FFB74D}
        .direction-up{color:#4CAF50}
        .direction-down{color:#f44336}
        .modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:1000;justify-content:center;align-items:center;padding:20px;backdrop-filter:blur(8px)}
        .modal.active{display:flex}
        .modal-content{background:#1a1a2e;border:1px solid rgba(255,255,255,0.1);border-radius:18px;max-width:950px;width:100%;max-height:90vh;overflow-y:auto;padding:22px}
        .modal-close{font-size:26px;cursor:pointer;background:none;border:none;color:#666;padding:0 6px}
        .modal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0}
        .modal-stat{background:rgba(255,255,255,0.05);border-radius:6px;padding:8px}
        .modal-stat .label{color:#666;font-size:8px;text-transform:uppercase}
        .modal-stat .value{font-size:13px;font-weight:bold;margin-top:2px}
        .modal-chart{height:220px;margin:10px 0}
        .paper-trading-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;margin-bottom:15px}
        .paper-trading-panel h3{font-size:14px;margin-bottom:8px;color:#888}
        .paper-stats{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}
        .paper-stat{text-align:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px}
        .paper-stat .value{font-size:18px;font-weight:bold}
        .paper-stat .label{font-size:9px;color:#888}
        .trade-form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:8px;padding:10px;background:rgba(255,255,255,0.03);border-radius:8px}
        .trade-form input{width:80px;padding:4px 8px;border:1px solid rgba(255,255,255,0.1);border-radius:4px;background:rgba(255,255,255,0.05);color:#fff;font-size:11px}
        .trade-form select{padding:4px 8px;border:1px solid rgba(255,255,255,0.1);border-radius:4px;background:rgba(255,255,255,0.05);color:#fff;font-size:11px}
        .portfolio-item{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:12px}
        .profit-positive{color:#4CAF50}
        .profit-negative{color:#f44336}
        .sell-btn{background:rgba(244,67,54,0.15);border:1px solid rgba(244,67,54,0.2);color:#f44336;padding:2px 10px;border-radius:4px;cursor:pointer;font-size:10px}
        .sell-btn:hover{background:rgba(244,67,54,0.25)}
        .transaction-item{font-size:10px;color:#888;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.03)}
        .transaction-buy{color:#4CAF50}
        .transaction-sell{color:#f44336}
        .portfolio-scroll{max-height:200px;overflow-y:auto}
        .transactions-scroll{max-height:150px;overflow-y:auto}
        .sidebar{width:400px;min-width:400px;background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;max-height:calc(100vh - 40px);overflow-y:auto}
        .sidebar.collapsed{width:0;min-width:0;padding:0;border:none;overflow:hidden}
        .filters-section{background:rgba(255,255,255,0.03);border-radius:8px;padding:12px;margin-bottom:12px}
        .filters-section h4{font-size:12px;color:#888;margin-bottom:8px}
        .filter-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
        .filter-row label{font-size:10px;color:#888;min-width:60px}
        .filter-row input{width:70px;padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px}
        .filter-row select{padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px}
        @media(max-width:768px){.app-container{flex-direction:column}.sidebar{width:100%;min-width:unset;max-height:400px}.card-grid{grid-template-columns:1fr}.paper-stats{grid-template-columns:1fr 1fr}}
    </style>
</head>
<body>
<div class="app-container">
    <div class="sidebar" id="sidebar">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:15px">
            <h3 style="font-size:16px">⚙️ Filters & Settings</h3>
            <button onclick="document.getElementById('sidebar').classList.toggle('collapsed')" style="background:none;border:none;color:#666;font-size:20px;cursor:pointer">✕</button>
        </div>
        
        <!-- PAPER TRADING -->
        <div class="paper-trading-panel">
            <h3>💼 Paper Trading</h3>
            <div class="paper-stats">
                <div class="paper-stat"><div class="value gold" id="paperCash">$10,000</div><div class="label">Cash</div></div>
                <div class="paper-stat"><div class="value blue" id="paperValue">$0</div><div class="label">Holdings</div></div>
                <div class="paper-stat"><div class="value" id="paperTotal">$10,000</div><div class="label">Total Value</div></div>
                <div class="paper-stat"><div class="value" id="paperPL">+$0.00</div><div class="label">P&L</div></div>
            </div>
            <div class="trade-form">
                <select id="tradeAction" style="width:70px"><option value="buy">BUY</option><option value="sell">SELL</option></select>
                <input id="tradeTicker" placeholder="Ticker" style="width:70px">
                <input id="tradeShares" placeholder="Shares" type="number" style="width:70px">
                <button class="btn-success btn-sm" onclick="executeTrade()">Execute</button>
                <button class="btn-danger btn-sm" onclick="resetPaperTrading()">Reset</button>
            </div>
            <div id="tradeMessage" style="font-size:11px;margin-top:4px;color:#888"></div>
        </div>
        
        <!-- PORTFOLIO -->
        <div class="paper-trading-panel">
            <h3>📊 Portfolio</h3>
            <div class="portfolio-scroll" id="portfolioList"><div style="color:#666;font-size:11px;padding:8px 0">No holdings</div></div>
        </div>
        
        <!-- TRANSACTIONS -->
        <div class="paper-trading-panel">
            <h3>📜 Recent Transactions</h3>
            <div class="transactions-scroll" id="transactionList"><div style="color:#666;font-size:11px;padding:8px 0">No transactions</div></div>
        </div>
        
        <!-- FILTERS -->
        <div class="filters-section"><h4>💰 Price</h4><div class="filter-row"><label>Min</label><input id="minPrice" placeholder="0" type="number"><label>Max</label><input id="maxPrice" placeholder="10000" type="number"></div></div>
        <div class="filters-section"><h4>📊 RSI</h4><div class="filter-row"><label>Min</label><input id="minRSI" placeholder="0" type="number"><label>Max</label><input id="maxRSI" placeholder="100" type="number"></div></div>
        <div class="filters-section"><h4>📈 Volume Ratio</h4><div class="filter-row"><label>Min</label><input id="minVolumeRatio" placeholder="0" type="number" step="0.1"></div></div>
        <div class="filters-section"><h4>📊 Daily Change</h4><div class="filter-row"><label>Min</label><input id="minChange" placeholder="-100" type="number" step="0.1"><label>Max</label><input id="maxChange" placeholder="100" type="number" step="0.1"></div></div>
        <div class="filters-section"><h4>📈 Trend Filter</h4><div class="filter-row"><select id="trendFilter"><option value="all">All Trends</option><option value="uptrend">Uptrend Only</option><option value="downtrend">Downtrend Only</option></select></div></div>
        <div class="filters-section"><h4>📰 Sentiment</h4><div class="filter-row"><select id="sentimentFilter"><option value="all">All Sentiment</option><option value="positive">Positive Only</option><option value="negative">Negative Only</option></select></div></div>
        
        <div style="display:flex;gap:6px;margin-top:12px">
            <button onclick="applyFilters()" style="flex:1;padding:6px;border:none;border-radius:6px;background:#667eea;color:#fff;cursor:pointer">Apply Filters</button>
            <button onclick="resetFilters()" style="flex:1;padding:6px;border:1px solid rgba(255,255,255,0.1);border-radius:6px;background:transparent;color:#888;cursor:pointer">Reset</button>
        </div>
    </div>
    
    <div class="main-content">
        <button onclick="document.getElementById('sidebar').classList.toggle('collapsed')" style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:8px 16px;color:#aaa;cursor:pointer;margin-bottom:15px">⚙️ Filters</button>
        
        <div class="header">
            <div class="header-top">
                <div><h1>🚀 <span class="gradient">AI Stock Analyzer Pro</span></h1><div class="subtitle">Stocks • News • Paper Trading</div></div>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                    <span id="lastUpdate" style="font-size:9px;color:#666">Never</span>
                    <button onclick="exportData()" style="background:rgba(102,126,234,0.2);border:1px solid rgba(102,126,234,0.3);color:#667eea;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:11px">📥 Export</button>
                    <button class="btn" onclick="refreshData()">🔄 Refresh</button>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-card"><div class="stat-number blue" id="totalStocks">0</div><div class="stat-label">Total</div></div>
                <div class="stat-card"><div class="stat-number green" id="buyCount">0</div><div class="stat-label">Buy</div></div>
                <div class="stat-card"><div class="stat-number orange" id="watchCount">0</div><div class="stat-label">Watch</div></div>
                <div class="stat-card"><div class="stat-number red" id="sellCount">0</div><div class="stat-label">Sell</div></div>
                <div class="stat-card"><div class="stat-number gold" id="pinnedCount">0</div><div class="stat-label">📌 Pinned</div></div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab-btn active" data-tab="all" onclick="switchTab('all')">📊 All Stocks</button>
            <button class="tab-btn" data-tab="pinned" onclick="switchTab('pinned')">📌 Pinned</button>
            <button class="tab-btn" data-tab="gainers" onclick="switchTab('gainers')">📈 Top Gainers</button>
            <button class="tab-btn" data-tab="losers" onclick="switchTab('losers')">📉 Top Losers</button>
            <button class="tab-btn" data-tab="uptrend" onclick="switchTab('uptrend')">📈 Uptrend</button>
            <button class="tab-btn" data-tab="downtrend" onclick="switchTab('downtrend')">📉 Downtrend</button>
        </div>
        
        <div class="controls">
            <input class="search-box" id="searchInput" placeholder="🔍 Search ticker or company..." oninput="filterCards()">
            
            <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center">
                <span style="font-size:9px;color:#666;font-weight:600">Sort:</span>
                <button class="sort-btn active" data-sort="rank_score" onclick="setSort('rank_score')">Rank ▼</button>
                <button class="sort-btn" data-sort="change_1d" onclick="setSort('change_1d')">Change ▼</button>
                <button class="sort-btn" data-sort="momentum_score" onclick="setSort('momentum_score')">Momentum ▼</button>
                <button class="sort-btn" data-sort="rsi" onclick="setSort('rsi')">RSI ▼</button>
                <button class="sort-btn" data-sort="volume_ratio" onclick="setSort('volume_ratio')">Volume ▼</button>
                <button class="sort-btn" data-sort="price" onclick="setSort('price')">Price ▼</button>
            </div>
            
            <div style="display:flex;gap:4px;flex-wrap:wrap">
                <button class="filter-btn active" data-sector="all" onclick="setSector('all')">All</button>
                <button class="filter-btn" data-sector="Technology" onclick="setSector('Technology')">Tech</button>
                <button class="filter-btn" data-sector="Financial" onclick="setSector('Financial')">Fin</button>
                <button class="filter-btn" data-sector="Healthcare" onclick="setSector('Healthcare')">Health</button>
                <button class="filter-btn" data-sector="Consumer" onclick="setSector('Consumer')">Cons</button>
                <button class="filter-btn" data-sector="Energy" onclick="setSector('Energy')">Energy</button>
                <button class="filter-btn" data-sector="Industrial" onclick="setSector('Industrial')">Ind</button>
            </div>
            
            <label class="checkbox-label"><input type="checkbox" id="aiToggle" checked onchange="toggleAI()"> 🧠 AI</label>
        </div>
        
        <div id="loadingState" class="loading"><div class="spinner"></div><div style="color:#888;font-size:14px">📊 Loading stocks with AI analysis...</div></div>
        
        <div id="resultsContent" style="display:none">
            <div id="cardGrid" class="card-grid"></div>
            <div class="load-more-container"><button class="load-more-btn" onclick="loadMoreStocks()">➕ Add More Stocks</button></div>
        </div>
    </div>
</div>

<div class="modal" id="detailModal">
    <div class="modal-content">
        <div class="modal-header" style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div><h2 id="modalTicker" style="font-size:20px"></h2><span id="modalCompany" style="color:#888;font-size:12px"></span></div>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-grid" id="modalStats"></div>
        <div class="modal-chart"><canvas id="modalChart"></canvas></div>
        <div id="modalNews" style="margin-top:10px;max-height:200px;overflow-y:auto"></div>
        <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05)">
            <button class="btn-success btn-sm" onclick="quickTrade('buy')">📈 Buy</button>
            <button class="btn-danger btn-sm" onclick="quickTrade('sell')">📉 Sell</button>
            <span id="quickTradeInfo" style="font-size:11px;color:#888;margin-left:8px"></span>
        </div>
    </div>
</div>

<script>
// ============================================================
// GLOBAL VARIABLES
// ============================================================
let allData = [];
let currentSector = 'all';
let currentTab = 'all';
let useAI = true;
let chart = null;
let pinnedStocks = JSON.parse(localStorage.getItem('pinnedStocks') || '[]');
let currentOffset = 0;
let hasMore = true;
let isLoadingMore = false;
let currentSort = 'rank_score';
let sortDescending = true;
let currentModalTicker = '';

// ============================================================
// PAPER TRADING FUNCTIONS
// ============================================================
async function updatePaperStatus() {
    try {
        const response = await fetch('/api/paper/status');
        const data = await response.json();
        if (data.success) {
            const portfolio = data.portfolio;
            document.getElementById('paperCash').textContent = '$' + portfolio.cash.toFixed(2);
            document.getElementById('paperValue').textContent = '$' + portfolio.total_holdings_value.toFixed(2);
            document.getElementById('paperTotal').textContent = '$' + portfolio.total_value.toFixed(2);
            
            const pl = portfolio.total_profit_loss || 0;
            const plElement = document.getElementById('paperPL');
            plElement.textContent = (pl >= 0 ? '+' : '') + '$' + pl.toFixed(2);
            plElement.style.color = pl >= 0 ? '#4CAF50' : '#f44336';
            
            const list = document.getElementById('portfolioList');
            if (portfolio.holdings && portfolio.holdings.length > 0) {
                list.innerHTML = portfolio.holdings.map(h => `
                    <div class="portfolio-item">
                        <span><strong>${h.ticker}</strong> ${h.shares} shares @ $${h.avg_price.toFixed(2)}</span>
                        <span>$${h.value.toFixed(2)} <span class="${h.profit_loss >= 0 ? 'profit-positive' : 'profit-negative'}">${h.profit_loss >= 0 ? '+' : ''}${h.profit_loss_pct.toFixed(1)}%</span>
                        <button class="sell-btn" onclick="quickSell('${h.ticker}', ${h.shares})">Sell</button></span>
                    </div>
                `).join('');
            } else {
                list.innerHTML = '<div style="color:#666;font-size:11px;padding:8px 0">No holdings</div>';
            }
            
            const transList = document.getElementById('transactionList');
            if (data.transactions && data.transactions.length > 0) {
                transList.innerHTML = data.transactions.slice(0, 10).map(t => `
                    <div class="transaction-item">
                        <span class="transaction-${t.type.toLowerCase()}">${t.type}</span>
                        <strong>${t.ticker}</strong> ${t.shares} @ $${t.price.toFixed(2)} 
                        <span style="float:right">$${t.total.toFixed(2)}</span>
                    </div>
                `).join('');
            } else {
                transList.innerHTML = '<div style="color:#666;font-size:11px;padding:8px 0">No transactions</div>';
            }
        }
    } catch(e) { console.error('Paper status error:', e); }
}

async function executeTrade() {
    const action = document.getElementById('tradeAction').value;
    const ticker = document.getElementById('tradeTicker').value.toUpperCase().trim();
    const shares = parseFloat(document.getElementById('tradeShares').value);
    
    if (!ticker || !shares || shares <= 0) {
        document.getElementById('tradeMessage').textContent = '⚠️ Invalid input';
        document.getElementById('tradeMessage').style.color = '#f44336';
        return;
    }
    
    const endpoint = action === 'buy' ? '/api/paper/buy' : '/api/paper/sell';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, shares })
        });
        const data = await response.json();
        
        document.getElementById('tradeMessage').textContent = data.message;
        document.getElementById('tradeMessage').style.color = data.success ? '#4CAF50' : '#f44336';
        
        if (data.success) {
            document.getElementById('tradeTicker').value = '';
            document.getElementById('tradeShares').value = '';
            updatePaperStatus();
        }
    } catch(e) {
        document.getElementById('tradeMessage').textContent = '⚠️ Error executing trade';
        document.getElementById('tradeMessage').style.color = '#f44336';
    }
}

async function quickTrade(action) {
    if (!currentModalTicker) return;
    const shares = prompt(`Enter number of shares to ${action} for ${currentModalTicker}:`, '1');
    if (!shares || parseFloat(shares) <= 0) return;
    
    const endpoint = action === 'buy' ? '/api/paper/buy' : '/api/paper/sell';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: currentModalTicker, shares: parseFloat(shares) })
        });
        const data = await response.json();
        document.getElementById('quickTradeInfo').textContent = data.message;
        document.getElementById('quickTradeInfo').style.color = data.success ? '#4CAF50' : '#f44336';
        if (data.success) updatePaperStatus();
    } catch(e) {
        document.getElementById('quickTradeInfo').textContent = '⚠️ Error';
        document.getElementById('quickTradeInfo').style.color = '#f44336';
    }
}

async function quickSell(ticker, maxShares) {
    const shares = prompt(`Enter shares to sell for ${ticker} (max ${maxShares}):`, maxShares);
    if (!shares || parseFloat(shares) <= 0) return;
    
    try {
        const response = await fetch('/api/paper/sell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, shares: parseFloat(shares) })
        });
        const data = await response.json();
        if (data.success) updatePaperStatus();
        else alert(data.message);
    } catch(e) { alert('Error selling'); }
}

async function resetPaperTrading() {
    if (!confirm('Reset paper trading account to $10,000?')) return;
    try {
        const response = await fetch('/api/paper/reset', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            updatePaperStatus();
            document.getElementById('tradeMessage').textContent = '✅ Account reset';
            document.getElementById('tradeMessage').style.color = '#4CAF50';
        }
    } catch(e) { alert('Error resetting'); }
}

// ============================================================
// CORE FUNCTIONS
// ============================================================
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    renderCards();
}

function togglePin(ticker, event) {
    event.stopPropagation();
    const index = pinnedStocks.indexOf(ticker);
    if (index > -1) pinnedStocks.splice(index, 1);
    else pinnedStocks.push(ticker);
    localStorage.setItem('pinnedStocks', JSON.stringify(pinnedStocks));
    renderCards();
    updateStats();
}

function isPinned(ticker) { return pinnedStocks.includes(ticker); }

function setSort(sort) {
    if (currentSort === sort) {
        sortDescending = !sortDescending;
    } else {
        currentSort = sort;
        sortDescending = true;
    }
    document.querySelectorAll('.sort-btn').forEach(b => {
        b.classList.remove('active');
        if (b.dataset.sort === sort) {
            b.classList.add('active');
            b.textContent = b.textContent.replace(/[▼▲]/g, '') + (sortDescending ? ' ▼' : ' ▲');
        }
    });
    renderCards();
}

function filterCards() { renderCards(); }

function toggleAI() {
    useAI = document.getElementById('aiToggle').checked;
    currentOffset = 0;
    allData = [];
    refreshData();
}

function setSector(sector) {
    currentSector = sector;
    document.querySelectorAll('[data-sector]').forEach(b => b.classList.toggle('active', b.dataset.sector === sector));
    currentOffset = 0;
    allData = [];
    refreshData();
}

function getFilters() {
    return {
        min_price: parseFloat(document.getElementById('minPrice').value) || 0,
        max_price: parseFloat(document.getElementById('maxPrice').value) || 10000,
        min_rsi: parseFloat(document.getElementById('minRSI').value) || 0,
        max_rsi: parseFloat(document.getElementById('maxRSI').value) || 100,
        min_volume_ratio: parseFloat(document.getElementById('minVolumeRatio').value) || 0,
        min_change: parseFloat(document.getElementById('minChange').value) || -100,
        max_change: parseFloat(document.getElementById('maxChange').value) || 100,
        sentiment_filter: document.getElementById('sentimentFilter').value,
        trend_filter: document.getElementById('trendFilter').value
    };
}

function applyFilters() {
    currentOffset = 0;
    allData = [];
    refreshData();
}

function resetFilters() {
    document.getElementById('minPrice').value = '';
    document.getElementById('maxPrice').value = '';
    document.getElementById('minRSI').value = '';
    document.getElementById('maxRSI').value = '';
    document.getElementById('minVolumeRatio').value = '';
    document.getElementById('minChange').value = '';
    document.getElementById('maxChange').value = '';
    document.getElementById('sentimentFilter').value = 'all';
    document.getElementById('trendFilter').value = 'all';
    currentOffset = 0;
    allData = [];
    refreshData();
}

// ============================================================
// DATA REFRESH
// ============================================================
async function refreshData() {
    const btn = document.querySelector('.btn');
    btn.disabled = true;
    btn.textContent = '⏳...';
    
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultsContent').style.display = 'none';
    
    try {
        const payload = {
            sector: currentSector !== 'all' ? currentSector : null,
            limit: 30,
            offset: 0,
            load_more: false,
            use_ai: useAI,
            pinned: pinnedStocks,
            filters: getFilters()
        };
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const res = await response.json();
        
        if (res.success) {
            allData = res.results;
            hasMore = res.has_more || false;
            currentOffset = 30;
            
            document.getElementById('lastUpdate').textContent = 'Updated: ' + res.last_update;
            renderCards();
            updateStats();
            updatePaperStatus();
        }
    } catch(err) {
        console.error(err);
    } finally {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';
        btn.disabled = false;
        btn.textContent = '🔄 Refresh';
    }
}

async function loadMoreStocks() {
    if (isLoadingMore || !hasMore) return;
    isLoadingMore = true;
    const btn = document.querySelector('.load-more-btn');
    btn.textContent = '⏳ Loading...';
    btn.disabled = true;
    
    try {
        const payload = {
            sector: currentSector !== 'all' ? currentSector : null,
            limit: 30,
            offset: currentOffset,
            load_more: true,
            use_ai: useAI,
            pinned: pinnedStocks,
            filters: getFilters()
        };
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const res = await response.json();
        
        if (res.success && res.results.length > 0) {
            const existingTickers = new Set(allData.map(d => d.ticker));
            const newResults = res.results.filter(d => !existingTickers.has(d.ticker));
            allData = [...allData, ...newResults];
            hasMore = res.has_more || false;
            currentOffset += 30;
            renderCards();
            updateStats();
            updatePaperStatus();
        }
    } catch(err) {
        console.error(err);
    } finally {
        isLoadingMore = false;
        btn.textContent = '➕ Add More Stocks';
        btn.disabled = false;
    }
}

// ============================================================
// RENDER FUNCTIONS
// ============================================================
function renderCards() {
    const grid = document.getElementById('cardGrid');
    
    let filtered = allData.filter(item => {
        if (currentSector !== 'all' && item.sector !== currentSector) return false;
        const search = document.getElementById('searchInput').value.toLowerCase();
        if (search && !item.ticker.toLowerCase().includes(search) && !item.company.toLowerCase().includes(search)) return false;
        return true;
    });
    
    if (currentTab === 'pinned') {
        filtered = filtered.filter(item => isPinned(item.ticker));
    } else if (currentTab === 'gainers') {
        filtered = filtered.filter(item => item.change_1d > 2).sort((a, b) => b.change_1d - a.change_1d);
    } else if (currentTab === 'losers') {
        filtered = filtered.filter(item => item.change_1d < -2).sort((a, b) => a.change_1d - b.change_1d);
    } else if (currentTab === 'uptrend') {
        filtered = filtered.filter(item => item.trend && (item.trend.includes('BULLISH') || item.trend.includes('UPTREND')));
    } else if (currentTab === 'downtrend') {
        filtered = filtered.filter(item => item.trend && (item.trend.includes('BEARISH') || item.trend.includes('DOWNTREND')));
    }
    
    if (currentSort && currentTab !== 'gainers' && currentTab !== 'losers') {
        filtered.sort((a, b) => {
            let va = a[currentSort] ?? 0;
            let vb = b[currentSort] ?? 0;
            if (typeof va === 'string') {
                return sortDescending ? va.localeCompare(vb) : vb.localeCompare(va);
            }
            return sortDescending ? vb - va : va - vb;
        });
    }
    
    grid.innerHTML = '';
    
    filtered.forEach((item) => {
        const card = document.createElement('div');
        let cardClass = 'stock-card';
        if (isPinned(item.ticker)) cardClass += ' pinned';
        card.className = cardClass;
        card.onclick = () => openModal(item);
        
        const recClass = (item.recommendation || 'WATCH').toLowerCase().replace(' ', '-');
        const pinned = isPinned(item.ticker);
        const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
        const directionClass = direction === 'UP' ? 'direction-up' : 'direction-down';
        const trendClass = item.trend && item.trend.includes('BULLISH') ? 'trend-bullish' : 
                          item.trend && item.trend.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
        const momentum = item.momentum_score || 50;
        
        let newsCount = 0;
        if (item.news) {
            for (const s in item.news) {
                if (item.news[s]) newsCount += item.news[s].length;
            }
        }
        
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-size:17px;font-weight:700">${item.ticker} ${pinned ? '📌' : ''}</div>
                    <div style="font-size:11px;color:#888">${item.company}</div>
                </div>
                <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">
                    <button class="pin-btn ${pinned ? 'pinned' : ''}" onclick="togglePin('${item.ticker}', event)">📌</button>
                    <span class="card-rec ${recClass}">${item.recommendation || 'WATCH'}</span>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="font-size:20px;font-weight:700">$${item.price?.toFixed(2) || 'N/A'}</span>
                <span style="font-size:14px;font-weight:600;color:${item.change_1d >= 0 ? '#4CAF50' : '#f44336'}">
                    ${item.change_1d?.toFixed(1) || '0.0'}% <span class="${directionClass}">${direction === 'UP' ? '▲' : '▼'}</span>
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;margin:4px 0;font-size:11px;color:#aaa">
                <div>RSI: ${item.rsi || 'N/A'}</div>
                <div>Vol: ${item.volume_ratio?.toFixed(1) || 'N/A'}x</div>
                <div class="${trendClass}">${item.trend_icon || '➡️'} ${item.trend || 'NEUTRAL'}</div>
                <div>📰 ${newsCount}</div>
            </div>
            ${item.consecutive_down_days >= 2 ? `<div style="font-size:10px;color:#f44336;font-weight:bold">⚠️ ${item.consecutive_down_days} consecutive down days</div>` : ''}
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin:2px 0">
                <span>Momentum: ${momentum}%</span>
                <span>Conf: ${item.confidence || 0}%</span>
                <span>Score: ${Math.round(item.rank_score || 0)}</span>
            </div>
            <div style="font-size:11px;color:#bbb;flex:1;margin:4px 0">${item.summary || 'No analysis'}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#666;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.05)">
                <span>${item.ai_source || 'Technical'}</span>
                <span>${item.sector || 'Unknown'}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

function updateStats() {
    document.getElementById('totalStocks').textContent = allData.length;
    const buys = allData.filter(d => d.recommendation && d.recommendation.includes('BUY')).length;
    const watches = allData.filter(d => d.recommendation === 'WATCH').length;
    const sells = allData.filter(d => d.recommendation === 'SELL' || d.recommendation === 'AVOID').length;
    document.getElementById('buyCount').textContent = buys;
    document.getElementById('watchCount').textContent = watches;
    document.getElementById('sellCount').textContent = sells;
    document.getElementById('pinnedCount').textContent = pinnedStocks.length;
}

// ============================================================
// MODAL FUNCTIONS
// ============================================================
function openModal(item) {
    currentModalTicker = item.ticker;
    document.getElementById('modalTicker').textContent = item.ticker;
    document.getElementById('modalCompany').textContent = item.company + ' • ' + item.sector;
    
    const stats = [
        { label: 'Price', value: '$' + (item.price?.toFixed(2) || 'N/A') },
        { label: 'Change', value: (item.change_1d?.toFixed(1) || '0.0') + '%', class: item.change_1d >= 0 ? 'trend-bullish' : 'trend-bearish' },
        { label: 'RSI', value: item.rsi || 'N/A' },
        { label: 'Volume Ratio', value: item.volume_ratio?.toFixed(2) || 'N/A' },
        { label: 'Momentum', value: (item.momentum_score || 50) + '%' },
        { label: 'P/E', value: item.pe_ratio || 'N/A' },
        { label: 'SMA20', value: '$' + (item.sma20?.toFixed(2) || 'N/A') },
        { label: 'SMA50', value: '$' + (item.sma50?.toFixed(2) || 'N/A') },
        { label: 'Recommendation', value: item.recommendation || 'WATCH' },
        { label: 'Confidence', value: item.confidence + '%' },
        { label: 'Rank Score', value: Math.round(item.rank_score || 0) }
    ];
    
    const modalStats = document.getElementById('modalStats');
    modalStats.innerHTML = '';
    stats.forEach(s => {
        const div = document.createElement('div');
        div.className = 'modal-stat';
        div.innerHTML = `<div class="label">${s.label}</div><div class="value ${s.class || ''}">${s.value}</div>`;
        modalStats.appendChild(div);
    });
    
    if (chart) chart.destroy();
    const ctx = document.getElementById('modalChart').getContext('2d');
    const hist = item.historical;
    if (hist && hist.dates && hist.dates.length > 0) {
        chart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: hist.dates, 
                datasets: [{ label: 'Price', data: hist.prices, borderColor: '#667eea', backgroundColor: 'rgba(102,126,234,0.1)', fill: true, tension: 0.4, pointRadius: 0 }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, labels: { color: '#888', font: { size: 9 } } } } }
        });
    }
    
    const modalNews = document.getElementById('modalNews');
    modalNews.innerHTML = '';
    if (item.news_items && item.news_items.length > 0) {
        modalNews.innerHTML = '<div style="font-weight:bold;color:#667eea;margin-bottom:6px">📰 Recent News</div>';
        item.news_items.slice(0, 5).forEach(n => {
            const div = document.createElement('div');
            div.style.cssText = 'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:11px;color:#aaa';
            const emoji = n.sentiment === 'BULLISH' ? '🟢' : n.sentiment === 'BEARISH' ? '🔴' : '🟡';
            div.innerHTML = `${emoji} [${n.source}] ${n.headline?.substring(0, 150) || ''}...`;
            modalNews.appendChild(div);
        });
    } else {
        modalNews.innerHTML = '<div style="color:#666;padding:10px">No news available</div>';
    }
    
    document.getElementById('quickTradeInfo').textContent = '';
    document.getElementById('detailModal').classList.add('active');
}

function closeModal() {
    document.getElementById('detailModal').classList.remove('active');
    if (chart) { chart.destroy(); chart = null; }
}

document.getElementById('detailModal').addEventListener('click', e => { if (e.target === e.currentTarget) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ============================================================
// EXPORT DATA
// ============================================================
async function exportData() {
    if (allData.length === 0) {
        alert('No data to export.');
        return;
    }
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: allData })
        });
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `stock_analysis_${new Date().toISOString().slice(0,10)}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    } catch(err) {
        alert('Error exporting: ' + err.message);
    }
}

// ============================================================
// INITIALIZATION
// ============================================================
refreshData();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global scan_stats, loaded_tickers
    scan_stats = {"technical": 0, "total": 0}
    
    data = request.get_json() or {}
    tickers = data.get('tickers', [])
    use_ai = data.get('use_ai', True)
    sector = data.get('sector', None)
    limit = data.get('limit', 30)
    offset = data.get('offset', 0)
    load_more = data.get('load_more', False)
    pinned = data.get('pinned', [])
    
    filters = data.get('filters', {})
    if filters:
        stock_analyzer.set_filters(filters)
    
    if not tickers:
        if load_more:
            new_tickers = get_next_batch(sector, offset, limit, loaded_tickers)
            if not new_tickers:
                return jsonify({'success': True, 'results': [], 'total': 0, 'has_more': False, 'stats': scan_stats})
            all_tickers = list(loaded_tickers) + new_tickers
        else:
            loaded_tickers.clear()
            batch = get_next_batch(sector, 0, limit, None)
            loaded_tickers.update(batch)
            all_tickers = batch
        tickers = all_tickers
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_stock_complete, t, use_ai): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=15)
                if result and result.get('passes_filters', True):
                    results.append(result)
            except Exception as e:
                print(f"⚠️ Error for {ticker}: {e}")
    
    results.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
    elapsed = round(time.time() - start_time, 2)
    
    if pinned:
        pinned_results = [r for r in results if r['ticker'] in pinned]
        results = pinned_results + [r for r in results if r['ticker'] not in pinned]
    
    all_available = get_tickers_by_sector(sector)
    has_more = len(loaded_tickers) < len(all_available)
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results),
        'has_more': has_more,
        'loaded_count': len(loaded_tickers),
        'total_available': len(all_available),
        'stats': scan_stats,
        'elapsed': elapsed,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filters': stock_analyzer.filters
    })

@app.route('/api/export', methods=['POST'])
def export_data():
    data = request.get_json() or {}
    results = data.get('results', [])
    if not results:
        return jsonify({'success': False, 'error': 'No data'}), 400
    
    df = pd.DataFrame(results)
    columns = ['ticker', 'company', 'sector', 'price', 'change_1d', 'rsi', 'volume_ratio', 
               'trend', 'recommendation', 'confidence', 'rank_score', 'news_count', 
               'sentiment_aggregate', 'ai_source', 'momentum_score']
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'stock_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route('/api/paper/status', methods=['GET'])
def paper_status():
    user_id, cash = paper_trading.get_or_create_user()
    portfolio_value = paper_trading.get_portfolio_value(user_id)
    transactions = paper_trading.get_transactions(user_id)
    return jsonify({
        'success': True,
        'user_id': user_id,
        'cash': cash,
        'portfolio': portfolio_value,
        'transactions': transactions
    })

@app.route('/api/paper/buy', methods=['POST'])
def paper_buy():
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    shares = float(data.get('shares', 0))
    
    if not ticker or shares <= 0:
        return jsonify({'success': False, 'error': 'Invalid input'}), 400
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return jsonify({'success': False, 'error': f'Could not get price for {ticker}'}), 400
        price = hist['Close'].iloc[-1]
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error getting price: {str(e)}'}), 400
    
    user_id, _ = paper_trading.get_or_create_user()
    success, message = paper_trading.buy_stock(user_id, ticker, shares, price)
    portfolio = paper_trading.get_portfolio_value(user_id)
    
    return jsonify({
        'success': success,
        'message': message,
        'ticker': ticker,
        'shares': shares,
        'price': price,
        'total': shares * price,
        'portfolio': portfolio
    })

@app.route('/api/paper/sell', methods=['POST'])
def paper_sell():
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    shares = float(data.get('shares', 0))
    
    if not ticker or shares <= 0:
        return jsonify({'success': False, 'error': 'Invalid input'}), 400
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return jsonify({'success': False, 'error': f'Could not get price for {ticker}'}), 400
        price = hist['Close'].iloc[-1]
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error getting price: {str(e)}'}), 400
    
    user_id, _ = paper_trading.get_or_create_user()
    success, message = paper_trading.sell_stock(user_id, ticker, shares, price)
    portfolio = paper_trading.get_portfolio_value(user_id)
    
    return jsonify({
        'success': success,
        'message': message,
        'ticker': ticker,
        'shares': shares,
        'price': price,
        'total': shares * price,
        'portfolio': portfolio
    })

@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    user_id, _ = paper_trading.get_or_create_user()
    
    conn = sqlite3.connect(paper_trading.db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM portfolio WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
    cursor.execute('UPDATE users SET cash = 10000 WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    if user_id in paper_trading.cache:
        del paper_trading.cache[user_id]
    
    return jsonify({'success': True, 'message': 'Account reset to $10,000'})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'total_stocks': len(ALL_STOCKS),
        'filters': stock_analyzer.filters
    })

# ============================================================
# RUN THE APP
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*80)
    print("🚀 AI Stock Analyzer Pro")
    print("="*80)
    print(f"📈 Total Stocks: {len(ALL_STOCKS)}")
    print(f"📰 News Sources: {len(NEWS_SOURCES)}")
    print("="*80)
    print(f"🌐 Starting server on port {port}")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
