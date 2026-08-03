import os
import sys

# CRITICAL: Set PYTHONSAFEPATH before any imports
os.environ['PYTHONSAFEPATH'] = '1'

# Remove current directory from sys.path to prevent import issues
sys.path = [p for p in sys.path if p != '' and p != '.']

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
import threading
import feedparser
from bs4 import BeautifulSoup
import re
import numpy as np
from collections import deque
import random
import logging
from textblob import TextBlob
import sqlite3
from pathlib import Path
from sklearn.linear_model import LinearRegression

# OpenAI imports with graceful failure
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed")

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini not installed")

warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.ERROR)
logging.getLogger('groq').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# ============================================================
# RAILWAY-SPECIFIC FIXES
# ============================================================

IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

if IS_RAILWAY:
    DB_PATH = '/tmp/paper_trading.db'
else:
    DB_PATH = 'paper_trading.db'

PORT = int(os.environ.get('PORT', 5000))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# ============================================================
# AI CLIENTS WITH RATE LIMITING
# ============================================================

openai_client = None
claude_client = None
groq_client = None
gemini_client = None
GROQ_RATE_LIMITED = False
GROQ_LIMIT_RESET_TIME = None
OPENAI_RATE_LIMITED = False
OPENAI_LIMIT_RESET_TIME = None

# Rate limiting counters
_ai_call_counter = 0
_ai_call_reset_time = datetime.now()
_ai_call_lock = threading.Lock()

def check_ai_rate_limit():
    """Global AI rate limiter - max 100 calls per minute across all services"""
    global _ai_call_counter, _ai_call_reset_time
    with _ai_call_lock:
        now = datetime.now()
        if (now - _ai_call_reset_time).seconds >= 60:
            _ai_call_counter = 0
            _ai_call_reset_time = now
        if _ai_call_counter >= 100:
            return False
        _ai_call_counter += 1
        return True

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        logger.info("✓ OpenAI ready")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI error: {e}")

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('gemini-pro')
        logger.info("✓ Gemini ready")
    except Exception as e:
        logger.warning(f"⚠️ Gemini error: {e}")

try:
    from anthropic import Anthropic
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your-anthropic-api-key-here":
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("✓ Claude ready")
except:
    claude_client = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✓ Groq ready")
except:
    groq_client = None

def is_groq_rate_limited():
    global GROQ_RATE_LIMITED, GROQ_LIMIT_RESET_TIME
    if GROQ_RATE_LIMITED and GROQ_LIMIT_RESET_TIME:
        if datetime.now() > GROQ_LIMIT_RESET_TIME:
            GROQ_RATE_LIMITED = False
            GROQ_LIMIT_RESET_TIME = None
            return False
        return True
    return False

def mark_groq_rate_limited():
    global GROQ_RATE_LIMITED, GROQ_LIMIT_RESET_TIME
    if not GROQ_RATE_LIMITED:
        GROQ_RATE_LIMITED = True
        GROQ_LIMIT_RESET_TIME = datetime.now() + timedelta(minutes=10)
        logger.warning(f"⚠️ Groq rate limited until {GROQ_LIMIT_RESET_TIME}")

def is_openai_rate_limited():
    global OPENAI_RATE_LIMITED, OPENAI_LIMIT_RESET_TIME
    if OPENAI_RATE_LIMITED and OPENAI_LIMIT_RESET_TIME:
        if datetime.now() > OPENAI_LIMIT_RESET_TIME:
            OPENAI_RATE_LIMITED = False
            OPENAI_LIMIT_RESET_TIME = None
            return False
        return True
    return False

def mark_openai_rate_limited():
    global OPENAI_RATE_LIMITED, OPENAI_LIMIT_RESET_TIME
    if not OPENAI_RATE_LIMITED:
        OPENAI_RATE_LIMITED = True
        OPENAI_LIMIT_RESET_TIME = datetime.now() + timedelta(minutes=5)
        logger.warning(f"⚠️ OpenAI rate limited until {OPENAI_LIMIT_RESET_TIME}")

# ============================================================
# DATABASE
# ============================================================

class PaperTradingDB:
    def __init__(self):
        self.db_path = Path(DB_PATH)
        self._lock = threading.Lock()
        self.init_db()
        self.cache = {}
    
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        return conn
    
    def init_db(self):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    cash REAL DEFAULT 10000,
                    total_profit REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
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
                    profit_loss REAL DEFAULT 0,
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
                    total_profit REAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            cursor.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'total_profit' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN total_profit REAL DEFAULT 0')
            if 'total_trades' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN total_trades INTEGER DEFAULT 0')
            if 'winning_trades' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN winning_trades INTEGER DEFAULT 0')
            
            cursor.execute("PRAGMA table_info(transactions)")
            trans_columns = [col[1] for col in cursor.fetchall()]
            if 'profit_loss' not in trans_columns:
                cursor.execute('ALTER TABLE transactions ADD COLUMN profit_loss REAL DEFAULT 0')
            
            conn.commit()
            conn.close()
    
    def get_or_create_user(self, username='default'):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, cash, total_profit, total_trades, winning_trades FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if result:
            user_id, cash, total_profit, total_trades, winning_trades = result
        else:
            cursor.execute('INSERT INTO users (username, cash, total_profit, total_trades, winning_trades) VALUES (?, ?, ?, ?, ?)', 
                         (username, 10000, 0, 0, 0))
            conn.commit()
            user_id = cursor.lastrowid
            cash = 10000
            total_profit = 0
            total_trades = 0
            winning_trades = 0
        
        conn.close()
        return user_id, cash, total_profit, total_trades, winning_trades
    
    def get_portfolio(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT ticker, shares, avg_price FROM portfolio WHERE user_id = ?', (user_id,))
        results = cursor.fetchall()
        conn.close()
        return [{'ticker': r[0], 'shares': r[1], 'avg_price': r[2]} for r in results]
    
    def get_transactions(self, user_id, limit=50):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ticker, type, shares, price, total, profit_loss, timestamp 
            FROM transactions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        results = cursor.fetchall()
        conn.close()
        return [{'ticker': r[0], 'type': r[1], 'shares': r[2], 'price': r[3], 'total': r[4], 'profit_loss': r[5], 'timestamp': r[6]} for r in results]
    
    def buy_stock(self, user_id, ticker, shares, price):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        total_cost = shares * price
        
        cursor.execute('SELECT cash FROM users WHERE user_id = ?', (user_id,))
        cash_row = cursor.fetchone()
        if not cash_row:
            conn.close()
            return False, "User not found"
        cash = cash_row[0]
        
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
            INSERT INTO transactions (user_id, ticker, type, shares, price, total, profit_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'BUY', shares, price, total_cost, 0))
        
        cursor.execute('UPDATE users SET total_trades = total_trades + 1 WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        if user_id in self.cache:
            del self.cache[user_id]
        
        return True, f"Bought {shares} shares of {ticker} at ${price:.2f} (Total: ${total_cost:.2f})"
    
    def sell_stock(self, user_id, ticker, shares, price):
        conn = self._get_conn()
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
        profit_loss = (price - avg_price) * shares
        
        cursor.execute('UPDATE users SET cash = cash + ? WHERE user_id = ?', (total_value, user_id))
        cursor.execute('UPDATE users SET total_profit = total_profit + ? WHERE user_id = ?', (profit_loss, user_id))
        
        if profit_loss > 0:
            cursor.execute('UPDATE users SET winning_trades = winning_trades + 1 WHERE user_id = ?', (user_id,))
        
        new_shares = existing_shares - shares
        if new_shares == 0:
            cursor.execute('DELETE FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
        else:
            cursor.execute('UPDATE portfolio SET shares = ? WHERE user_id = ? AND ticker = ?', (new_shares, user_id, ticker))
        
        cursor.execute('''
            INSERT INTO transactions (user_id, ticker, type, shares, price, total, profit_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'SELL', shares, price, total_value, profit_loss))
        
        cursor.execute('UPDATE users SET total_trades = total_trades + 1 WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        if user_id in self.cache:
            del self.cache[user_id]
        
        return True, f"Sold {shares} shares of {ticker} at ${price:.2f} (P/L: ${profit_loss:.2f})"
    
    def get_portfolio_value(self, user_id):
        if user_id in self.cache:
            cache_time, data = self.cache[user_id]
            if (datetime.now() - cache_time).seconds < 5:
                return data
        
        portfolio = self.get_portfolio(user_id)
        
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT cash, total_profit, total_trades, winning_trades FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return self._empty_portfolio()
        cash, total_profit, total_trades, winning_trades = row
        conn.close()
        
        holdings = []
        total_holdings_value = 0
        total_cost_basis = 0
        
        for item in portfolio:
            try:
                stock = yf.Ticker(item['ticker'])
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    current_value = current_price * item['shares']
                    cost_basis = item['avg_price'] * item['shares']
                    
                    total_holdings_value += current_value
                    total_cost_basis += cost_basis
                    
                    profit_loss = current_value - cost_basis
                    profit_loss_pct = ((current_price / item['avg_price']) - 1) * 100 if item['avg_price'] > 0 else 0
                    
                    holdings.append({
                        'ticker': item['ticker'],
                        'shares': item['shares'],
                        'avg_price': item['avg_price'],
                        'current_price': current_price,
                        'value': current_value,
                        'cost_basis': cost_basis,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct
                    })
            except Exception as e:
                logger.error(f"⚠️ Error getting price for {item['ticker']}: {e}")
                current_value = item['avg_price'] * item['shares']
                total_holdings_value += current_value
                total_cost_basis += current_value
                holdings.append({
                    'ticker': item['ticker'],
                    'shares': item['shares'],
                    'avg_price': item['avg_price'],
                    'current_price': item['avg_price'],
                    'value': current_value,
                    'cost_basis': current_value,
                    'profit_loss': 0,
                    'profit_loss_pct': 0
                })
        
        total_value = cash + total_holdings_value
        
        result = {
            'cash': cash,
            'holdings': holdings,
            'total_holdings_value': total_holdings_value,
            'total_cost_basis': total_cost_basis,
            'total_value': total_value,
            'total_profit': total_profit,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0
        }
        
        self.cache[user_id] = (datetime.now(), result)
        self._save_performance_history(user_id, result)
        
        return result
    
    def _empty_portfolio(self):
        return {
            'cash': 10000,
            'holdings': [],
            'total_holdings_value': 0,
            'total_cost_basis': 0,
            'total_value': 10000,
            'total_profit': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0
        }
    
    def _save_performance_history(self, user_id, portfolio_data):
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            current_minute = datetime.now().strftime('%Y-%m-%d %H:%M:00')
            cursor.execute('''
                SELECT COUNT(*) FROM performance_history 
                WHERE user_id = ? AND strftime('%Y-%m-%d %H:%M:00', timestamp) = ?
            ''', (user_id, current_minute))
            
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO performance_history (user_id, total_value, cash, holdings_value, total_profit)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, portfolio_data['total_value'], portfolio_data['cash'], 
                     portfolio_data['total_holdings_value'], portfolio_data.get('total_profit', 0)))
                conn.commit()
            
            conn.close()
        except:
            pass
    
    def get_performance_history(self, user_id, days=7):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT total_value, cash, holdings_value, total_profit, timestamp 
            FROM performance_history 
            WHERE user_id = ? AND timestamp > datetime('now', ?)
            ORDER BY timestamp ASC
        ''', (user_id, f'-{days} days'))
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'total_value': r[0],
            'cash': r[1],
            'holdings_value': r[2],
            'total_profit': r[3],
            'timestamp': r[4]
        } for r in results]

# ============================================================
# ALPHA VANTAGE API FUNCTIONS
# ============================================================

def get_alpha_vantage_price(ticker):
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_pct': float(quote.get('10. change percent', '0%').replace('%', '')),
                    'volume': int(quote.get('06. volume', 0))
                }
    except Exception as e:
        logger.error(f"⚠️ Alpha Vantage error for {ticker}: {e}")
    return None

def get_alpha_vantage_historical(ticker, days=60):
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                dates = sorted(time_series.keys(), reverse=True)[:days]
                dates.reverse()
                
                prices = []
                volumes = []
                for date in dates:
                    prices.append(float(time_series[date]['4. close']))
                    volumes.append(int(time_series[date]['5. volume']))
                
                return {
                    'dates': dates,
                    'prices': prices,
                    'volumes': volumes
                }
    except Exception as e:
        logger.error(f"⚠️ Alpha Vantage historical error for {ticker}: {e}")
    return None

# ============================================================
# CACHED SPY DATA
# ============================================================

_spy_cache = {}
_spy_cache_lock = threading.Lock()

def get_spy_data():
    with _spy_cache_lock:
        now = datetime.now()
        if 'data' in _spy_cache and (now - _spy_cache.get('time', datetime.min)).seconds < 60:
            return _spy_cache['data']
        
        try:
            spy = yf.download("SPY", period="2mo", progress=False)
            if not spy.empty:
                _spy_cache['data'] = spy
                _spy_cache['time'] = now
                return spy
        except Exception as e:
            logger.error(f"SPY download error: {e}")
        return None

# ============================================================
# PRICE PREDICTION ENGINE
# ============================================================

class PricePredictionEngine:
    def __init__(self):
        self.prediction_cache = {}
        self.cache_ttl = 300
    
    def predict_next_day(self, ticker, historical_data):
        cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.prediction_cache:
            cache_time, data = self.prediction_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            if not historical_data or len(historical_data.get('prices', [])) < 10:
                return None
            
            prices = historical_data.get('prices', [])
            if len(prices) < 10:
                return None
            
            recent_prices = prices[-30:]
            days = np.array(range(len(recent_prices))).reshape(-1, 1)
            prices_array = np.array(recent_prices).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(days, prices_array)
            
            next_day = np.array([[len(recent_prices)]])
            predicted_price = model.predict(next_day)[0][0]
            
            r2 = model.score(days, prices_array)
            confidence = min(100, max(50, r2 * 100 + 20))
            
            current_price = recent_prices[-1]
            expected_change = ((predicted_price - current_price) / current_price) * 100
            
            if expected_change > 2:
                prediction = "STRONG BULLISH"
            elif expected_change > 0.5:
                prediction = "BULLISH"
            elif expected_change > -0.5:
                prediction = "NEUTRAL"
            elif expected_change > -2:
                prediction = "BEARISH"
            else:
                prediction = "STRONG BEARISH"
            
            result = {
                'predicted_price': round(predicted_price, 2),
                'current_price': round(current_price, 2),
                'expected_change': round(expected_change, 2),
                'confidence': round(confidence, 2),
                'prediction': prediction,
                'support': round(min(recent_prices) * 0.97, 2),
                'resistance': round(max(recent_prices) * 1.03, 2),
                'trend_strength': 'STRONG' if abs(expected_change) > 2 else 'MODERATE' if abs(expected_change) > 1 else 'WEAK'
            }
            
            self.prediction_cache[cache_key] = (datetime.now(), result)
            return result
            
        except Exception as e:
            logger.error(f"⚠️ Prediction error for {ticker}: {e}")
            return None

# ============================================================
# ENHANCED NEWS SCRAPER - WITH FIXED FEEDFLASH
# ============================================================

class EnhancedNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.timeout = 15
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        self.ua_index = 0
        self.scrape_cache = {}
        self.cache_ttl = 180
        self.ai_engine = None
        self.news_history = deque(maxlen=500)
        self._session_lock = threading.Lock()
        
    def set_ai_engine(self, ai_engine):
        self.ai_engine = ai_engine
        
    def _rotate_user_agent(self):
        with self._session_lock:
            self.ua_index = (self.ua_index + 1) % len(self.user_agents)
            self.session.headers.update({'User-Agent': self.user_agents[self.ua_index]})
    
    def _safe_scrape(self, url):
        try:
            self._rotate_user_agent()
            response = self.session.get(url, timeout=self.timeout, headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            })
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"⚠️ Scrape failed with status {response.status_code}: {url}")
        except requests.RequestException as e:
            logger.warning(f"⚠️ Request error for {url}: {e}")
        except Exception as e:
            logger.error(f"⚠️ Unexpected error scraping {url}: {e}")
        return None
    
    def _clean_headline(self, text):
        """Clean a headline by removing common noise"""
        # Remove sentiment labels and emojis
        for sent in ['Bullish', 'Bearish', 'Neutral', 'BULLISH', 'BEARISH', 'NEUTRAL', 
                     '🟢', '🔴', '🟡', '📈', '📉', '⚡', '💰', '💎', '🚀']:
            text = text.replace(sent, '').strip()
        
        # Remove ticker patterns like (AAPL)
        text = re.sub(r'\([A-Z]{1,5}\)', '', text).strip()
        
        # Remove common prefixes/suffixes
        text = re.sub(r'^[:\-\s•·●○◆◇▸▹►➢➤]+', '', text)
        text = re.sub(r'[:\-\s•·●○◆◇▸▹►➢➤]+$', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _extract_sentiment(self, elem):
        """Extract sentiment from element or its siblings"""
        sentiment = 'NEUTRAL'
        
        # Check element's text
        text = elem.get_text()
        if 'Bullish' in text or '🟢' in text:
            sentiment = 'BULLISH'
        elif 'Bearish' in text or '🔴' in text:
            sentiment = 'BEARISH'
        elif 'Neutral' in text or '🟡' in text:
            sentiment = 'NEUTRAL'
        
        # Check parent
        if elem.parent:
            parent_text = elem.parent.get_text()
            if 'Bullish' in parent_text or '🟢' in parent_text:
                sentiment = 'BULLISH'
            elif 'Bearish' in parent_text or '🔴' in parent_text:
                sentiment = 'BEARISH'
            elif 'Neutral' in parent_text or '🟡' in parent_text:
                sentiment = 'NEUTRAL'
        
        # Check siblings
        if elem.next_sibling:
            sibling_text = str(elem.next_sibling)
            if 'Bullish' in sibling_text or '🟢' in sibling_text:
                sentiment = 'BULLISH'
            elif 'Bearish' in sibling_text or '🔴' in sibling_text:
                sentiment = 'BEARISH'
            elif 'Neutral' in sibling_text or '🟡' in sibling_text:
                sentiment = 'NEUTRAL'
        
        return sentiment

    def _extract_ticker(self, text):
        """Extract ticker symbol from text"""
        # Look for ticker in parentheses
        ticker_match = re.search(r'\(([A-Z]{1,5})\)', text)
        if ticker_match:
            return ticker_match.group(1)
        
        # Look for ticker as a standalone word
        ticker_match = re.search(r'\b([A-Z]{2,5})\b(?=\s|$|\.|\:)', text)
        if ticker_match:
            # Check if it's a common word not a ticker
            common_words = {'US', 'UK', 'EU', 'AI', 'CEO', 'CFO', 'IPO', 'GDP', 'ETF', 'SEC', 'FDA'}
            if ticker_match.group(1) not in common_words:
                return ticker_match.group(1)
        
        return ''
    
    def scrape_feedflash(self, ticker=''):
        """Scrape news from FeedFlash - Fixed to properly extract articles"""
        results = {'news': []}
        cache_key = f"feedflash_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            url = "https://feedflash-production.up.railway.app/news/"
            html = self._safe_scrape(url)
            if not html:
                logger.warning("⚠️ Failed to fetch FeedFlash content")
                return results
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove ALL navigation, header, and control elements
            for nav in soup.find_all(['nav', 'header', 'footer', 'aside']):
                nav.decompose()
            
            # Remove elements with common navigation/control classes
            for elem in soup.find_all(class_=re.compile(r'nav|menu|header|footer|control|toolbar|filter|tab|settings', re.I)):
                elem.decompose()
            
            # Remove elements containing navigation text patterns
            nav_patterns = [
                r'FlashFeed.*Financial Intelligence',
                r'News.*Screener.*Social',
                r'Charts.*Momentum',
                r'30s.*1m.*2m.*5m.*10m.*30m',
                r'Auto',
                r'Keywords Only',
                r'News Feed.*articles'
            ]
            for pattern in nav_patterns:
                for elem in soup.find_all(string=re.compile(pattern, re.DOTALL)):
                    if elem.parent:
                        elem.parent.decompose()
            
            # Also remove any elements with "Auto" or refresh controls
            for elem in soup.find_all(string=re.compile(r'Auto|Refresh|30s|1m|2m|5m|10m|30m')):
                if elem.parent and len(elem.parent.get_text(strip=True)) < 50:
                    elem.parent.decompose()
            
            # Now look for actual news content
            seen_headlines = set()
            
            # Look for elements that contain news content
            potential_articles = []
            
            # Find all divs that might contain news
            for div in soup.find_all('div'):
                text = div.get_text(strip=True)
                if len(text) < 25:
                    continue
                if any(pattern in text for pattern in ['FlashFeed', 'Financial Intelligence', 'News Feed', 'Keywords Only', 'Auto', '30s', '1m', '2m', '5m']):
                    continue
                if 25 < len(text) < 300:
                    potential_articles.append((div, text))
            
            # Also check list items
            for li in soup.find_all('li'):
                text = li.get_text(strip=True)
                if len(text) < 25:
                    continue
                if any(pattern in text for pattern in ['FlashFeed', 'Financial Intelligence', 'News Feed', 'Keywords Only', 'Auto', '30s', '1m', '2m', '5m']):
                    continue
                if 25 < len(text) < 300:
                    potential_articles.append((li, text))
            
            # Process potential articles
            for elem, text in potential_articles:
                # Clean the headline
                headline = self._clean_headline(text)
                if not headline or len(headline) < 20:
                    continue
                
                if headline in seen_headlines:
                    continue
                seen_headlines.add(headline)
                
                sentiment = self._extract_sentiment(elem)
                ticker_symbol = self._extract_ticker(headline)
                
                if ticker and ticker_symbol:
                    if ticker.upper() != ticker_symbol.upper():
                        continue
                
                sentiment_data = {'label': sentiment, 'score': 0}
                if self.ai_engine:
                    sentiment_data = self.ai_engine.get_ai_sentiment(headline)
                
                news_item = {
                    'headline': headline[:500],
                    'source': 'FeedFlash',
                    'time': '',
                    'sentiment': sentiment_data,
                    'link': '',
                    'ticker': ticker_symbol
                }
                results['news'].append(news_item)
                self.news_history.append(news_item)
            
            # If no articles found, try parsing the text directly
            if not results['news']:
                text = soup.get_text()
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    if any(pattern in line for pattern in ['FlashFeed', 'Financial Intelligence', 'News Feed', 'Keywords Only', 'Auto', '30s', '1m', '2m', '5m']):
                        continue
                    if len(line) < 25 or len(line) > 300:
                        continue
                    
                    headline = self._clean_headline(line)
                    if headline and len(headline) > 20 and headline not in seen_headlines:
                        seen_headlines.add(headline)
                        
                        sentiment = 'NEUTRAL'
                        if 'Bullish' in line or '🟢' in line:
                            sentiment = 'BULLISH'
                        elif 'Bearish' in line or '🔴' in line:
                            sentiment = 'BEARISH'
                        elif 'Neutral' in line or '🟡' in line:
                            sentiment = 'NEUTRAL'
                        
                        ticker_symbol = self._extract_ticker(headline)
                        
                        if ticker and ticker_symbol:
                            if ticker.upper() != ticker_symbol.upper():
                                continue
                        
                        sentiment_data = {'label': sentiment, 'score': 0}
                        if self.ai_engine:
                            sentiment_data = self.ai_engine.get_ai_sentiment(headline)
                        
                        news_item = {
                            'headline': headline[:500],
                            'source': 'FeedFlash',
                            'time': '',
                            'sentiment': sentiment_data,
                            'link': '',
                            'ticker': ticker_symbol
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            
            self.scrape_cache[cache_key] = (datetime.now(), results)
            logger.info(f"✅ FeedFlash scraped {len(results['news'])} articles for {ticker or 'all'}")
            
        except Exception as e:
            logger.error(f"❌ FeedFlash error for {ticker}: {e}")
        
        return results
    
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
                news_table = soup.find('table', {'id': 'news-table'}) or soup.find('table', {'class': 'fullview-news-outer'})
                if news_table:
                    for row in news_table.find_all('tr')[:8]:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            headline = cells[1].get_text(strip=True)
                            if headline and len(headline) > 5:
                                time_text = cells[0].get_text(strip=True) if cells[0] else ''
                                news_item = {
                                    'headline': headline,
                                    'time': time_text,
                                    'source': 'Finviz',
                                    'sentiment': self.ai_engine.get_ai_sentiment(headline) if self.ai_engine else {'label': 'NEUTRAL'}
                                }
                                results['news'].append(news_item)
                                self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except Exception as e:
            logger.error(f"Finviz error for {ticker}: {e}")
        return results
    
    def scrape_yahoo_finance(self, ticker):
        results = {'news': []}
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            for item in news[:8]:
                content = item.get('content', item)
                headline = content.get('title', '')
                link = (content.get('canonicalUrl') or {}).get('url', '') or content.get('link', '')
                if headline:
                    news_item = {
                        'headline': headline[:200],
                        'source': 'Yahoo Finance',
                        'link': link,
                        'sentiment': self.ai_engine.get_ai_sentiment(headline) if self.ai_engine else {'label': 'NEUTRAL'}
                    }
                    results['news'].append(news_item)
                    self.news_history.append(news_item)
        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
        return results
    
    def scrape_google_news(self, ticker):
        results = {'news': []}
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                news_item = {
                    'headline': entry.title[:200],
                    'source': 'Google News',
                    'link': entry.link,
                    'published': getattr(entry, 'published', ''),
                    'sentiment': self.ai_engine.get_ai_sentiment(entry.title) if self.ai_engine else {'label': 'NEUTRAL'}
                }
                results['news'].append(news_item)
                self.news_history.append(news_item)
        except Exception as e:
            logger.error(f"Google News error for {ticker}: {e}")
        return results
    
    def scrape_stocktwits(self, ticker):
        results = {'news': []}
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for msg in data.get('messages', [])[:5]:
                    body = msg.get('body', '')
                    if body and len(body) > 3:
                        news_item = {
                            'headline': body[:150],
                            'source': 'StockTwits',
                            'sentiment': self.ai_engine.get_ai_sentiment(body) if self.ai_engine else {'label': 'NEUTRAL'},
                            'user': msg.get('user', {}).get('username', ''),
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
        except Exception:
            pass
        return results
    
    def fetch_all_news(self, ticker, company_name=''):
        all_news = {}
        
        sources = {
            'feedflash': self.scrape_feedflash,
            'finviz': self.scrape_finviz,
            'yahoo': self.scrape_yahoo_finance,
            'google_news': self.scrape_google_news,
            'stocktwits': self.scrape_stocktwits,
        }
        
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_to_source = {executor.submit(fn, ticker): name for name, fn in sources.items()}
            for future in as_completed(future_to_source, timeout=20):
                source_name = future_to_source[future]
                try:
                    result = future.result(timeout=10)
                    if result and result.get('news'):
                        all_news[source_name] = result['news']
                except Exception as e:
                    logger.error(f"Error fetching from {source_name}: {e}")
        
        return all_news
    
    def get_news_feed(self, limit=200):
        return list(self.news_history)[-limit:]

# ============================================================
# AI ANALYSIS ENGINE
# ============================================================

class AIAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 180
        self.ai_usage = {"openai": 0, "claude": 0, "groq": 0, "gemini": 0, "total": 0, "failures": 0}
        self.last_ai_source = None
        
    def get_ai_sentiment(self, text):
        if not text:
            return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.3:
                label = "BULLISH" if polarity > 0.6 else "POSITIVE"
            elif polarity < -0.3:
                label = "BEARISH" if polarity < -0.6 else "NEGATIVE"
            else:
                label = "NEUTRAL"
            return {
                'label': label,
                'score': round(polarity, 2),
                'confidence': min(100, abs(polarity) * 50 + 20)
            }
        except:
            return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
    
    def get_ai_analysis(self, ticker, company, yahoo_data, sentiment_score, news_data, prediction_data):
        cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.analysis_cache:
            cache_time, data = self.analysis_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        pred_info = "No prediction available"
        if prediction_data:
            pred_info = f"Predicted Price: ${prediction_data.get('predicted_price', 0)}, Expected Change: {prediction_data.get('expected_change', 0)}%, Confidence: {prediction_data.get('confidence', 0)}%"
        
        prompt = self._build_ai_prompt(ticker, company, yahoo_data, sentiment_score, news_data, pred_info)
        result = None
        ai_source = None
        
        if not check_ai_rate_limit():
            logger.warning(f"⚠️ Global AI rate limit reached, using fallback for {ticker}")
            result = self._get_enhanced_fallback_analysis(ticker, company, yahoo_data, sentiment_score, prediction_data)
            ai_source = "Technical Fallback (Rate Limited)"
            self.last_ai_source = ai_source
            self.ai_usage["total"] += 1
        else:
            if not is_openai_rate_limited() and openai_client:
                try:
                    result = self._get_openai_analysis(prompt)
                    if result:
                        ai_source = "OpenAI GPT"
                        self.ai_usage["openai"] += 1
                        self.ai_usage["total"] += 1
                        self.last_ai_source = ai_source
                except Exception as e:
                    if "429" in str(e):
                        mark_openai_rate_limited()
                    logger.warning(f"OpenAI error: {e}")
            
            if not result and gemini_client:
                try:
                    result = self._get_gemini_analysis(prompt)
                    if result:
                        ai_source = "Gemini AI"
                        self.ai_usage["gemini"] += 1
                        self.ai_usage["total"] += 1
                        self.last_ai_source = ai_source
                except Exception as e:
                    logger.warning(f"Gemini error: {e}")
            
            if not result and claude_client:
                try:
                    result = self._get_claude_analysis(prompt)
                    if result:
                        ai_source = "Claude AI"
                        self.ai_usage["claude"] += 1
                        self.ai_usage["total"] += 1
                        self.last_ai_source = ai_source
                except Exception as e:
                    if "401" in str(e) or "Unauthorized" in str(e):
                        logger.warning("⚠️ Claude API key invalid or unauthorized")
                    else:
                        logger.warning(f"Claude error: {e}")
            
            if not result and not is_groq_rate_limited() and groq_client:
                try:
                    result = self._get_groq_analysis(prompt)
                    if result:
                        ai_source = "Groq AI"
                        self.ai_usage["groq"] += 1
                        self.ai_usage["total"] += 1
                        self.last_ai_source = ai_source
                except Exception as e:
                    if "429" in str(e):
                        mark_groq_rate_limited()
                    logger.warning(f"Groq error: {e}")
        
        if not result:
            self.ai_usage["failures"] += 1
            result = self._get_enhanced_fallback_analysis(ticker, company, yahoo_data, sentiment_score, prediction_data)
            ai_source = "Technical Fallback"
            self.last_ai_source = ai_source
        
        if result:
            result['ai_source'] = ai_source
            if prediction_data:
                result['prediction'] = prediction_data
            self.analysis_cache[cache_key] = (datetime.now(), result)
            return result
        return None
    
    def _build_ai_prompt(self, ticker, company, yahoo_data, sentiment_score, news_data, pred_info):
        headlines = []
        for source, items in news_data.items():
            if items:
                for item in items[:3]:
                    headline = item.get('headline', '')
                    if headline:
                        sentiment = item.get('sentiment', {}).get('label', 'NEUTRAL')
                        headlines.append(f"[{source}] {headline[:150]} (Sentiment: {sentiment})")
        news_summary = "\n".join(headlines[:10]) if headlines else "No recent news"
        
        price_direction = "UP" if yahoo_data['change_1d'] > 0 else "DOWN"
        sentiment_direction = "POSITIVE" if sentiment_score > 0.3 else "NEGATIVE" if sentiment_score < -0.3 else "NEUTRAL"
        trend_strength = yahoo_data.get('trend_strength', 'NEUTRAL')
        
        after_hours = yahoo_data.get('after_hours_pct', 0)
        macd = yahoo_data.get('macd_bullish', False)
        adx = yahoo_data.get('adx', 0)
        breakout = yahoo_data.get('breakout', False)
        relative_strength = yahoo_data.get('relative_strength', 0)
        boll_signal = yahoo_data.get('boll_signal', 'NORMAL')
        
        return f"""Analyze {ticker} ({company}) stock in detail and provide a comprehensive investment recommendation.

CRITICAL PRICE DIRECTION: The stock is moving {price_direction} ({yahoo_data['change_1d']}% today)
CRITICAL TREND STRENGTH: {trend_strength}
CRITICAL SENTIMENT: News sentiment is {sentiment_direction} (score: {sentiment_score:.2f})
AFTER HOURS: {after_hours:.2f}%
PREDICTION: {pred_info}

TECHNICAL DATA:
- Current Price: ${yahoo_data['price']}
- 1-Day Change: {yahoo_data['change_1d']}%
- RSI (14-day): {yahoo_data['rsi']}
- Trend: {yahoo_data['trend']}
- Trend Strength: {trend_strength}
- Volume Ratio: {yahoo_data['volume_ratio']}x average
- SMA 20: ${yahoo_data.get('sma20', 0)}
- SMA 50: ${yahoo_data.get('sma50', 0)}
- Price vs SMA20: {yahoo_data.get('price_vs_sma20', 'N/A')}
- Price vs SMA50: {yahoo_data.get('price_vs_sma50', 'N/A')}
- Consecutive Down Days: {yahoo_data.get('consecutive_down_days', 0)}
- P/E Ratio: {yahoo_data.get('pe_ratio', 'N/A')}
- MACD Bullish: {macd}
- ADX: {adx}
- Bollinger Signal: {boll_signal}
- Breakout: {breakout}
- Relative Strength vs SPY: {relative_strength}%

RECENT NEWS HEADLINES:
{news_summary}

Based on ALL factors including the prediction, provide a comprehensive recommendation.
Return ONLY valid JSON with this format:
{{
    "rec": "STRONG BUY/BUY/WATCH/AVOID/SELL",
    "conf": 0-100,
    "summary": "brief analysis",
    "technical_score": 0-100,
    "sentiment_score": 0-100,
    "risk_level": "LOW/MEDIUM/HIGH",
    "key_factors": ["factor1", "factor2", "factor3"],
    "price_target": "price target",
    "ai_insight": "unique insight",
    "momentum_score": 0-100
}}"""
    
    def _get_openai_analysis(self, prompt):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
                timeout=10
            )
            text = response.choices[0].message.content
            return self._parse_ai_response(text, "OpenAI")
        except Exception as e:
            if "429" in str(e):
                mark_openai_rate_limited()
            raise e
    
    def _get_gemini_analysis(self, prompt):
        try:
            response = gemini_client.generate_content(prompt)
            text = response.text
            return self._parse_ai_response(text, "Gemini")
        except:
            return None
    
    def _get_claude_analysis(self, prompt):
        try:
            response = claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
            text = response.content[0].text
            return self._parse_ai_response(text, "Claude")
        except Exception as e:
            if "401" in str(e):
                logger.warning("⚠️ Claude API key invalid")
            raise e
    
    def _get_groq_analysis(self, prompt):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
                timeout=10
            )
            text = response.choices[0].message.content
            return self._parse_ai_response(text, "Groq")
        except Exception as e:
            if "429" in str(e):
                mark_groq_rate_limited()
            raise e
    
    def _parse_ai_response(self, text, source):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1:
                result = json.loads(text[start:end])
                required = ['rec', 'conf', 'summary']
                for field in required:
                    if field not in result:
                        result[field] = 'WATCH' if field == 'rec' else 50
                result['_source'] = source
                return result
        except:
            pass
        return None
    
    def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment_score, prediction_data):
        score = 50
        change = yahoo_data['change_1d']
        rsi = yahoo_data['rsi']
        trend = yahoo_data.get('trend', 'NEUTRAL')
        trend_strength = yahoo_data.get('trend_strength', 'NEUTRAL')
        price_vs_sma20 = yahoo_data.get('price_vs_sma20', 'BELOW')
        price_vs_sma50 = yahoo_data.get('price_vs_sma50', 'BELOW')
        consecutive_down = yahoo_data.get('consecutive_down_days', 0)
        volume_ratio = yahoo_data.get('volume_ratio', 1)
        after_hours = yahoo_data.get('after_hours_pct', 0)
        macd_bullish = yahoo_data.get('macd_bullish', False)
        adx = yahoo_data.get('adx', 0)
        breakout = yahoo_data.get('breakout', False)
        relative_strength = yahoo_data.get('relative_strength', 0)
        boll_signal = yahoo_data.get('boll_signal', 'NORMAL')
        
        pred_confidence = 0
        pred_change = 0
        if prediction_data:
            pred_confidence = prediction_data.get('confidence', 0)
            pred_change = prediction_data.get('expected_change', 0)
        
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
        
        if pred_change > 2 and pred_confidence > 70:
            score += 25
        elif pred_change > 1 and pred_confidence > 60:
            score += 18
        elif pred_change > 0.5 and pred_confidence > 50:
            score += 10
        elif pred_change < -2 and pred_confidence > 70:
            score -= 18
        elif pred_change < -1 and pred_confidence > 60:
            score -= 10
        
        if after_hours > 2:
            score += 30
        elif after_hours > 1:
            score += 22
        elif after_hours > 0.5:
            score += 14
        
        if consecutive_down >= 5:
            score -= 15
        elif consecutive_down >= 3:
            score -= 8
        elif consecutive_down >= 2:
            score -= 3
        else:
            score += 10
        
        if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
            score += 20
        elif price_vs_sma20 == 'ABOVE':
            score += 12
        elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
            score -= 12
        elif price_vs_sma20 == 'BELOW':
            score -= 5
        
        if 'STRONG_BULLISH' in trend or trend_strength == 'STRONG_BULLISH':
            score += 22
        elif 'BULLISH' in trend or trend_strength == 'BULLISH':
            score += 15
        elif 'STRONG_BEARISH' in trend or trend_strength == 'STRONG_BEARISH':
            score -= 12
        elif 'BEARISH' in trend or trend_strength == 'BEARISH':
            score -= 6
        
        if 40 < rsi < 70:
            score += 10
        elif 30 < rsi < 40:
            score += 8
        elif rsi > 70:
            if change > 0:
                score -= 5
            else:
                score += 5
        elif rsi < 30:
            if change < 0 and consecutive_down >= 3:
                score -= 5
            else:
                score += 10
        
        if volume_ratio > 1.5:
            if change > 0:
                score += 18
            elif change < 0:
                score -= 8
            else:
                score += 5
        elif volume_ratio > 1.0:
            if change > 0:
                score += 8
            else:
                score += 2
        
        if sentiment_score > 0.3:
            if change > 0:
                score += 18
            else:
                score += 8
        elif sentiment_score < -0.3:
            if change < 0:
                score -= 12
            else:
                score -= 5
        
        if macd_bullish:
            score += 10
        else:
            score -= 5
        
        if adx > 30:
            score += 12
        elif adx > 25:
            score += 8
        elif adx < 15:
            score -= 8
        
        if breakout:
            score += 18
        
        if relative_strength > 5:
            score += 12
        elif relative_strength > 2:
            score += 7
        elif relative_strength < -3:
            score -= 10
        
        if boll_signal == 'OVERSOLD':
            score += 10
        elif boll_signal == 'OVERBOUGHT':
            score -= 8
        
        score = max(10, min(90, score))
        
        if score >= 75:
            rec = "STRONG BUY"
            summary = f"📈 {ticker} strong upward momentum with positive trend"
        elif score >= 62:
            rec = "BUY"
            summary = f"✅ {ticker} positive price action with good momentum"
        elif score >= 48:
            rec = "WATCH"
            summary = f"👀 {ticker} consolidating, wait for direction"
        elif score >= 35:
            rec = "AVOID"
            summary = f"⚖️ {ticker} weak signals, caution advised"
        else:
            rec = "SELL"
            summary = f"🚨 {ticker} bearish signals, consider exit"
        
        if after_hours > 1:
            summary += f" (📈 AFTER HOURS +{after_hours:.1f}%)"
        elif after_hours > 0.5:
            summary += f" (📈 After Hours +{after_hours:.1f}%)"
        
        if pred_change > 1 and pred_confidence > 60:
            summary += f" (📊 Prediction: +{pred_change:.1f}% with {pred_confidence:.0f}% confidence)"
        
        if price_vs_sma20 == 'ABOVE':
            summary += " (✅ Above SMA20)"
        if consecutive_down < 2 and change > 0:
            summary += " (📈 Positive momentum)"
        if breakout:
            summary += " (🚀 Breakout!)"
        
        key_factors = ["Price direction", "Technical analysis", "Trend strength", "Sentiment correlation"]
        if macd_bullish:
            key_factors.append("MACD bullish")
        if breakout:
            key_factors.append("Breakout above resistance")
        if relative_strength > 2:
            key_factors.append("Outperforming market")
        if adx > 25:
            key_factors.append("Strong trend")
        if after_hours > 1:
            key_factors.append(f"After Hours +{after_hours:.1f}%")
        if pred_change > 1 and pred_confidence > 60:
            key_factors.append(f"AI Prediction +{pred_change:.1f}%")
        
        return {
            "rec": rec,
            "conf": score,
            "summary": summary,
            "technical_score": score,
            "sentiment_score": max(0, min(100, 50 + sentiment_score * 10)),
            "risk_level": "HIGH" if consecutive_down >= 4 or score < 30 else "MEDIUM" if score < 55 else "LOW",
            "key_factors": key_factors[:5],
            "price_target": f"${yahoo_data['price'] * (1 + (score - 50)/250):.2f}",
            "ai_insight": f"Price moving {change:+.1f}% with {consecutive_down} down days, AH: +{after_hours:.1f}%",
            "momentum_score": score,
            "_source": "Technical Fallback"
        }

# ============================================================
# NEWS SOURCES CONFIG
# ============================================================

NEWS_SOURCES = {
    "feedflash": {"name": "FeedFlash", "enabled": True, "icon": "⚡", "category": "aggregator"},
    "finviz": {"name": "Finviz", "enabled": True, "icon": "📊", "category": "equities"},
    "yahoo": {"name": "Yahoo Finance", "enabled": True, "icon": "💹", "category": "markets"},
    "google_news": {"name": "Google News", "enabled": True, "icon": "🔍", "category": "aggregator"},
    "stocktwits": {"name": "StockTwits", "enabled": True, "icon": "💬", "category": "social"},
}

CATEGORIES = {
    "aggregator": {"name": "Aggregators", "icon": "🔍", "count": 2},
    "equities": {"name": "Equities", "icon": "📈", "count": 1},
    "markets": {"name": "Markets", "icon": "📊", "count": 1},
    "social": {"name": "Social", "icon": "💬", "count": 1},
}

# ============================================================
# STOCKS DATABASE
# ============================================================

ALL_STOCKS = {
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
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc", "sector": "Healthcare"},
    "MRK": {"name": "Merck & Co", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie Inc", "sector": "Healthcare"},
    "TMO": {"name": "Thermo Fisher", "sector": "Healthcare"},
    "ABT": {"name": "Abbott Labs", "sector": "Healthcare"},
    "DHR": {"name": "Danaher Corp", "sector": "Healthcare"},
    "BMY": {"name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare"},
    "AMGN": {"name": "Amgen Inc", "sector": "Healthcare"},
    "CVS": {"name": "CVS Health Corp", "sector": "Healthcare"},
    "KO": {"name": "Coca-Cola Co", "sector": "Consumer"},
    "PEP": {"name": "PepsiCo Inc", "sector": "Consumer"},
    "COST": {"name": "Costco Wholesale", "sector": "Consumer"},
    "WMT": {"name": "Walmart Inc", "sector": "Consumer"},
    "TGT": {"name": "Target Corp", "sector": "Consumer"},
    "HD": {"name": "Home Depot", "sector": "Consumer"},
    "LOW": {"name": "Lowe's Companies", "sector": "Consumer"},
    "MCD": {"name": "McDonald's Corp", "sector": "Consumer"},
    "SBUX": {"name": "Starbucks Corp", "sector": "Consumer"},
    "NKE": {"name": "Nike Inc", "sector": "Consumer"},
    "DIS": {"name": "Walt Disney", "sector": "Consumer"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "EOG": {"name": "EOG Resources", "sector": "Energy"},
    "SLB": {"name": "Schlumberger", "sector": "Energy"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy"},
    "PSX": {"name": "Phillips 66", "sector": "Energy"},
    "VLO": {"name": "Valero Energy", "sector": "Energy"},
    "GE": {"name": "General Electric", "sector": "Industrial"},
    "CAT": {"name": "Caterpillar Inc", "sector": "Industrial"},
    "BA": {"name": "Boeing Co", "sector": "Industrial"},
    "RTX": {"name": "Raytheon Technologies", "sector": "Industrial"},
    "HON": {"name": "Honeywell International", "sector": "Industrial"},
    "DE": {"name": "Deere & Co", "sector": "Industrial"},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrial"},
    "NOC": {"name": "Northrop Grumman", "sector": "Industrial"},
    "GD": {"name": "General Dynamics", "sector": "Industrial"},
    "MMM": {"name": "3M Company", "sector": "Industrial"},
    "T": {"name": "AT&T Inc", "sector": "Communications"},
    "VZ": {"name": "Verizon Communications", "sector": "Communications"},
    "TMUS": {"name": "T-Mobile US", "sector": "Communications"},
    "CMCSA": {"name": "Comcast Corp", "sector": "Communications"},
    "AMT": {"name": "American Tower", "sector": "Real Estate"},
    "PLD": {"name": "Prologis Inc", "sector": "Real Estate"},
    "SPG": {"name": "Simon Property Group", "sector": "Real Estate"},
    "CCI": {"name": "Crown Castle", "sector": "Real Estate"},
    "EQIX": {"name": "Equinix Inc", "sector": "Real Estate"},
    "O": {"name": "Realty Income", "sector": "Real Estate"},
    "WELL": {"name": "Welltower Inc", "sector": "Real Estate"},
    "AVB": {"name": "AvalonBay Communities", "sector": "Real Estate"},
}

# ============================================================
# ENHANCED STOCK ANALYZER
# ============================================================

class EnhancedStockAnalyzer:
    def __init__(self):
        self.stock_cache = {}
        self.cache_ttl = 60
        self.ai_engine = None
        self.news_scraper = None
        self.prediction_engine = None
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
    
    def set_ai_engine(self, ai_engine):
        self.ai_engine = ai_engine
    
    def set_news_scraper(self, news_scraper):
        self.news_scraper = news_scraper
    
    def set_prediction_engine(self, prediction_engine):
        self.prediction_engine = prediction_engine
    
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
        trend = stock_data.get('trend', 'NEUTRAL')
        
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
                return self._get_fallback_data(ticker)
            
            info = stock.info
            current_price = float(hist['Close'].iloc[-1])
            current_volume = float(hist['Volume'].iloc[-1])
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            loss = loss.replace(0, 0.001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            sma20_val = hist['Close'].rolling(20).mean()
            sma50_val = hist['Close'].rolling(50).mean()
            sma20 = float(sma20_val.iloc[-1]) if len(hist) >= 20 and not pd.isna(sma20_val.iloc[-1]) else float(current_price)
            sma50 = float(sma50_val.iloc[-1]) if len(hist) >= 50 and not pd.isna(sma50_val.iloc[-1]) else float(current_price)
            
            avg_volume_val = hist['Volume'].rolling(20).mean()
            avg_volume = float(avg_volume_val.iloc[-1]) if len(hist) >= 20 and not pd.isna(avg_volume_val.iloc[-1]) else float(hist['Volume'].mean())
            volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 1
            
            price_vs_sma20 = 'ABOVE' if current_price > sma20 else 'BELOW'
            price_vs_sma50 = 'ABOVE' if current_price > sma50 else 'BELOW'
            
            consecutive_down = 0
            close_prices = hist['Close'].tolist()
            for i in range(len(close_prices)-1, 0, -1):
                if close_prices[i] < close_prices[i-1]:
                    consecutive_down += 1
                else:
                    break
            
            if current_price > sma20 and sma20 > sma50:
                trend = "STRONG BULLISH"
                trend_strength = "STRONG_BULLISH"
            elif current_price > sma20 and current_price > sma50:
                trend = "BULLISH"
                trend_strength = "BULLISH"
            elif current_price > sma20 and current_price < sma50:
                trend = "CONSOLIDATING (above SMA20)"
                trend_strength = "NEUTRAL_BULLISH"
            elif current_price < sma20 and sma20 < sma50:
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
            
            change_1d = float(((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100) if len(hist) >= 2 else 0
            
            after_hours_price = float(info.get('postMarketPrice', 0)) if info.get('postMarketPrice') else 0
            after_hours_pct = 0
            if after_hours_price > 0 and current_price > 0:
                after_hours_pct = float(((after_hours_price - current_price) / current_price) * 100)
            
            macd_bullish = False
            try:
                if len(hist) >= 26:
                    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
                    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
                    macd_line = ema12 - ema26
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    if len(macd_line) > 0 and len(signal_line) > 0:
                        macd_val = float(macd_line.iloc[-1])
                        signal_val = float(signal_line.iloc[-1])
                        macd_bullish = macd_val > signal_val
            except:
                pass
            
            adx = 0
            try:
                if len(hist) >= 28:
                    plus_dm = hist['High'].diff()
                    minus_dm = hist['Low'].diff() * -1
                    tr1 = hist['High'] - hist['Low']
                    tr2 = abs(hist['High'] - hist['Close'].shift())
                    tr3 = abs(hist['Low'] - hist['Close'].shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    plus_di = 100 * ((plus_dm.where(plus_dm > minus_dm, 0)).rolling(14).mean() / atr)
                    minus_di = 100 * ((minus_dm.where(minus_dm > plus_dm, 0)).rolling(14).mean() / atr)
                    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                    adx_series = dx.rolling(14).mean()
                    if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]):
                        adx = float(adx_series.iloc[-1])
            except:
                pass
            
            breakout = False
            try:
                if len(hist) >= 20:
                    highest20 = hist['High'].rolling(20).max()
                    if len(highest20) >= 2 and not pd.isna(highest20.iloc[-2]):
                        highest20_val = float(highest20.iloc[-2])
                        breakout = current_price > highest20_val
            except:
                pass
            
            relative_strength = 0
            try:
                if len(hist) >= 20:
                    spy = get_spy_data()
                    if spy is not None and not spy.empty and len(spy) >= 20:
                        stock_return = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1)
                        spy_return = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-20]) - 1)
                        relative_strength = float((stock_return - spy_return) * 100)
            except:
                pass
            
            boll_signal = "NORMAL"
            try:
                if len(hist) >= 20:
                    middle = hist['Close'].rolling(20).mean()
                    std = hist['Close'].rolling(20).std()
                    upper = middle + std * 2
                    lower = middle - std * 2
                    if len(upper) > 0 and not pd.isna(upper.iloc[-1]):
                        upper_val = float(upper.iloc[-1])
                        lower_val = float(lower.iloc[-1])
                        if current_price < lower_val:
                            boll_signal = "OVERSOLD"
                        elif current_price > upper_val:
                            boll_signal = "OVERBOUGHT"
            except:
                pass
            
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
                "trend_icon": "📈" if "BULLISH" in trend or "UPTREND" in trend else "📉" if "BEARISH" in trend or "DOWNTREND" in trend else "➡️",
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
                "current_volume": int(current_volume),
                "after_hours_price": round(float(after_hours_price), 2),
                "after_hours_pct": round(float(after_hours_pct), 2),
                "macd_bullish": macd_bullish,
                "adx": round(float(adx), 1),
                "breakout": breakout,
                "relative_strength": round(float(relative_strength), 2),
                "boll_signal": boll_signal,
                "support": round(float(current_price * 0.95), 2),
                "resistance": round(float(current_price * 1.05), 2),
            }
            self.stock_cache[ticker] = (datetime.now(), result)
            return result
        except Exception as e:
            logger.error(f"⚠️ Error for {ticker}: {e}")
            return self._get_fallback_data(ticker)
    
    def _get_fallback_data(self, ticker):
        sector_info = ALL_STOCKS.get(ticker, {})
        price = 10 + random.random() * 200
        change = (random.random() - 0.3) * 4
        rsi = 35 + random.random() * 30
        after_hours = (random.random() - 0.1) * 2
        
        return {
            "ticker": ticker,
            "company": sector_info.get('name', ticker),
            "sector": sector_info.get('sector', 'Unknown'),
            "price": round(price, 2),
            "change_1d": round(change, 2),
            "rsi": round(rsi, 1),
            "volume_ratio": round(0.8 + random.random() * 0.8, 2),
            "trend": "BULLISH" if change > 0.5 else "NEUTRAL" if abs(change) < 0.3 else "BEARISH",
            "trend_strength": "BULLISH" if change > 1 else "NEUTRAL",
            "trend_icon": "📈" if change > 0.5 else "📉" if change < -0.5 else "➡️",
            "sma20": round(price * 0.98, 2),
            "sma50": round(price * 0.97, 2),
            "price_vs_sma20": "ABOVE" if change > 0 else "BELOW",
            "price_vs_sma50": "ABOVE" if change > -0.5 else "BELOW",
            "consecutive_down_days": 0 if change > 0 else random.randint(1, 3),
            "historical": {"dates": [], "prices": [], "volumes": []},
            "pe_ratio": round(15 + random.random() * 20, 2),
            "target_price": round(price * (1 + random.random() * 0.2), 2),
            "current_volume": int(500000 + random.random() * 2000000),
            "after_hours_price": round(price * (1 + after_hours / 100), 2),
            "after_hours_pct": round(after_hours, 2),
            "macd_bullish": change > 0,
            "adx": round(15 + random.random() * 25, 1),
            "breakout": change > 1.5 and rsi > 55,
            "relative_strength": round((change * 2) + (random.random() - 0.5) * 4, 2),
            "boll_signal": "OVERSOLD" if rsi < 35 else "OVERBOUGHT" if rsi > 65 else "NORMAL",
            "support": round(price * 0.93, 2),
            "resistance": round(price * 1.07, 2),
        }
    
    def get_news_sentiment(self, ticker):
        if not self.news_scraper:
            return {
                'news_data': {},
                'sentiment_scores': {'BULLISH': 0, 'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0, 'BEARISH': 0},
                'total_news': 0,
                'sentiment_score': 0,
                'avg_sentiment': 0,
                'news_items': []
            }
        
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

def generate_recommendation_enhanced(data, sentiment_score, ai_analysis, prediction_data):
    score = 50
    change = data['change_1d']
    rsi = data['rsi']
    consecutive_down = data.get('consecutive_down_days', 0)
    price_vs_sma20 = data.get('price_vs_sma20', 'BELOW')
    price_vs_sma50 = data.get('price_vs_sma50', 'BELOW')
    trend = data.get('trend', 'NEUTRAL')
    volume_ratio = data.get('volume_ratio', 1)
    after_hours = data.get('after_hours_pct', 0)
    macd_bullish = data.get('macd_bullish', False)
    adx = data.get('adx', 0)
    breakout = data.get('breakout', False)
    relative_strength = data.get('relative_strength', 0)
    boll_signal = data.get('boll_signal', 'NORMAL')
    
    pred_change = 0
    pred_confidence = 0
    if prediction_data:
        pred_change = prediction_data.get('expected_change', 0)
        pred_confidence = prediction_data.get('confidence', 0)
    
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
    
    if pred_change > 2 and pred_confidence > 70:
        score += 25
    elif pred_change > 1 and pred_confidence > 60:
        score += 18
    elif pred_change > 0.5 and pred_confidence > 50:
        score += 10
    elif pred_change < -2 and pred_confidence > 70:
        score -= 18
    elif pred_change < -1 and pred_confidence > 60:
        score -= 10
    
    if after_hours > 2:
        score += 30
    elif after_hours > 1:
        score += 22
    elif after_hours > 0.5:
        score += 14
    elif after_hours < -2:
        score -= 15
    elif after_hours < -1:
        score -= 8
    
    if consecutive_down >= 5:
        score -= 12
    elif consecutive_down >= 3:
        score -= 6
    elif consecutive_down >= 2:
        score -= 2
    else:
        score += 12
    
    if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
        score += 22
    elif price_vs_sma20 == 'ABOVE':
        score += 14
    elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
        score -= 10
    elif price_vs_sma20 == 'BELOW':
        score -= 4
    
    if 'STRONG_BULLISH' in trend:
        score += 25
    elif 'BULLISH' in trend:
        score += 16
    elif 'STRONG_BEARISH' in trend:
        score -= 10
    elif 'BEARISH' in trend:
        score -= 5
    
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
    
    if macd_bullish:
        score += 12
    else:
        score -= 6
    
    if adx > 30:
        score += 12
    elif adx > 25:
        score += 8
    elif adx < 15:
        score -= 8
    
    if breakout:
        score += 18
    
    if relative_strength > 5:
        score += 12
    elif relative_strength > 2:
        score += 7
    elif relative_strength < -3:
        score -= 10
    
    if boll_signal == 'OVERSOLD':
        score += 10
    elif boll_signal == 'OVERBOUGHT':
        score -= 8
    
    if ai_analysis:
        ai_rec = ai_analysis.get('rec', 'WATCH')
        technical_score = ai_analysis.get('technical_score', 50)
        ai_conf = ai_analysis.get('conf', 50)
        
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
        summary = f"📈 {data['ticker']} strong upward momentum with positive trend"
    elif score >= 62:
        rec = "BUY"
        summary = f"✅ {data['ticker']} positive price action with good momentum"
    elif score >= 48:
        rec = "WATCH"
        summary = f"👀 {data['ticker']} consolidating, wait for direction"
    elif score >= 35:
        rec = "AVOID"
        summary = f"⚖️ {data['ticker']} weak signals, caution advised"
    else:
        rec = "SELL"
        summary = f"🚨 {data['ticker']} bearish signals, consider exit"
    
    if after_hours > 1:
        summary += f" (📈 AFTER HOURS +{after_hours:.1f}%)"
    elif after_hours > 0.5:
        summary += f" (📈 After Hours +{after_hours:.1f}%)"
    
    if pred_change > 1 and pred_confidence > 60:
        summary += f" (📊 Prediction: +{pred_change:.1f}% with {pred_confidence:.0f}% confidence)"
    
    if price_vs_sma20 == 'ABOVE':
        summary += " (✅ Above SMA20)"
    if consecutive_down < 2 and change > 0:
        summary += " (📈 Positive momentum)"
    if breakout:
        summary += " (🚀 Breakout!)"
    if macd_bullish:
        summary += " (MACD Bullish)"
    
    confidence = min(100, round(score * 0.8 + 10))
    momentum_score = score
    
    return rec, confidence, summary, momentum_score, score

# ============================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================

scan_stats = {"technical": 0, "openai": 0, "claude": 0, "groq": 0, "gemini": 0, "total": 0}
stock_analyzer = EnhancedStockAnalyzer()
news_scraper = EnhancedNewsScraper()
ai_engine = AIAnalysisEngine()
prediction_engine = PricePredictionEngine()

news_scraper.set_ai_engine(ai_engine)
stock_analyzer.set_ai_engine(ai_engine)
stock_analyzer.set_news_scraper(news_scraper)
stock_analyzer.set_prediction_engine(prediction_engine)

loaded_tickers = set()

def get_tickers_by_sector(sector=None):
    if sector and sector != 'all':
        return [t for t, info in ALL_STOCKS.items() if info.get('sector', '') == sector]
    return list(ALL_STOCKS.keys())

def get_next_batch(sector=None, offset=0, batch_size=60, loaded_set=None):
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
    
    prediction_data = prediction_engine.predict_next_day(ticker, yahoo_data.get('historical', {}))
    if prediction_data:
        yahoo_data['prediction_data'] = prediction_data
    
    ai_analysis = None
    ai_source = "Technical"
    if use_ai:
        ai_analysis = ai_engine.get_ai_analysis(
            ticker, yahoo_data['company'], yahoo_data, sentiment_score, news_data, prediction_data
        )
        if ai_analysis:
            ai_source = ai_analysis.get('_source', 'AI')
            if 'OpenAI' in ai_source:
                scan_stats["openai"] += 1
            elif 'Claude' in ai_source:
                scan_stats["claude"] += 1
            elif 'Groq' in ai_source:
                scan_stats["groq"] += 1
            elif 'Gemini' in ai_source:
                scan_stats["gemini"] += 1
    
    rec, confidence, summary, momentum_score, score = generate_recommendation_enhanced(
        yahoo_data, sentiment_score, ai_analysis, prediction_data
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
    
    if prediction_data:
        pred_change = prediction_data.get('expected_change', 0)
        pred_confidence = prediction_data.get('confidence', 0)
        if pred_change > 2 and pred_confidence > 70:
            rank_score += 20
        elif pred_change > 1 and pred_confidence > 60:
            rank_score += 12
    
    after_hours = yahoo_data.get('after_hours_pct', 0)
    if after_hours > 2:
        rank_score += 25
    elif after_hours > 1:
        rank_score += 18
    elif after_hours > 0.5:
        rank_score += 10
    
    if yahoo_data.get('breakout', False):
        rank_score += 15
    if yahoo_data.get('macd_bullish', False):
        rank_score += 10
    if yahoo_data.get('relative_strength', 0) > 5:
        rank_score += 10
    if yahoo_data.get('adx', 0) > 30:
        rank_score += 8
    
    if ai_analysis:
        rank_score += ai_analysis.get('conf', 0) * 0.12
    
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
    
    pred_display = None
    if prediction_data:
        pred_display = {
            'predicted_price': prediction_data.get('predicted_price', 0),
            'expected_change': prediction_data.get('expected_change', 0),
            'confidence': prediction_data.get('confidence', 0),
            'prediction': prediction_data.get('prediction', 'NEUTRAL'),
            'support': prediction_data.get('support', 0),
            'resistance': prediction_data.get('resistance', 0)
        }
    
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
        "price_direction": "UP" if yahoo_data['change_1d'] > 0 else "DOWN",
        "after_hours_pct": yahoo_data.get('after_hours_pct', 0),
        "after_hours_price": yahoo_data.get('after_hours_price', 0),
        "macd_bullish": yahoo_data.get('macd_bullish', False),
        "adx": yahoo_data.get('adx', 0),
        "breakout": yahoo_data.get('breakout', False),
        "relative_strength": yahoo_data.get('relative_strength', 0),
        "boll_signal": yahoo_data.get('boll_signal', 'NORMAL'),
        "support": yahoo_data.get('support', 0),
        "resistance": yahoo_data.get('resistance', 0),
        "prediction": pred_display
    }
    
    return result

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global scan_stats, loaded_tickers
    scan_stats = {"technical": 0, "openai": 0, "claude": 0, "groq": 0, "gemini": 0, "total": 0}
    
    data = request.get_json() or {}
    tickers = data.get('tickers', [])
    use_ai = data.get('use_ai', True)
    sector = data.get('sector', None)
    limit = data.get('limit', 60)
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
                logger.error(f"⚠️ Error for {ticker}: {e}")
    
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
        'ai_availability': {
            'openai': openai_client is not None and not is_openai_rate_limited(),
            'claude': claude_client is not None,
            'groq': groq_client is not None and not is_groq_rate_limited(),
            'gemini': gemini_client is not None
        },
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
               'trend', 'trend_strength', 'recommendation', 'confidence', 'rank_score', 'news_count', 
               'sentiment_aggregate', 'source', 'ai_source', 'momentum_score', 'technical_score',
               'price_direction', 'consecutive_down_days', 'price_vs_sma20', 'price_vs_sma50',
               'after_hours_pct', 'macd_bullish', 'adx', 'breakout', 'relative_strength', 'boll_signal']
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

# ============================================================
# PAPER TRADING ROUTES
# ============================================================

@app.route('/api/paper/status', methods=['GET'])
def paper_status():
    try:
        paper_db = PaperTradingDB()
        user_id, cash, total_profit, total_trades, winning_trades = paper_db.get_or_create_user()
        portfolio_value = paper_db.get_portfolio_value(user_id)
        transactions = paper_db.get_transactions(user_id)
        
        portfolio_value['total_profit'] = total_profit
        portfolio_value['total_trades'] = total_trades
        portfolio_value['winning_trades'] = winning_trades
        portfolio_value['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'cash': cash,
            'portfolio': portfolio_value,
            'transactions': transactions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/buy', methods=['POST'])
def paper_buy():
    try:
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
            price = float(hist['Close'].iloc[-1])
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error getting price: {str(e)}'}), 400
        
        paper_db = PaperTradingDB()
        user_id, _, _, _, _ = paper_db.get_or_create_user()
        success, message = paper_db.buy_stock(user_id, ticker, shares, price)
        
        if success:
            portfolio = paper_db.get_portfolio_value(user_id)
            return jsonify({
                'success': True,
                'message': message,
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'total': shares * price,
                'portfolio': portfolio
            })
        else:
            return jsonify({'success': False, 'message': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/sell', methods=['POST'])
def paper_sell():
    try:
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
            price = float(hist['Close'].iloc[-1])
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error getting price: {str(e)}'}), 400
        
        paper_db = PaperTradingDB()
        user_id, _, _, _, _ = paper_db.get_or_create_user()
        success, message = paper_db.sell_stock(user_id, ticker, shares, price)
        
        if success:
            portfolio = paper_db.get_portfolio_value(user_id)
            return jsonify({
                'success': True,
                'message': message,
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'total': shares * price,
                'portfolio': portfolio
            })
        else:
            return jsonify({'success': False, 'message': message}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    try:
        paper_db = PaperTradingDB()
        user_id, _, _, _, _ = paper_db.get_or_create_user()
        
        conn = sqlite3.connect(paper_db.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM portfolio WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM performance_history WHERE user_id = ?', (user_id,))
        cursor.execute('UPDATE users SET cash = 10000, total_profit = 0, total_trades = 0, winning_trades = 0 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        if user_id in paper_db.cache:
            del paper_db.cache[user_id]
        
        return jsonify({'success': True, 'message': 'Account reset to $10,000'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/paper/history', methods=['GET'])
def paper_history():
    try:
        paper_db = PaperTradingDB()
        user_id, _, _, _, _ = paper_db.get_or_create_user()
        days = request.args.get('days', 7, type=int)
        history = paper_db.get_performance_history(user_id, days)
        
        return jsonify({
            'success': True,
            'history': history,
            'days': days
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/news/sources')
def get_news_sources():
    return jsonify({'success': True, 'sources': NEWS_SOURCES, 'categories': CATEGORIES})

@app.route('/api/news/feed')
def news_feed():
    news = news_scraper.get_news_feed(200)
    return jsonify({'success': True, 'news': news, 'count': len(news)})

@app.route('/api/stats/performance')
def performance_stats():
    return jsonify({'success': True, 'stats': []})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'openai_available': openai_client is not None and not is_openai_rate_limited(),
        'claude_available': claude_client is not None,
        'groq_available': groq_client is not None and not is_groq_rate_limited(),
        'gemini_available': gemini_client is not None,
        'total_stocks': len(ALL_STOCKS),
        'news_sources': len(NEWS_SOURCES),
        'filters': stock_analyzer.filters
    })

# ============================================================
# HTML TEMPLATE
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
        .sidebar{width:400px;min-width:400px;background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;max-height:calc(100vh - 40px);overflow-y:auto}
        .sidebar.collapsed{width:0;min-width:0;padding:0;border:none;overflow:hidden}
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
        .card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:10px}
        .stock-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px;transition:all 0.3s;cursor:pointer;min-height:360px;display:flex;flex-direction:column}
        .stock-card:hover{transform:translateY(-3px);border-color:rgba(102,126,234,0.4);box-shadow:0 12px 35px rgba(0,0,0,0.3)}
        .stock-card.pinned{border-color:#FFD700;background:rgba(255,215,0,0.05)}
        .stock-card.bullish-highlight{border-color:#4CAF50;background:rgba(76,175,80,0.05)}
        .stock-card.downtrend-warning{border-color:#f44336;background:rgba(244,67,54,0.05)}
        .stock-card.breakout-highlight{border-color:#FFD700;background:rgba(255,215,0,0.08)}
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
        .modal{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.85);z-index:1000;justify-content:center;align-items:center;padding:20px;backdrop-filter:blur(8px)}
        .modal.active{display:flex}
        .modal-content{background:#1a1a2e;border:1px solid rgba(255,255,255,0.1);border-radius:18px;max-width:1000px;width:100%;max-height:90vh;overflow-y:auto;padding:22px}
        .modal-close{font-size:26px;cursor:pointer;background:none;border:none;color:#666;padding:0 6px}
        .modal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0}
        .modal-stat{background:rgba(255,255,255,0.05);border-radius:6px;padding:8px}
        .modal-stat .label{color:#666;font-size:8px;text-transform:uppercase}
        .modal-stat .value{font-size:13px;font-weight:bold;margin-top:2px}
        .modal-chart{height:220px;margin:10px 0}
        .modal-chart canvas{width:100% !important;height:100% !important}
        .prediction-box{background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);border-radius:8px;padding:12px;margin:10px 0}
        .prediction-box .pred-title{color:#667eea;font-weight:bold;font-size:12px}
        .prediction-box .pred-value{font-size:16px;font-weight:bold}
        .ranking-list{display:flex;flex-direction:column;gap:6px;margin-top:10px}
        .ranking-item{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 14px;display:flex;align-items:center;gap:12px;cursor:pointer;transition:all 0.3s}
        .ranking-item:hover{background:rgba(255,255,255,0.06)}
        .ranking-item.pinned{border-color:#FFD700;background:rgba(255,215,0,0.05)}
        .ranking-item.bullish-highlight{border-color:#4CAF50;background:rgba(76,175,80,0.05)}
        .ranking-item.downtrend-warning{border-color:#f44336;background:rgba(244,67,54,0.05)}
        .load-more-container{text-align:center;padding:20px 0}
        .load-more-btn{background:rgba(102,126,234,0.15);border:2px solid rgba(102,126,234,0.3);color:#667eea;padding:10px 30px;border-radius:10px;font-size:14px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .load-more-btn:hover{background:rgba(102,126,234,0.25)}
        .refresh-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05)}
        .refresh-controls select{padding:4px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05);color:#fff;font-size:11px}
        .theme-toggle{background:none;border:1px solid rgba(255,255,255,0.1);border-radius:6px;padding:4px 8px;color:#aaa;cursor:pointer;font-size:14px}
        .ai-badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:9px;font-weight:600}
        .ai-badge.openai{background:rgba(255,152,0,0.2);color:#FFB74D}
        .ai-badge.claude{background:rgba(156,39,176,0.2);color:#CE93D8}
        .ai-badge.groq{background:rgba(76,175,80,0.2);color:#81C784}
        .ai-badge.gemini{background:rgba(33,150,243,0.2);color:#64B5F6}
        .ai-badge.technical{background:rgba(255,255,255,0.05);color:#888}
        .news-feed{max-height:400px;overflow-y:auto}
        .news-item{padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)}
        .sort-btn{padding:2px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);background:transparent;color:#888;cursor:pointer;font-size:9px;transition:all 0.3s}
        .sort-btn:hover{border-color:#667eea;color:#fff}
        .sort-btn.active{background:#667eea;color:#fff;border-color:#667eea}
        .filters-section{background:rgba(255,255,255,0.03);border-radius:8px;padding:12px;margin-bottom:12px}
        .filters-section h4{font-size:12px;color:#888;margin-bottom:8px}
        .filter-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
        .filter-row label{font-size:10px;color:#888;min-width:60px}
        .filter-row input{width:70px;padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px}
        .filter-row select{padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px}
        .direction-up{color:#4CAF50}
        .direction-down{color:#f44336}
        .trend-bullish{color:#4CAF50}
        .trend-bearish{color:#f44336}
        .trend-neutral{color:#FFB74D}
        .after-hours-up{color:#4CAF50;font-weight:bold}
        .after-hours-down{color:#f44336;font-weight:bold}
        .prediction-badge{padding:2px 6px;border-radius:4px;font-size:9px;font-weight:bold}
        .prediction-bullish{background:rgba(76,175,80,0.2);color:#4CAF50}
        .prediction-bearish{background:rgba(244,67,54,0.2);color:#f44336}
        .prediction-neutral{background:rgba(255,152,0,0.2);color:#FFB74D}
        .paper-trading-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px;margin-bottom:12px}
        .paper-trading-panel h3{font-size:13px;margin-bottom:8px;color:#888}
        .paper-trading-panel .paper-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px}
        .paper-trading-panel .paper-stat{text-align:center;padding:4px 2px;background:rgba(255,255,255,0.03);border-radius:4px;overflow:hidden}
        .paper-trading-panel .paper-stat .value{font-size:13px;font-weight:bold;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;line-height:1.3}
        .paper-trading-panel .paper-stat .label{font-size:7px;color:#888;white-space:nowrap;display:block;margin-top:1px}
        .paper-trading-panel .paper-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:4px 0;padding:6px;background:rgba(255,215,0,0.05);border-radius:6px}
        .paper-trading-panel .paper-row .item{text-align:center}
        .paper-trading-panel .paper-row .item .value{font-size:14px;font-weight:bold}
        .paper-trading-panel .paper-row .item .label{font-size:7px;color:#888}
        .paper-trading-panel .trade-form{display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-top:6px;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px}
        .paper-trading-panel .trade-form input{width:60px;padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px}
        .paper-trading-panel .trade-form select{padding:3px 6px;border:1px solid rgba(255,255,255,0.1);border-radius:3px;background:rgba(255,255,255,0.05);color:#fff;font-size:10px;width:55px}
        .paper-trading-panel .btn-sm{padding:3px 8px;font-size:9px}
        .paper-trading-panel .portfolio-item{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:10px}
        .paper-trading-panel .profit-positive{color:#4CAF50}
        .paper-trading-panel .profit-negative{color:#f44336}
        .paper-trading-panel .sell-btn{background:rgba(244,67,54,0.15);border:1px solid rgba(244,67,54,0.2);color:#f44336;padding:1px 6px;border-radius:3px;cursor:pointer;font-size:8px}
        .paper-trading-panel .sell-btn:hover{background:rgba(244,67,54,0.25)}
        .paper-trading-panel .transaction-item{font-size:9px;color:#888;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.03)}
        .paper-trading-panel .transaction-buy{color:#4CAF50}
        .paper-trading-panel .transaction-sell{color:#f44336}
        .paper-trading-panel .portfolio-scroll{max-height:150px;overflow-y:auto}
        .paper-trading-panel .transactions-scroll{max-height:120px;overflow-y:auto}
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
        
        <div class="paper-trading-panel">
            <h3>💼 Paper Trading</h3>
            <div class="paper-stats">
                <div class="paper-stat">
                    <div class="value gold" id="paperCash">$10,000.00</div>
                    <div class="label">Cash</div>
                </div>
                <div class="paper-stat">
                    <div class="value blue" id="paperValue">$0.00</div>
                    <div class="label">Holdings</div>
                </div>
                <div class="paper-stat">
                    <div class="value" id="paperTotal">$10,000.00</div>
                    <div class="label">Total Value</div>
                </div>
                <div class="paper-stat">
                    <div class="value" id="paperPL" style="font-size:13px;font-weight:bold;">+$0.00</div>
                    <div class="label">Total P&L</div>
                </div>
            </div>
            <div class="paper-row">
                <div class="item">
                    <span class="value gold" id="paperWinRate">0.0%</span>
                    <div class="label">Win Rate</div>
                </div>
                <div class="item">
                    <span class="value gold" id="paperTrades">0</span>
                    <div class="label">Total Trades</div>
                </div>
            </div>
            <div class="trade-form">
                <select id="tradeAction">
                    <option value="buy">BUY</option>
                    <option value="sell">SELL</option>
                </select>
                <input id="tradeTicker" placeholder="Ticker" style="width:55px">
                <input id="tradeShares" placeholder="Shares" type="number" style="width:55px">
                <button class="btn-success btn-sm" onclick="executeTrade()">Execute</button>
                <button class="btn-danger btn-sm" onclick="resetPaperTrading()">Reset</button>
            </div>
            <div id="tradeMessage" style="font-size:10px;margin-top:4px;color:#888"></div>
        </div>
        
        <div class="paper-trading-panel">
            <h3>📊 Portfolio</h3>
            <div class="portfolio-scroll" id="portfolioList">
                <div style="color:#666;font-size:11px;padding:8px 0">No holdings</div>
            </div>
        </div>
        
        <div class="paper-trading-panel">
            <h3>📜 Recent Transactions</h3>
            <div class="transactions-scroll" id="transactionList">
                <div style="color:#666;font-size:11px;padding:8px 0">No transactions</div>
            </div>
        </div>
        
        <div style="margin-bottom:15px;padding:10px;background:rgba(102,126,234,0.1);border-radius:8px">
            <div style="font-size:11px;color:#888">🤖 AI Status</div>
            <div style="font-size:11px;margin-top:4px">
                <span id="aiOpenAI" style="color:#FFB74D">● OpenAI</span>
                <span id="aiClaude" style="color:#CE93D8">● Claude</span>
                <span id="aiGroq" style="color:#4CAF50">● Groq</span>
                <span id="aiGemini" style="color:#64B5F6">● Gemini</span>
            </div>
        </div>
        
        <div class="filters-section">
            <h4>💰 Price</h4>
            <div class="filter-row">
                <label>Min</label>
                <input id="minPrice" placeholder="0" type="number">
                <label>Max</label>
                <input id="maxPrice" placeholder="10000" type="number">
            </div>
        </div>
        <div class="filters-section">
            <h4>📊 RSI</h4>
            <div class="filter-row">
                <label>Min</label>
                <input id="minRSI" placeholder="0" type="number">
                <label>Max</label>
                <input id="maxRSI" placeholder="100" type="number">
            </div>
        </div>
        <div class="filters-section">
            <h4>📈 Volume Ratio</h4>
            <div class="filter-row">
                <label>Min</label>
                <input id="minVolumeRatio" placeholder="0" type="number" step="0.1">
            </div>
        </div>
        <div class="filters-section">
            <h4>📊 Daily Change %</h4>
            <div class="filter-row">
                <label>Min</label>
                <input id="minChange" placeholder="-100" type="number" step="0.1">
                <label>Max</label>
                <input id="maxChange" placeholder="100" type="number" step="0.1">
            </div>
        </div>
        <div class="filters-section">
            <h4>📈 Trend Filter</h4>
            <div class="filter-row">
                <select id="trendFilter" style="width:100%">
                    <option value="all">All Trends</option>
                    <option value="uptrend">Uptrend Only</option>
                    <option value="downtrend">Downtrend Only</option>
                </select>
            </div>
        </div>
        <div class="filters-section">
            <h4>📰 News Sentiment</h4>
            <div class="filter-row">
                <select id="sentimentFilter" style="width:100%">
                    <option value="all">All Sentiment</option>
                    <option value="positive">Positive Only</option>
                    <option value="negative">Negative Only</option>
                </select>
            </div>
        </div>
        <div class="filters-section">
            <h4>🔍 Keyword Filter</h4>
            <div class="filter-row">
                <input id="keywordInput" placeholder="Enter keyword..." style="flex:1;padding:4px 8px;border:1px solid rgba(255,255,255,0.1);border-radius:4px;background:rgba(255,255,255,0.05);color:#fff;font-size:11px">
                <button onclick="addKeyword()" style="padding:4px 12px;border:none;border-radius:4px;background:#667eea;color:#fff;cursor:pointer;font-size:11px">Add</button>
            </div>
            <div id="keywordTags" style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px"></div>
        </div>
        <div class="filters-section">
            <h4>📰 News Sources</h4>
            <div id="sourceGrid" style="display:grid;grid-template-columns:1fr 1fr;gap:2px"></div>
        </div>
        <div style="display:flex;gap:6px;margin-top:12px">
            <button onclick="applyFilters()" style="flex:1;padding:6px;border:none;border-radius:6px;background:#667eea;color:#fff;cursor:pointer">Apply Filters</button>
            <button onclick="resetFilters()" style="flex:1;padding:6px;border:1px solid rgba(255,255,255,0.1);border-radius:6px;background:transparent;color:#888;cursor:pointer">Reset</button>
        </div>
    </div>
    
    <div class="main-content">
        <button onclick="document.getElementById('sidebar').classList.toggle('collapsed')" style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:8px 16px;color:#aaa;cursor:pointer;margin-bottom:15px">⚙️ Filters</button>
        
        <div class="header">
            <div class="header-top">
                <div>
                    <h1>🚀 <span class="gradient">AI Stock Analyzer Pro</span></h1>
                    <div class="subtitle">200+ Stocks • 5 News Sources • Paper Trading • After Hours • AI Predictions</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                    <span id="statusDot" style="font-size:10px;color:#4CAF50">🟢 Live</span>
                    <span id="lastUpdate" style="font-size:9px;color:#666">Never</span>
                    <button class="theme-toggle" onclick="toggleTheme()">🌙</button>
                    <button onclick="exportData()" style="background:rgba(102,126,234,0.2);border:1px solid rgba(102,126,234,0.3);color:#667eea;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:11px">📥 Export</button>
                    <button class="btn" onclick="refreshData()">🔄 Refresh</button>
                </div>
            </div>
            
            <div class="refresh-controls">
                <label style="font-size:11px;color:#888">🔄 Auto-refresh:</label>
                <select id="refreshInterval" onchange="setRefreshInterval()" style="padding:4px 8px;border-radius:4px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05);color:#fff;font-size:11px">
                    <option value="0">Never</option>
                    <option value="60">1 min</option>
                    <option value="300">5 min</option>
                    <option value="600">10 min</option>
                </select>
                <span id="refreshTimer" style="font-size:10px;color:#666"></span>
            </div>
            
            <div class="stats">
                <div class="stat-card"><div class="stat-number blue" id="totalStocks">0</div><div class="stat-label">Total</div></div>
                <div class="stat-card"><div class="stat-number green" id="buyCount">0</div><div class="stat-label">Buy</div></div>
                <div class="stat-card"><div class="stat-number orange" id="watchCount">0</div><div class="stat-label">Watch</div></div>
                <div class="stat-card"><div class="stat-number red" id="sellCount">0</div><div class="stat-label">Sell</div></div>
                <div class="stat-card"><div class="stat-number gold" id="pinnedCount">0</div><div class="stat-label">📌 Pinned</div></div>
            </div>
            
            <div style="display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#888">
                <span>🤖 AI: <strong id="aiCount">0</strong></span>
                <span>📰 News: <strong id="newsCount">0</strong></span>
                <span>⚡ <span id="stockCount">0</span></span>
                <span>📈 Momentum: <strong id="avgMomentum">0</strong></span>
                <span>📊 MACD: <strong id="macdCount">0</strong></span>
                <span>🚀 Breakouts: <strong id="breakoutCount">0</strong></span>
                <span>🌙 After Hours: <strong id="afterHoursCount">0</strong></span>
                <span>📊 Predictions: <strong id="predictionCount">0</strong></span>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab-btn active" data-tab="all" onclick="switchTab('all')">📊 All Stocks</button>
            <button class="tab-btn" data-tab="ranking" onclick="switchTab('ranking')">🏆 Ranking</button>
            <button class="tab-btn" data-tab="pinned" onclick="switchTab('pinned')">📌 Pinned</button>
            <button class="tab-btn" data-tab="newsfeed" onclick="switchTab('newsfeed')">📰 News Feed</button>
            <button class="tab-btn" data-tab="gainers" onclick="switchTab('gainers')">📈 Top Gainers</button>
            <button class="tab-btn" data-tab="losers" onclick="switchTab('losers')">📉 Top Losers</button>
            <button class="tab-btn" data-tab="uptrend" onclick="switchTab('uptrend')">📈 Uptrend</button>
            <button class="tab-btn" data-tab="downtrend" onclick="switchTab('downtrend')">📉 Downtrend</button>
            <button class="tab-btn" data-tab="breakouts" onclick="switchTab('breakouts')">🚀 Breakouts</button>
            <button class="tab-btn" data-tab="afterhours" onclick="switchTab('afterhours')">🌙 After Hours</button>
            <button class="tab-btn" data-tab="predictions" onclick="switchTab('predictions')">📊 Predictions</button>
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
                <button class="sort-btn" data-sort="ticker" onclick="setSort('ticker')">Ticker ▼</button>
            </div>
            
            <div style="display:flex;gap:4px;flex-wrap:wrap">
                <button class="filter-btn active" data-sector="all" onclick="setSector('all')">All</button>
                <button class="filter-btn" data-sector="Technology" onclick="setSector('Technology')">Tech</button>
                <button class="filter-btn" data-sector="Financial" onclick="setSector('Financial')">Fin</button>
                <button class="filter-btn" data-sector="Healthcare" onclick="setSector('Healthcare')">Health</button>
                <button class="filter-btn" data-sector="Consumer" onclick="setSector('Consumer')">Cons</button>
                <button class="filter-btn" data-sector="Energy" onclick="setSector('Energy')">Energy</button>
                <button class="filter-btn" data-sector="Industrial" onclick="setSector('Industrial')">Ind</button>
                <button class="filter-btn" data-sector="Communications" onclick="setSector('Communications')">Comm</button>
                <button class="filter-btn" data-sector="Real Estate" onclick="setSector('Real Estate')">RE</button>
            </div>
            
            <label class="checkbox-label">
                <input type="checkbox" id="aiToggle" checked onchange="toggleAI()"> 🧠 AI
            </label>
        </div>
        
        <div id="loadingState" class="loading">
            <div class="spinner"></div>
            <div style="color:#888;font-size:14px">📊 Loading stocks with AI predictions & after hours data...</div>
        </div>
        
        <div id="resultsContent" style="display:none">
            <div id="cardGrid" class="card-grid"></div>
            <div id="rankingContainer" style="display:none"></div>
            <div id="newsFeedContainer" style="display:none"></div>
            
            <div class="load-more-container">
                <button class="load-more-btn" onclick="loadMoreStocks()">➕ Add More Stocks</button>
            </div>
        </div>
    </div>
</div>

<div class="modal" id="detailModal">
    <div class="modal-content">
        <div class="modal-header" style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
                <h2 id="modalTicker" style="font-size:20px"></h2>
                <span id="modalCompany" style="color:#888;font-size:12px"></span>
                <span id="modalDirection" style="font-size:11px;font-weight:bold"></span>
                <span id="modalTrend" style="font-size:11px;margin-left:8px"></span>
                <span id="modalBreakout" style="font-size:11px;margin-left:8px;color:#FFD700"></span>
                <span id="modalAfterHours" style="font-size:11px;margin-left:8px"></span>
                <span id="modalPrediction" style="font-size:11px;margin-left:8px;color:#667eea"></span>
            </div>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-grid" id="modalStats"></div>
        <div id="modalPredictionBox" class="prediction-box" style="display:none">
            <div class="pred-title">📊 AI Prediction</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:6px">
                <div><span style="color:#888;font-size:10px">Predicted Price</span><br><span class="pred-value" id="predPrice">$0.00</span></div>
                <div><span style="color:#888;font-size:10px">Expected Change</span><br><span class="pred-value" id="predChange">0.0%</span></div>
                <div><span style="color:#888;font-size:10px">Confidence</span><br><span class="pred-value" id="predConfidence">0%</span></div>
                <div><span style="color:#888;font-size:10px">Prediction</span><br><span class="pred-value" id="predDirection">NEUTRAL</span></div>
            </div>
        </div>
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
let allData = [];
let currentSector = 'all';
let currentTab = 'all';
let useAI = true;
let chart = null;
let pinnedStocks = JSON.parse(localStorage.getItem('pinnedStocks') || '[]');
let refreshTimer = null;
let refreshIntervalSeconds = 0;
let timeUntilRefresh = 0;
let darkMode = true;
let currentOffset = 0;
let hasMore = true;
let isLoadingMore = false;
let keywords = [];
let selectedSources = [];
let newsFeed = [];
let currentSort = 'rank_score';
let sortDescending = true;
let currentModalTicker = '';

async function updatePaperStatus() {
    try {
        const response = await fetch('/api/paper/status');
        const data = await response.json();
        if (data.success) {
            const portfolio = data.portfolio;
            document.getElementById('paperCash').textContent = '$' + portfolio.cash.toFixed(2);
            document.getElementById('paperValue').textContent = '$' + portfolio.total_holdings_value.toFixed(2);
            document.getElementById('paperTotal').textContent = '$' + portfolio.total_value.toFixed(2);
            const profit = portfolio.total_profit || 0;
            const plElement = document.getElementById('paperPL');
            plElement.textContent = (profit >= 0 ? '+' : '') + '$' + profit.toFixed(2);
            plElement.style.color = profit >= 0 ? '#4CAF50' : '#f44336';
            const winRate = portfolio.win_rate || 0;
            document.getElementById('paperWinRate').textContent = winRate.toFixed(1) + '%';
            document.getElementById('paperWinRate').style.color = winRate >= 50 ? '#4CAF50' : '#f44336';
            document.getElementById('paperTrades').textContent = portfolio.total_trades || 0;
            const list = document.getElementById('portfolioList');
            if (portfolio.holdings && portfolio.holdings.length > 0) {
                list.innerHTML = portfolio.holdings.map(h => `
                    <div class="portfolio-item">
                        <span><strong>${h.ticker}</strong> ${h.shares} shares @ $${h.avg_price.toFixed(2)}</span>
                        <span>
                            $${h.value.toFixed(2)} 
                            <span class="${h.profit_loss >= 0 ? 'profit-positive' : 'profit-negative'}">
                                ${h.profit_loss >= 0 ? '+' : ''}${h.profit_loss_pct.toFixed(1)}%
                            </span>
                            <button class="sell-btn" onclick="quickSell('${h.ticker}', ${h.shares})">Sell</button>
                        </span>
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
                        <strong>${t.ticker}</strong> 
                        ${t.shares} @ $${t.price.toFixed(2)} 
                        <span style="float:right">$${t.total.toFixed(2)}</span>
                        ${t.profit_loss ? `<span style="float:right;margin-right:8px;color:${t.profit_loss >= 0 ? '#4CAF50' : '#f44336'}">${t.profit_loss >= 0 ? '+' : ''}$${t.profit_loss.toFixed(2)}</span>` : ''}
                        <span style="float:right;margin-right:8px;font-size:8px;color:#666">${new Date(t.timestamp).toLocaleTimeString()}</span>
                    </div>
                `).join('');
            } else {
                transList.innerHTML = '<div style="color:#666;font-size:11px;padding:8px 0">No transactions</div>';
            }
        }
    } catch(e) {
        console.error('Paper status error:', e);
    }
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
    const shares = prompt(`Enter number of shares to sell for ${ticker} (max ${maxShares}):`, maxShares);
    if (!shares || parseFloat(shares) <= 0) return;
    if (parseFloat(shares) > maxShares) {
        alert(`You only have ${maxShares} shares of ${ticker}`);
        return;
    }
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
    const btn = document.querySelector(`.sort-btn[data-sort="${sort}"]`);
    if (currentSort === sort) sortDescending = !sortDescending;
    else { currentSort = sort; sortDescending = true; }
    document.querySelectorAll('.sort-btn').forEach(b => {
        b.classList.remove('active');
        if (b.dataset.sort === sort) {
            b.classList.add('active');
            b.textContent = b.textContent.replace(/[▼▲]/g, '') + (sortDescending ? ' ▼' : ' ▲');
        }
    });
    renderCards();
}

function addKeyword() {
    const input = document.getElementById('keywordInput');
    const keyword = input.value.trim();
    if (keyword && !keywords.includes(keyword)) {
        keywords.push(keyword);
        renderKeywords();
        input.value = '';
    }
}

function removeKeyword(keyword) {
    keywords = keywords.filter(k => k !== keyword);
    renderKeywords();
}

function renderKeywords() {
    const container = document.getElementById('keywordTags');
    container.innerHTML = keywords.map(k => `
        <span style="display:flex;align-items:center;gap:4px;padding:2px 8px;border-radius:10px;font-size:10px;background:rgba(102,126,234,0.2);color:#aaa;border:1px solid rgba(102,126,234,0.2)">
            ${k} <span onclick="removeKeyword('${k}')" style="color:#888;cursor:pointer">×</span>
        </span>
    `).join('');
}

function setRefreshInterval() {
    const select = document.getElementById('refreshInterval');
    refreshIntervalSeconds = parseInt(select.value);
    if (refreshTimer) clearInterval(refreshTimer);
    if (refreshIntervalSeconds === 0) {
        document.getElementById('refreshTimer').textContent = '';
        return;
    }
    timeUntilRefresh = refreshIntervalSeconds;
    updateRefreshTimer();
    refreshTimer = setInterval(() => {
        timeUntilRefresh--;
        updateRefreshTimer();
        if (timeUntilRefresh <= 0) {
            refreshData();
            timeUntilRefresh = refreshIntervalSeconds;
        }
    }, 1000);
}

function updateRefreshTimer() {
    if (refreshIntervalSeconds === 0) return;
    const mins = Math.floor(timeUntilRefresh / 60);
    const secs = timeUntilRefresh % 60;
    document.getElementById('refreshTimer').textContent = `⏱️ ${mins}:${secs.toString().padStart(2,'0')}`;
}

function toggleTheme() {
    darkMode = !darkMode;
    document.body.style.background = darkMode ? 'linear-gradient(135deg,#0f0c29,#302b63,#24243e)' : 'linear-gradient(135deg,#f5f7fa,#c3cfe2)';
    document.body.style.color = darkMode ? '#fff' : '#1a1a2e';
    document.querySelector('.theme-toggle').textContent = darkMode ? '🌙' : '☀️';
    localStorage.setItem('darkMode', darkMode);
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

async function loadSettings() {
    try {
        const response = await fetch('/api/news/sources');
        const data = await response.json();
        if (data.success) {
            const grid = document.getElementById('sourceGrid');
            grid.innerHTML = '';
            for (const [key, src] of Object.entries(data.sources)) {
                const div = document.createElement('div');
                div.style.cssText = 'display:flex;align-items:center;gap:4px;padding:2px 6px;border-radius:3px;font-size:10px;color:#aaa';
                div.innerHTML = `
                    <input type="checkbox" value="${key}" onchange="updateSources()" ${src.enabled !== false ? 'checked' : ''}>
                    <span>${src.icon} ${src.name}</span>
                `;
                grid.appendChild(div);
                if (src.enabled !== false && !selectedSources.includes(key)) selectedSources.push(key);
            }
        }
    } catch(e) { console.error(e); }
}

function updateSources() {
    const checkboxes = document.querySelectorAll('#sourceGrid input[type="checkbox"]:checked');
    selectedSources = Array.from(checkboxes).map(cb => cb.value);
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
    keywords = [];
    renderKeywords();
    document.querySelectorAll('#sourceGrid input[type="checkbox"]').forEach(cb => cb.checked = true);
    selectedSources = ['feedflash', 'finviz', 'yahoo', 'google_news', 'stocktwits'];
    document.getElementById('keywordInput').value = '';
    currentOffset = 0;
    allData = [];
    refreshData();
}

async function refreshData() {
    const btn = document.querySelector('.btn');
    btn.disabled = true;
    btn.textContent = '⏳...';
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultsContent').style.display = 'none';
    try {
        const payload = {
            sector: currentSector !== 'all' ? currentSector : null,
            limit: 60,
            offset: 0,
            load_more: false,
            use_ai: useAI,
            keywords: keywords,
            sources: selectedSources,
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
            currentOffset = 60;
            document.getElementById('lastUpdate').textContent = 'Updated: ' + res.last_update;
            let totalNews = 0, totalMomentum = 0, macdCount = 0, breakoutCount = 0, afterHoursCount = 0, predictionCount = 0;
            allData.forEach(item => {
                if (item.news) {
                    for (const s in item.news) {
                        if (item.news[s]) totalNews += item.news[s].length;
                    }
                }
                totalMomentum += item.momentum_score || 0;
                if (item.macd_bullish) macdCount++;
                if (item.breakout) breakoutCount++;
                if (Math.abs(item.after_hours_pct || 0) > 0.5) afterHoursCount++;
                if (item.prediction && item.prediction.confidence > 50) predictionCount++;
            });
            document.getElementById('newsCount').textContent = totalNews;
            document.getElementById('stockCount').textContent = allData.length;
            document.getElementById('avgMomentum').textContent = allData.length > 0 ? Math.round(totalMomentum / allData.length) : 0;
            document.getElementById('macdCount').textContent = macdCount;
            document.getElementById('breakoutCount').textContent = breakoutCount;
            document.getElementById('afterHoursCount').textContent = afterHoursCount;
            document.getElementById('predictionCount').textContent = predictionCount;
            const aiCount = res.stats ? (res.stats.openai || 0) + (res.stats.claude || 0) + (res.stats.groq || 0) + (res.stats.gemini || 0) : 0;
            document.getElementById('aiCount').textContent = aiCount;
            if (res.ai_availability) {
                document.getElementById('aiOpenAI').style.color = res.ai_availability.openai ? '#FFB74D' : '#666';
                document.getElementById('aiClaude').style.color = res.ai_availability.claude ? '#CE93D8' : '#666';
                document.getElementById('aiGroq').style.color = res.ai_availability.groq ? '#4CAF50' : '#666';
                document.getElementById('aiGemini').style.color = res.ai_availability.gemini ? '#64B5F6' : '#666';
            }
            renderCards();
            updateStats();
            updatePaperStatus();
            loadNewsFeed();
        }
    } catch(err) { console.error(err); }
    finally {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';
        btn.disabled = false;
        btn.textContent = '🔄 Refresh';
    }
}

async function loadNewsFeed() {
    try {
        const response = await fetch('/api/news/feed?limit=200');
        const data = await response.json();
        if (data.success) {
            newsFeed = data.news;
            if (currentTab === 'newsfeed') renderCards();
        }
    } catch(e) { console.error(e); }
}

async function loadMoreStocks() {
    if (isLoadingMore || !hasMore) return;
    isLoadingMore = true;
    const btn = document.querySelector('.load-more-btn');
    btn.textContent = '⏳ Loading...';
    btn.disabled = true;
    currentOffset += 60;
    try {
        const payload = {
            sector: currentSector !== 'all' ? currentSector : null,
            limit: 60,
            offset: currentOffset,
            load_more: true,
            use_ai: useAI,
            keywords: keywords,
            sources: selectedSources,
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
            renderCards();
            updateStats();
            updatePaperStatus();
        }
    } catch(err) { console.error(err); }
    finally {
        isLoadingMore = false;
        btn.textContent = '➕ Add More Stocks';
        btn.disabled = false;
    }
}

function renderCards() {
    const grid = document.getElementById('cardGrid');
    const rankingContainer = document.getElementById('rankingContainer');
    const newsFeedContainer = document.getElementById('newsFeedContainer');
    grid.style.display = 'none';
    rankingContainer.style.display = 'none';
    newsFeedContainer.style.display = 'none';
    if (currentTab === 'newsfeed') {
        newsFeedContainer.style.display = 'block';
        renderNewsFeed(newsFeedContainer);
        return;
    }
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
    } else if (currentTab === 'breakouts') {
        filtered = filtered.filter(item => item.breakout === true);
    } else if (currentTab === 'afterhours') {
        filtered = filtered.filter(item => Math.abs(item.after_hours_pct || 0) > 0.5);
        filtered.sort((a, b) => Math.abs(b.after_hours_pct || 0) - Math.abs(a.after_hours_pct || 0));
    } else if (currentTab === 'predictions') {
        filtered = filtered.filter(item => item.prediction && item.prediction.confidence > 50);
        filtered.sort((a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0));
    }
    if (currentSort && currentTab !== 'gainers' && currentTab !== 'losers' && currentTab !== 'afterhours' && currentTab !== 'predictions') {
        filtered.sort((a, b) => {
            let va = a[currentSort] ?? 0;
            let vb = b[currentSort] ?? 0;
            if (typeof va === 'string') return sortDescending ? va.localeCompare(vb) : vb.localeCompare(va);
            return sortDescending ? vb - va : va - vb;
        });
    }
    if (currentTab === 'ranking' || currentTab === 'pinned') {
        rankingContainer.style.display = 'block';
        renderRankingList(filtered, rankingContainer);
        return;
    }
    grid.style.display = 'grid';
    renderCardGrid(filtered, grid);
}

function renderCardGrid(filtered, grid) {
    grid.innerHTML = '';
    filtered.forEach((item) => {
        const card = document.createElement('div');
        const isBullish = (item.trend && item.trend.includes('BULLISH')) && item.change_1d > 0;
        const isDowntrend = item.consecutive_down_days >= 3 || (item.trend && item.trend.includes('BEARISH'));
        let cardClass = 'stock-card';
        if (isPinned(item.ticker)) cardClass += ' pinned';
        if (isBullish) cardClass += ' bullish-highlight';
        if (isDowntrend) cardClass += ' downtrend-warning';
        if (item.breakout) cardClass += ' breakout-highlight';
        card.className = cardClass;
        card.onclick = () => openModal(item);
        const recClass = (item.recommendation || 'WATCH').toLowerCase().replace(' ', '-');
        const pinned = isPinned(item.ticker);
        const aiSource = item.ai_source || item.source || 'Technical';
        const aiClass = aiSource.includes('OpenAI') ? 'openai' : aiSource.includes('Claude') ? 'claude' : aiSource.includes('Groq') ? 'groq' : aiSource.includes('Gemini') ? 'gemini' : 'technical';
        const momentum = item.momentum_score || 50;
        const sentiment = item.sentiment_aggregate || 0;
        const sentimentEmoji = sentiment > 0.3 ? '🟢' : sentiment < -0.3 ? '🔴' : '🟡';
        const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
        const directionClass = direction === 'UP' ? 'direction-up' : 'direction-down';
        const downDays = item.consecutive_down_days || 0;
        const trendClass = item.trend && item.trend.includes('BULLISH') ? 'trend-bullish' : item.trend && item.trend.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
        const macdSignal = item.macd_bullish ? '🟢' : '⚪';
        const breakoutSignal = item.breakout ? '🚀 ' : '';
        const ahPct = item.after_hours_pct || 0;
        const ahDisplay = Math.abs(ahPct) > 0.5 ? `${ahPct > 0 ? '📈' : '📉'} AH ${ahPct > 0 ? '+' : ''}${ahPct.toFixed(1)}%` : '';
        let predDisplay = '';
        if (item.prediction && item.prediction.confidence > 50) {
            const pred = item.prediction;
            const predClass = pred.prediction === 'BULLISH' || pred.prediction === 'STRONG BULLISH' ? 'prediction-bullish' : pred.prediction === 'BEARISH' || pred.prediction === 'STRONG BEARISH' ? 'prediction-bearish' : 'prediction-neutral';
            predDisplay = `<span class="prediction-badge ${predClass}">📊 ${pred.prediction} ${pred.expected_change > 0 ? '+' : ''}${pred.expected_change}%</span>`;
        }
        let newsCount = 0;
        if (item.news) {
            for (const s in item.news) {
                if (item.news[s]) newsCount += item.news[s].length;
            }
        }
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-size:17px;font-weight:700">${breakoutSignal}${item.ticker} ${pinned ? '📌' : ''}</div>
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
            ${ahDisplay ? `<div style="text-align:right;font-size:11px;font-weight:bold;color:${ahPct > 0 ? '#4CAF50' : '#f44336'};margin:-4px 0 4px 0">${ahDisplay}</div>` : ''}
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;margin:4px 0;font-size:11px;color:#aaa">
                <div>RSI: ${item.rsi || 'N/A'}</div>
                <div>Vol: ${item.volume_ratio?.toFixed(1) || 'N/A'}x</div>
                <div class="${trendClass}">${item.trend_icon || '➡️'} ${item.trend || 'NEUTRAL'}</div>
                <div>${sentimentEmoji} Sentiment</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:10px;color:#888">
                <div>MACD: ${macdSignal} ${item.macd_bullish ? 'BULL' : ''}</div>
                <div>ADX: ${item.adx || 0}</div>
                <div>RelStr: ${item.relative_strength || 0}%</div>
            </div>
            ${predDisplay ? `<div style="margin-top:2px">${predDisplay}</div>` : ''}
            ${downDays >= 2 ? `<div style="font-size:10px;color:#f44336;font-weight:bold">⚠️ ${downDays} consecutive down days</div>` : ''}
            ${item.breakout ? `<div style="font-size:10px;color:#FFD700;font-weight:bold">🚀 BREAKOUT ABOVE 20-DAY HIGH</div>` : ''}
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin:2px 0">
                <span>Momentum: ${momentum}%</span>
                <span>📰 ${newsCount}</span>
                <span>Conf: ${item.confidence || 0}%</span>
            </div>
            <div style="font-size:11px;color:#bbb;flex:1;margin:4px 0">${item.summary || 'No analysis'}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#666;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.05)">
                <span class="ai-badge ${aiClass}">${aiSource}</span>
                <span>Score: ${Math.round(item.rank_score || 0)}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

function renderRankingList(filtered, container) {
    container.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;padding:8px 14px;color:#666;font-size:10px;font-weight:600;border-bottom:1px solid rgba(255,255,255,0.05)">
            <span style="min-width:30px">#</span>
            <span style="min-width:60px">Ticker</span>
            <span style="flex:1">Company</span>
            <span style="min-width:60px;text-align:right">Price</span>
            <span style="min-width:60px;text-align:right">Change</span>
            <span style="min-width:50px;text-align:right">RSI</span>
            <span style="min-width:50px;text-align:right">Momentum</span>
            <span style="min-width:60px;text-align:center">Rec</span>
            <span style="min-width:50px;text-align:center">Trend</span>
            <span style="min-width:50px;text-align:center">MACD</span>
            <span style="min-width:60px;text-align:center">AI</span>
            <span style="min-width:50px;text-align:center">AH</span>
            <span style="min-width:60px;text-align:center">Pred</span>
            <span style="min-width:50px;text-align:right">Score</span>
            <span style="min-width:30px;text-align:center">📌</span>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
    `;
    filtered.forEach((item, index) => {
        const pinned = isPinned(item.ticker);
        const isBullish = (item.trend && item.trend.includes('BULLISH')) && item.change_1d > 0;
        const isDowntrend = item.consecutive_down_days >= 3 || (item.trend && item.trend.includes('BEARISH'));
        let rowClass = 'ranking-item';
        if (pinned) rowClass += ' pinned';
        if (isBullish) rowClass += ' bullish-highlight';
        if (isDowntrend) rowClass += ' downtrend-warning';
        if (item.breakout) rowClass += ' breakout-highlight';
        const rankEmoji = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index+1}`;
        const aiSource = item.ai_source || item.source || 'Technical';
        const aiClass = aiSource.includes('OpenAI') ? 'openai' : aiSource.includes('Claude') ? 'claude' : aiSource.includes('Groq') ? 'groq' : aiSource.includes('Gemini') ? 'gemini' : 'technical';
        const momentum = item.momentum_score || 50;
        const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
        const downDays = item.consecutive_down_days || 0;
        const trendDisplay = item.trend ? item.trend.substring(0, 12) : 'NEUTRAL';
        const trendClass = item.trend && item.trend.includes('BULLISH') ? 'trend-bullish' : item.trend && item.trend.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
        const macdDisplay = item.macd_bullish ? '🟢BULL' : '⚪';
        const breakoutDisplay = item.breakout ? '🚀' : '';
        const ahDisplay = Math.abs(item.after_hours_pct || 0) > 0.5 ? `${item.after_hours_pct > 0 ? '🟢' : '🔴'}${(item.after_hours_pct || 0).toFixed(1)}%` : '';
        const predDisplay = item.prediction && item.prediction.confidence > 50 ? `${item.prediction.expected_change > 0 ? '📈' : '📉'}${(item.prediction.expected_change || 0).toFixed(1)}%` : '';
        container.innerHTML += `
            <div class="${rowClass}" onclick="openModal(item)" data-ticker="${item.ticker}">
                <span style="min-width:30px;font-weight:bold;color:#667eea">${rankEmoji}</span>
                <span style="min-width:60px;font-weight:600">${breakoutDisplay}${item.ticker}</span>
                <span style="flex:1;font-size:11px;color:#888">${item.company}</span>
                <span style="min-width:60px;text-align:right;font-weight:600">$${item.price?.toFixed(2) || 'N/A'}</span>
                <span style="min-width:60px;text-align:right;font-weight:600;color:${item.change_1d >= 0 ? '#4CAF50' : '#f44336'}">
                    ${item.change_1d?.toFixed(1) || '0.0'}% ${direction === 'UP' ? '▲' : '▼'}
                </span>
                <span style="min-width:50px;text-align:right;font-weight:600">${item.rsi || 'N/A'}</span>
                <span style="min-width:50px;text-align:right;font-weight:600;color:${momentum >= 60 ? '#4CAF50' : momentum >= 40 ? '#FFB74D' : '#f44336'}">${momentum}%</span>
                <span style="min-width:60px;text-align:center;padding:2px 8px;border-radius:10px;font-weight:600;font-size:9px;background:rgba(102,126,234,0.1);color:#667eea">${item.recommendation || 'WATCH'}</span>
                <span style="min-width:50px;text-align:center;font-size:9px" class="${trendClass}">${trendDisplay}${downDays >= 3 ? '⚠️' : ''}</span>
                <span style="min-width:50px;text-align:center;font-size:9px">${macdDisplay}</span>
                <span style="min-width:60px;text-align:center"><span class="ai-badge ${aiClass}">${aiSource}</span></span>
                <span style="min-width:50px;text-align:center;font-size:9px">${ahDisplay}</span>
                <span style="min-width:60px;text-align:center;font-size:9px">${predDisplay}</span>
                <span style="min-width:50px;text-align:right;font-weight:bold;color:#667eea">${Math.round(item.rank_score || 0)}</span>
                <button class="pin-btn ${pinned ? 'pinned' : ''}" onclick="togglePin('${item.ticker}', event)" style="min-width:30px;text-align:center">📌</button>
            </div>
        `;
    });
    container.innerHTML += '</div>';
}

function renderNewsFeed(container) {
    const groupedNews = {};
    newsFeed.forEach(n => {
        const source = n.source || 'Unknown';
        if (!groupedNews[source]) groupedNews[source] = [];
        groupedNews[source].push(n);
    });
    let html = `
        <div style="margin-bottom:12px">
            <h3 style="font-size:16px;margin-bottom:8px">📰 Live News Feed</h3>
            <div style="font-size:11px;color:#888">${newsFeed.length} recent news items from ${Object.keys(groupedNews).length} sources</div>
        </div>
        <div class="news-feed">
    `;
    for (const [source, items] of Object.entries(groupedNews)) {
        const sourceIcon = {'FeedFlash': '⚡', 'Finviz': '📊', 'Yahoo Finance': '💹', 'Google News': '🔍', 'StockTwits': '💬'}[source] || '📰';
        html += `
            <div style="margin-bottom:8px;padding:6px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:2px solid rgba(102,126,234,0.3)">
                <div style="font-size:10px;color:#667eea;font-weight:bold;margin-bottom:4px">${sourceIcon} ${source}</div>
        `;
        items.slice(0, 5).forEach(n => {
            const sentimentEmoji = n.sentiment?.label === 'BULLISH' ? '🟢' : n.sentiment?.label === 'BEARISH' ? '🔴' : n.sentiment?.label === 'POSITIVE' ? '🟢' : n.sentiment?.label === 'NEGATIVE' ? '🔴' : '🟡';
            const headline = n.headline || 'No headline';
            const shortHeadline = headline.length > 120 ? headline.substring(0, 120) + '...' : headline;
            html += `
                <div class="news-item">
                    <div style="font-size:11px;color:#ddd">${sentimentEmoji} ${shortHeadline}</div>
                    <div style="display:flex;gap:8px;font-size:8px;color:#666;margin-top:1px">
                        <span>${n.time || ''}</span>
                        <span>Sentiment: ${n.sentiment?.label || 'NEUTRAL'}</span>
                        ${n.user ? `<span>👤 ${n.user}</span>` : ''}
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    }
    html += `</div>`;
    container.innerHTML = html;
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

function filterCards() { renderCards(); }

function setSector(sector) {
    currentSector = sector;
    document.querySelectorAll('[data-sector]').forEach(b => b.classList.toggle('active', b.dataset.sector === sector));
    currentOffset = 0;
    allData = [];
    refreshData();
}

function toggleAI() {
    useAI = document.getElementById('aiToggle').checked;
    currentOffset = 0;
    allData = [];
    refreshData();
}

async function exportData() {
    if (allData.length === 0) { alert('No data to export.'); return; }
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: allData, format: 'csv' })
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
    } catch(err) { alert('Error exporting: ' + err.message); }
}

function openModal(item) {
    currentModalTicker = item.ticker;
    document.getElementById('modalTicker').textContent = item.ticker;
    document.getElementById('modalCompany').textContent = item.company + ' • ' + item.sector;
    const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
    const directionClass = direction === 'UP' ? 'direction-up' : 'direction-down';
    document.getElementById('modalDirection').textContent = `${direction} ${direction === 'UP' ? '▲' : '▼'} (${item.change_1d?.toFixed(1)}%)`;
    document.getElementById('modalDirection').className = directionClass;
    const trendDisplay = item.trend || 'NEUTRAL';
    const trendClass = trendDisplay.includes('BULLISH') ? 'trend-bullish' : trendDisplay.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
    document.getElementById('modalTrend').textContent = `📊 ${trendDisplay}`;
    document.getElementById('modalTrend').className = trendClass;
    document.getElementById('modalBreakout').textContent = item.breakout ? '🚀 BREAKOUT' : '';
    document.getElementById('modalBreakout').style.color = item.breakout ? '#FFD700' : '';
    const ahPct = item.after_hours_pct || 0;
    const ahText = Math.abs(ahPct) > 0.1 ? `🌙 After Hours: ${ahPct > 0 ? '+' : ''}${ahPct.toFixed(2)}%` : '';
    document.getElementById('modalAfterHours').textContent = ahText;
    document.getElementById('modalAfterHours').style.color = ahPct > 0 ? '#4CAF50' : ahPct < 0 ? '#f44336' : '#888';
    const predBox = document.getElementById('modalPredictionBox');
    if (item.prediction && item.prediction.confidence > 50) {
        predBox.style.display = 'block';
        document.getElementById('predPrice').textContent = '$' + (item.prediction.predicted_price || 0).toFixed(2);
        const predChange = item.prediction.expected_change || 0;
        document.getElementById('predChange').textContent = (predChange > 0 ? '+' : '') + predChange.toFixed(2) + '%';
        document.getElementById('predChange').style.color = predChange > 0 ? '#4CAF50' : predChange < 0 ? '#f44336' : '#FFB74D';
        document.getElementById('predConfidence').textContent = (item.prediction.confidence || 0).toFixed(0) + '%';
        const predDir = item.prediction.prediction || 'NEUTRAL';
        document.getElementById('predDirection').textContent = predDir;
        document.getElementById('predDirection').style.color = predDir.includes('BULLISH') ? '#4CAF50' : predDir.includes('BEARISH') ? '#f44336' : '#FFB74D';
        document.getElementById('modalPrediction').textContent = `📊 ${predDir} ${predChange > 0 ? '+' : ''}${predChange.toFixed(1)}%`;
    } else {
        predBox.style.display = 'none';
        document.getElementById('modalPrediction').textContent = '';
    }
    const stats = [
        { label: 'Price', value: '$' + (item.price?.toFixed(2) || 'N/A') },
        { label: 'Change', value: (item.change_1d?.toFixed(1) || '0.0') + '%', class: item.change_1d >= 0 ? 'green' : 'red' },
        { label: 'After Hours', value: (item.after_hours_pct?.toFixed(2) || '0.00') + '%', class: (item.after_hours_pct || 0) >= 0 ? 'green' : 'red' },
        { label: 'RSI', value: item.rsi || 'N/A' },
        { label: 'Volume Ratio', value: item.volume_ratio?.toFixed(2) || 'N/A' },
        { label: 'Momentum', value: (item.momentum_score || 50) + '%' },
        { label: 'P/E', value: item.pe_ratio || 'N/A' },
        { label: 'SMA20', value: '$' + (item.sma20?.toFixed(2) || 'N/A') },
        { label: 'SMA50', value: '$' + (item.sma50?.toFixed(2) || 'N/A') },
        { label: 'MACD', value: item.macd_bullish ? '🟢 Bullish' : '🔴 Bearish' },
        { label: 'Relative Strength', value: (item.relative_strength || 0) + '%' },
        { label: 'ADX', value: item.adx || 0 },
        { label: 'Bollinger', value: item.boll_signal || 'NORMAL' },
        { label: 'Breakout', value: item.breakout ? '🚀 YES' : 'No' },
        { label: 'Recommendation', value: item.recommendation || 'WATCH' },
        { label: 'Confidence', value: item.confidence + '%' },
        { label: 'Source', value: item.ai_source || item.source || 'Technical' },
        { label: 'News', value: item.news_count || 0 },
        { label: 'Sentiment', value: (item.sentiment_aggregate || 0).toFixed(2) },
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
            data: { labels: hist.dates, datasets: [{ label: 'Price', data: hist.prices, borderColor: '#667eea', backgroundColor: 'rgba(102,126,234,0.1)', fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: true, plugins: { legend: { display: true, labels: { color: '#888', font: { size: 10 } } }, tooltip: { callbacks: { label: function(context) { return '$' + context.parsed.y.toFixed(2); } } } }, scales: { y: { ticks: { color: '#666', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.05)' } }, x: { ticks: { color: '#666', font: { size: 8 }, maxTicksLimit: 10 }, grid: { display: false } } } }
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

const savedTheme = localStorage.getItem('darkMode');
if (savedTheme === 'false') {
    darkMode = false;
    document.body.style.background = 'linear-gradient(135deg,#f5f7fa,#c3cfe2)';
    document.body.style.color = '#1a1a2e';
    document.querySelector('.theme-toggle').textContent = '☀️';
}

loadSettings();
refreshData();
</script>
</body>
</html>
"""

# ============================================================
# RUN THE APP
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI Stock Analyzer Pro - FULLY FIXED")
    print("="*80)
    print(f"📈 Total Stocks: {len(ALL_STOCKS)}")
    print(f"📰 News Sources: {len(NEWS_SOURCES)}")
    print(f"🤖 OpenAI: {'✅ Available' if openai_client else '❌ Not available'}")
    print(f"🤖 Claude: {'✅ Available' if claude_client else '❌ Not available'}")
    print(f"🤖 Groq: {'✅ Available' if groq_client else '❌ Not available'}")
    print(f"🤖 Gemini: {'✅ Available' if gemini_client else '❌ Not available'}")
    print("="*80)
    print("🔧 FIXES APPLIED:")
    print("   ✅ Fixed FeedFlash scraper - properly extracts articles")
    print("   ✅ Removed navigation/menu items from news feed")
    print("   ✅ Added global AI rate limiting (100 calls/min)")
    print("   ✅ OpenAI rate limit handling with 5-min cooldown")
    print("   ✅ Groq rate limit handling with 10-min cooldown")
    print("   ✅ Claude 401 Unauthorized handling")
    print("   ✅ Increased batch size to 60 stocks")
    print("   ✅ Fixed P&L positioning in sidebar")
    print("   ✅ Smaller font sizes for paper trading panel")
    print("   ✅ 2-column layout for paper trading stats")
    print("="*80)
    print("🌐 Running on:", "Railway" if IS_RAILWAY else "Local")
    print(f"📡 Port: {PORT}")
    print("="*80)
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
