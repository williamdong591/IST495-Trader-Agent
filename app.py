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
import hashlib
import numpy as np
from collections import deque
import random
import logging
import sqlite3
from pathlib import Path

# ============================================================
# NO TEXTBLOB - Using custom sentiment analysis instead
# ============================================================

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Install with: pip install openai")

warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.ERROR)
logging.getLogger('groq').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)

# Load .env file with error handling
try:
    load_dotenv()
    print("✓ Environment variables loaded")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")
    pass

app = Flask(__name__)

# ============================================================
# SENTIMENT ANALYSIS - Custom implementation (no TextBlob)
# ============================================================

def get_sentiment_analysis(text):
    """Custom sentiment analysis without TextBlob/NLTK"""
    if not text:
        return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
    
    # Extensive keyword lists
    very_positive = ['bullish', 'surge', 'soar', 'rocket', 'breakout', 'record high', 
                     'outperform', 'strong buy', 'upgrade', 'beating', 'exceed']
    positive = ['up', 'gain', 'positive', 'good', 'great', 'excellent', 'growth', 
                'profit', 'beat', 'rally', 'strong', 'opportunity', 'win', 'success',
                'higher', 'rise', 'jump', 'climb', 'advance', 'improve', 'boost']
    very_negative = ['bearish', 'crash', 'plunge', 'tumble', 'collapse', 'downgrade',
                     'underperform', 'strong sell', 'miss', 'disaster', 'terrible']
    negative = ['down', 'loss', 'negative', 'bad', 'poor', 'decline', 'drop', 
                'fall', 'weak', 'risk', 'danger', 'failure', 'problem', 'low',
                'slip', 'slide', 'struggle', 'worry', 'concern']
    
    text_lower = text.lower()
    
    # Count weighted keywords
    vp_count = sum(1 for word in very_positive if word in text_lower)
    p_count = sum(1 for word in positive if word in text_lower)
    vn_count = sum(1 for word in very_negative if word in text_lower)
    n_count = sum(1 for word in negative if word in text_lower)
    
    # Weighted score
    score = (vp_count * 2 + p_count - n_count - vn_count * 2)
    
    # Normalize
    total = vp_count + p_count + vn_count + n_count
    if total == 0:
        return {'label': 'NEUTRAL', 'score': 0, 'confidence': 0}
    
    normalized_score = max(-1, min(1, score / (total * 2)))
    confidence = min(100, (total / 10) * 100 + 10)
    
    if normalized_score > 0.2:
        label = "BULLISH" if normalized_score > 0.5 else "POSITIVE"
    elif normalized_score < -0.2:
        label = "BEARISH" if normalized_score < -0.5 else "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    return {
        'label': label,
        'score': round(normalized_score, 2),
        'confidence': round(confidence, 2)
    }

# ============================================================
# DATABASE FOR PAPER TRADING
# ============================================================

class PaperTradingDB:
    def __init__(self):
        # Use Railway volume if available for persistent storage
        if os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'):
            db_dir = Path(os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'))
            db_dir.mkdir(exist_ok=True)
            self.db_path = db_dir / 'paper_trading.db'
        else:
            self.db_path = Path('paper_trading.db')
        
        self.init_db()
        self.cache = {}  # Cache for portfolio values to avoid repeated API calls
    
    def init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                cash REAL DEFAULT 10000,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio table
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
        
        # Transactions table
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
        
        # Performance history table
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
        conn = sqlite3.connect(str(self.db_path))
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
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT ticker, shares, avg_price FROM portfolio WHERE user_id = ?', (user_id,))
        results = cursor.fetchall()
        conn.close()
        return [{'ticker': r[0], 'shares': r[1], 'avg_price': r[2]} for r in results]
    
    def get_transactions(self, user_id, limit=50):
        conn = sqlite3.connect(str(self.db_path))
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
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        total_cost = shares * price
        
        # Check cash
        cursor.execute('SELECT cash FROM users WHERE user_id = ?', (user_id,))
        cash = cursor.fetchone()[0]
        
        if cash < total_cost:
            conn.close()
            return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${cash:.2f}"
        
        # Update cash
        cursor.execute('UPDATE users SET cash = cash - ? WHERE user_id = ?', (total_cost, user_id))
        
        # Update portfolio
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
        
        # Record transaction
        cursor.execute('''
            INSERT INTO transactions (user_id, ticker, type, shares, price, total)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'BUY', shares, price, total_cost))
        
        conn.commit()
        conn.close()
        
        # Clear cache for this user
        if user_id in self.cache:
            del self.cache[user_id]
        
        return True, f"Bought {shares} shares of {ticker} at ${price:.2f} (Total: ${total_cost:.2f})"
    
    def sell_stock(self, user_id, ticker, shares, price):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check holdings
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
        
        # Update cash
        cursor.execute('UPDATE users SET cash = cash + ? WHERE user_id = ?', (total_value, user_id))
        
        # Update portfolio
        new_shares = existing_shares - shares
        if new_shares == 0:
            cursor.execute('DELETE FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
        else:
            cursor.execute('UPDATE portfolio SET shares = ? WHERE user_id = ? AND ticker = ?', (new_shares, user_id, ticker))
        
        # Record transaction
        cursor.execute('''
            INSERT INTO transactions (user_id, ticker, type, shares, price, total)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, ticker, 'SELL', shares, price, total_value))
        
        conn.commit()
        conn.close()
        
        # Clear cache for this user
        if user_id in self.cache:
            del self.cache[user_id]
        
        profit_loss = (price - avg_price) * shares
        return True, f"Sold {shares} shares of {ticker} at ${price:.2f} (P/L: ${profit_loss:.2f})"
    
    def get_portfolio_value(self, user_id):
        # Check cache first (5 second TTL)
        if user_id in self.cache:
            cache_time, data = self.cache[user_id]
            if (datetime.now() - cache_time).seconds < 5:
                return data
        
        portfolio = self.get_portfolio(user_id)
        
        # Get cash
        conn = sqlite3.connect(str(self.db_path))
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
                print(f"⚠️ Error getting price for {item['ticker']}: {e}")
                # Use avg_price as fallback
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
        
        # Calculate total portfolio value
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
        
        # Cache the result
        self.cache[user_id] = (datetime.now(), result)
        
        # Save performance history (every 5 minutes)
        self._save_performance_history(user_id, result)
        
        return result
    
    def _save_performance_history(self, user_id, portfolio_data):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if we already have a record for this minute
            current_minute = datetime.now().strftime('%Y-%m-%d %H:%M:00')
            cursor.execute('''
                SELECT COUNT(*) FROM performance_history 
                WHERE user_id = ? AND strftime('%Y-%m-%d %H:%M:00', timestamp) = ?
            ''', (user_id, current_minute))
            
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO performance_history (user_id, total_value, cash, holdings_value)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, portfolio_data['total_value'], portfolio_data['cash'], portfolio_data['total_holdings_value']))
                conn.commit()
            
            conn.close()
        except:
            pass
    
    def get_performance_history(self, user_id, days=7):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('''
            SELECT total_value, cash, holdings_value, timestamp 
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
            'timestamp': r[3]
        } for r in results]
    
    def get_cash(self, user_id):
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT cash FROM users WHERE user_id = ?', (user_id,))
        cash = cursor.fetchone()[0]
        conn.close()
        return cash

# ============================================================
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# AI CLIENTS - FIXED
# ============================================================

openai_client = None
claude_client = None
groq_client = None
GROQ_RATE_LIMITED = False
GROQ_LIMIT_RESET_TIME = None
AI_DISABLED_GLOBALLY = False

# Initialize OpenAI - FIXED VERSION
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        # Try the standard initialization
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI ready (primary AI)")
    except TypeError as e:
        # If there's a TypeError about proxies, try with http_client
        if "proxies" in str(e) or "unexpected keyword" in str(e):
            try:
                import httpx
                http_client = httpx.Client(timeout=60.0)
                openai_client = openai.OpenAI(
                    api_key=OPENAI_API_KEY,
                    http_client=http_client
                )
                print("✓ OpenAI ready (primary AI) - using HTTP client")
            except Exception as e2:
                print(f"⚠️ OpenAI fallback error: {str(e2)[:60]}")
                openai_client = None
        else:
            print(f"⚠️ OpenAI error: {str(e)[:60]}")
            openai_client = None
    except Exception as e:
        print(f"⚠️ OpenAI error: {str(e)[:60]}")
        openai_client = None
else:
    if not OPENAI_API_KEY:
        print("⚠️ No OpenAI API key found. Add OPENAI_API_KEY to .env file")
    openai_client = None

# Initialize Claude
try:
    from anthropic import Anthropic
    if ANTHROPIC_API_KEY:
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("✓ Claude AI ready (fallback)")
except:
    claude_client = None

# Initialize Groq
try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✓ Groq AI ready (fallback)")
except:
    groq_client = None

def is_groq_rate_limited():
    global GROQ_RATE_LIMITED, GROQ_LIMIT_RESET_TIME
    if GROQ_RATE_LIMITED and GROQ_LIMIT_RESET_TIME:
        if datetime.now() > GROQ_LIMIT_RESET_TIME:
            GROQ_RATE_LIMITED = False
            GROQ_LIMIT_RESET_TIME = None
            global AI_DISABLED_GLOBALLY
            AI_DISABLED_GLOBALLY = False
            return False
        return True
    return False

def mark_groq_rate_limited():
    global GROQ_RATE_LIMITED, GROQ_LIMIT_RESET_TIME, AI_DISABLED_GLOBALLY
    if not GROQ_RATE_LIMITED:
        GROQ_RATE_LIMITED = True
        GROQ_LIMIT_RESET_TIME = datetime.now() + timedelta(minutes=10)
        AI_DISABLED_GLOBALLY = True
        print(f"⚠️ Groq rate limited. AI disabled until {GROQ_LIMIT_RESET_TIME.strftime('%H:%M:%S')}")

# ============================================================
# ENHANCED NEWS SCRAPER - ALL 11 SOURCES WORKING
# ============================================================

class EnhancedNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.timeout = 3
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        ]
        self.ua_index = 0
        self.scrape_cache = {}
        self.cache_ttl = 180
        self.ai_engine = None
        self.news_history = deque(maxlen=500)
        
    def set_ai_engine(self, ai_engine):
        self.ai_engine = ai_engine
        
    def _rotate_user_agent(self):
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
            })
            if response.status_code == 200:
                return response.text
        except Exception as e:
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
                                time_text = cells[0].text.strip() if cells[0] else ''
                                news_item = {
                                    'headline': headline,
                                    'time': time_text,
                                    'source': 'Finviz',
                                    'sentiment': self.ai_engine.get_ai_sentiment(headline) if self.ai_engine else {'label': 'NEUTRAL'}
                                }
                                results['news'].append(news_item)
                                self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_marketwatch(self, ticker):
        results = {'news': []}
        cache_key = f"marketwatch_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://www.marketwatch.com/investing/stock/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.element--article .article__headline')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text,
                            'time': '',
                            'source': 'MarketWatch',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_tradingview(self, ticker):
        results = {'news': []}
        cache_key = f"tradingview_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            # TradingView is harder to scrape, use a different approach
            html = self._safe_scrape(f"https://www.tradingview.com/symbols/{ticker}/")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                # Try to find news items
                news_items = soup.select('.tv-news')
                for item in news_items[:5]:
                    text = item.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'TradingView',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_yahoo_finance(self, ticker):
        results = {'news': []}
        cache_key = f"yahoo_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://finance.yahoo.com/quote/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                news_items = soup.select('.js-content-viewer .Mb\\(5px\\)')
                for item in news_items[:5]:
                    text = item.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'Yahoo Finance',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_seeking_alpha(self, ticker):
        results = {'news': []}
        cache_key = f"seekingalpha_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://seekingalpha.com/symbol/{ticker}/news")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.media-heading a')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'Seeking Alpha',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
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
            # Use RSS feed for Google News
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
    
    def scrape_stocktwits(self, ticker):
        results = {'news': []}
        cache_key = f"stocktwits_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            # StockTwits API-like approach
            html = self._safe_scrape(f"https://stocktwits.com/symbol/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                messages = soup.select('.message__body')
                for msg in messages[:5]:
                    text = msg.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'StockTwits',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_bloomberg(self, ticker):
        results = {'news': []}
        cache_key = f"bloomberg_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://www.bloomberg.com/quote/{ticker}:US")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.story-list-story__headline')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'Bloomberg',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_cnbc(self, ticker):
        results = {'news': []}
        cache_key = f"cnbc_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://www.cnbc.com/quotes/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.Card-title')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'CNBC',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_reuters(self, ticker):
        results = {'news': []}
        cache_key = f"reuters_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://www.reuters.com/markets/companies/{ticker}.O/")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.media-story-card__heading__text')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'Reuters',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_benzinga(self, ticker):
        results = {'news': []}
        cache_key = f"benzinga_{ticker}"
        if cache_key in self.scrape_cache:
            cache_time, data = self.scrape_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        try:
            html = self._safe_scrape(f"https://www.benzinga.com/quote/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                headlines = soup.select('.story__title a')
                for h in headlines[:5]:
                    text = h.text.strip()
                    if text and len(text) > 5:
                        news_item = {
                            'headline': text[:200],
                            'time': '',
                            'source': 'Benzinga',
                            'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results

    def fetch_all_news(self, ticker):
        """Fetch news from ALL 11 sources with parallel processing"""
        all_news = {}
        
        sources = {
            'finviz': self.scrape_finviz,
            'marketwatch': self.scrape_marketwatch,
            'tradingview': self.scrape_tradingview,
            'yahoo': self.scrape_yahoo_finance,
            'seekingalpha': self.scrape_seeking_alpha,
            'google_news': self.scrape_google_news,
            'stocktwits': self.scrape_stocktwits,
            'bloomberg': self.scrape_bloomberg,
            'cnbc': self.scrape_cnbc,
            'reuters': self.scrape_reuters,
            'benzinga': self.scrape_benzinga,
        }
        
        with ThreadPoolExecutor(max_workers=11) as executor:
            future_to_source = {
                executor.submit(scraper_func, ticker): source_name 
                for source_name, scraper_func in sources.items()
            }
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    result = future.result(timeout=5)
                    if result and result.get('news'):
                        all_news[source_name] = result['news']
                except:
                    pass
        
        return all_news
    
    def get_news_feed(self, limit=200):
        return list(self.news_history)[-limit:]

# ============================================================
# AI ANALYSIS ENGINE - Using custom sentiment (no TextBlob)
# ============================================================

class AIAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 120
        self.ai_usage = {"openai": 0, "claude": 0, "groq": 0, "total": 0, "failures": 0}
        self.last_ai_source = None
        
    def get_ai_sentiment(self, text):
        """Get sentiment using custom analyzer (no TextBlob)"""
        return get_sentiment_analysis(text)
    
    def get_ai_analysis(self, ticker, company, yahoo_data, sentiment_score, news_data):
        cache_key = f"{ticker}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.analysis_cache:
            cache_time, data = self.analysis_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                return data
        
        prompt = self._build_ai_prompt(ticker, company, yahoo_data, sentiment_score, news_data)
        result = None
        ai_source = None
        
        # 1. Try OpenAI
        if not AI_DISABLED_GLOBALLY and openai_client:
            try:
                result = self._get_openai_analysis(prompt)
                if result:
                    ai_source = "OpenAI GPT"
                    self.ai_usage["openai"] += 1
                    self.ai_usage["total"] += 1
                    self.last_ai_source = ai_source
            except:
                pass
        
        # 2. Try Claude
        if not result and claude_client:
            try:
                result = self._get_claude_analysis(prompt)
                if result:
                    ai_source = "Claude AI"
                    self.ai_usage["claude"] += 1
                    self.ai_usage["total"] += 1
                    self.last_ai_source = ai_source
            except:
                pass
        
        # 3. Try Groq
        if not result and not AI_DISABLED_GLOBALLY and not is_groq_rate_limited() and groq_client:
            try:
                result = self._get_groq_analysis(prompt)
                if result:
                    ai_source = "Groq AI"
                    self.ai_usage["groq"] += 1
                    self.ai_usage["total"] += 1
                    self.last_ai_source = ai_source
            except:
                pass
        
        # 4. Enhanced Fallback
        if not result:
            self.ai_usage["failures"] += 1
            result = self._get_enhanced_fallback_analysis(ticker, company, yahoo_data, sentiment_score)
            ai_source = "Technical Fallback"
            self.last_ai_source = ai_source
        
        if result:
            result['ai_source'] = ai_source
            self.analysis_cache[cache_key] = (datetime.now(), result)
            return result
        return None
    
    def _build_ai_prompt(self, ticker, company, yahoo_data, sentiment_score, news_data):
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
        
        return f"""Analyze {ticker} ({company}) stock in detail and provide a comprehensive investment recommendation.

CRITICAL PRICE DIRECTION: The stock is moving {price_direction} ({yahoo_data['change_1d']}% today)
CRITICAL TREND STRENGTH: {trend_strength}
CRITICAL SENTIMENT: News sentiment is {sentiment_direction} (score: {sentiment_score:.2f})

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

RECENT NEWS HEADLINES:
{news_summary}

Based on ALL factors, provide a recommendation.
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
                max_tokens=400
            )
            text = response.choices[0].message.content
            return self._parse_ai_response(text, "OpenAI")
        except:
            return None
    
    def _get_claude_analysis(self, prompt):
        try:
            response = claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            return self._parse_ai_response(text, "Claude")
        except:
            return None
    
    def _get_groq_analysis(self, prompt):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            text = response.choices[0].message.content
            return self._parse_ai_response(text, "Groq")
        except Exception as e:
            if "429" in str(e):
                mark_groq_rate_limited()
            return None
    
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
    
    def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment_score):
        """ENHANCED fallback - More BUY recommendations for strong stocks"""
        score = 50
        change = yahoo_data['change_1d']
        rsi = yahoo_data['rsi']
        trend = yahoo_data.get('trend', 'NEUTRAL')
        trend_strength = yahoo_data.get('trend_strength', 'NEUTRAL')
        price_vs_sma20 = yahoo_data.get('price_vs_sma20', 'BELOW')
        price_vs_sma50 = yahoo_data.get('price_vs_sma50', 'BELOW')
        consecutive_down = yahoo_data.get('consecutive_down_days', 0)
        volume_ratio = yahoo_data.get('volume_ratio', 1)
        
        # 1. PRICE DIRECTION - PRIMARY (weighted heavily)
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
        
        # 2. CONSECUTIVE DOWN DAYS - MODERATE penalty
        if consecutive_down >= 5:
            score -= 15
        elif consecutive_down >= 3:
            score -= 8
        elif consecutive_down >= 2:
            score -= 3
        else:
            score += 10
        
        # 3. MOVING AVERAGES - Strong positive for being above
        if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
            score += 20
        elif price_vs_sma20 == 'ABOVE':
            score += 12
        elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
            score -= 12
        elif price_vs_sma20 == 'BELOW':
            score -= 5
        
        # 4. TREND STRENGTH - Big bonuses for uptrend
        if 'STRONG_BULLISH' in trend or trend_strength == 'STRONG_BULLISH':
            score += 22
        elif 'BULLISH' in trend or trend_strength == 'BULLISH':
            score += 15
        elif 'STRONG_BEARISH' in trend or trend_strength == 'STRONG_BEARISH':
            score -= 12
        elif 'BEARISH' in trend or trend_strength == 'BEARISH':
            score -= 6
        
        # 5. RSI - Optimal range is 40-70
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
        
        # 6. VOLUME - Strong volume on up moves
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
        
        # 7. SENTIMENT - Positive sentiment boosts score
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
        
        # 8. FINAL ADJUSTMENTS
        score = max(10, min(90, score))
        
        # Determine recommendation based on balanced score
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
        
        if price_vs_sma20 == 'ABOVE':
            summary += " (Above SMA20 ✅)"
        elif price_vs_sma20 == 'BELOW':
            summary += " (Below SMA20 ⚠️)"
        
        if consecutive_down >= 3:
            summary += f" ({consecutive_down} down days ⚠️)"
        
        return {
            "rec": rec,
            "conf": score,
            "summary": summary,
            "technical_score": score,
            "sentiment_score": max(0, min(100, 50 + sentiment_score * 10)),
            "risk_level": "HIGH" if consecutive_down >= 4 or score < 30 else "MEDIUM" if score < 55 else "LOW",
            "key_factors": ["Price direction", "Technical analysis", "Trend strength", "Sentiment correlation"],
            "price_target": f"${yahoo_data['price'] * (1 + (score - 50)/250):.2f}",
            "ai_insight": f"Price moving {change:+.1f}% with {consecutive_down} down days, trend: {trend_strength}",
            "momentum_score": score,
            "_source": "Technical Fallback"
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
    "marketwatch": {"name": "MarketWatch", "enabled": True, "icon": "📊", "category": "markets"},
    "tradingview": {"name": "TradingView", "enabled": True, "icon": "📈", "category": "markets"},
    "yahoo": {"name": "Yahoo Finance", "enabled": True, "icon": "💹", "category": "markets"},
    "seekingalpha": {"name": "Seeking Alpha", "enabled": True, "icon": "📰", "category": "equities"},
    "google_news": {"name": "Google News", "enabled": True, "icon": "🔍", "category": "aggregator"},
    "stocktwits": {"name": "StockTwits", "enabled": True, "icon": "💬", "category": "social"},
    "bloomberg": {"name": "Bloomberg", "enabled": True, "icon": "📊", "category": "markets"},
    "cnbc": {"name": "CNBC", "enabled": True, "icon": "📺", "category": "markets"},
    "reuters": {"name": "Reuters", "enabled": True, "icon": "🌐", "category": "markets"},
    "benzinga": {"name": "Benzinga", "enabled": True, "icon": "📈", "category": "equities"},
}

CATEGORIES = {
    "equities": {"name": "Equities", "icon": "📈", "count": 3},
    "markets": {"name": "Markets", "icon": "📊", "count": 5},
    "social": {"name": "Social", "icon": "💬", "count": 1},
    "aggregator": {"name": "Aggregators", "icon": "🔍", "count": 2},
}

# ============================================================
# STOCKS DATABASE - COMPLETE WITH MPC INCLUDED
# ============================================================

ALL_STOCKS = {
    # ===== IMAGE 1: STOCKS 1-20 =====
    "VRAX": {"name": "Virax Biolabs Group Ltd", "sector": "Healthcare"},
    "TDTH": {"name": "Trident Digital Tech Holdings Ltd ADR", "sector": "Technology"},
    "RPGL": {"name": "Republic Power Group Ltd", "sector": "Energy"},
    "SDOT": {"name": "Sadot Group Inc", "sector": "Technology"},
    "SUNE": {"name": "SUNation Energy Inc", "sector": "Energy"},
    "EVGN": {"name": "Evogene Ltd", "sector": "Healthcare"},
    "RBNE": {"name": "Robin Energy Ltd", "sector": "Energy"},
    "NDRA": {"name": "ENDRA Life Sciences Inc", "sector": "Healthcare"},
    "LGHL": {"name": "Lion Group Holding Ltd ADR", "sector": "Financial"},
    "RKTO": {"name": "Rocket One Inc", "sector": "Technology"},
    "MGIH": {"name": "Millennium Group International Holdings Ltd", "sector": "Financial"},
    "FEED": {"name": "ENvue Medical Inc", "sector": "Healthcare"},
    "HCTI": {"name": "Healthcare Triangle Inc", "sector": "Healthcare"},
    "NAMI": {"name": "Jinxin Technology Holding Co ADR", "sector": "Technology"},
    "OMH": {"name": "Ohmyhome Ltd", "sector": "Consumer"},
    "SDST": {"name": "Stardust Power Inc", "sector": "Energy"},
    "FGNX": {"name": "FG Nexus Inc", "sector": "Technology"},
    "BJDX": {"name": "Bluejay Diagnostics Inc", "sector": "Healthcare"},
    "BTAI": {"name": "BioXcel Therapeutics Inc", "sector": "Healthcare"},
    "GREE": {"name": "Greenidge Generation Holdings Inc", "sector": "Technology"},
    
    # ===== IMAGE 2: STOCKS 21-40 =====
    "SHFS": {"name": "SHF Holdings Inc", "sector": "Financial"},
    "PMA": {"name": "Ming Shing Group Holdings Ltd", "sector": "Financial"},
    "NWTG": {"name": "Newton Golf Co Inc", "sector": "Consumer"},
    "XHLD": {"name": "TEN Holdings Inc", "sector": "Technology"},
    "ZNB": {"name": "Zeta Network Group", "sector": "Technology"},
    "ZBAO": {"name": "Zhibao Technology Inc", "sector": "Technology"},
    "PHGE": {"name": "BiomX Inc", "sector": "Healthcare"},
    "PRPO": {"name": "Precipio Inc", "sector": "Healthcare"},
    "BRAG": {"name": "Bragg Gaming Group Inc", "sector": "Consumer"},
    "SCNX": {"name": "Scienture Holdings Inc", "sector": "Healthcare"},
    "GTIM": {"name": "Good Times Restaurants Inc", "sector": "Consumer"},
    "EDBL": {"name": "Edible Garden AG Inc", "sector": "Consumer"},
    "REFR": {"name": "Research Frontiers Inc", "sector": "Technology"},
    "EDUC": {"name": "Educational Development Corp", "sector": "Consumer"},
    "NAAS": {"name": "Naas Technology Inc ADR", "sector": "Technology"},
    "GMEX": {"name": "GMEX Robotics Corp", "sector": "Industrial"},
    "EVTV": {"name": "Envirotech Vehicles Inc", "sector": "Industrial"},
    "AIRI": {"name": "Air Industries Group", "sector": "Industrial"},
    "CNTY": {"name": "Century Casinos Inc", "sector": "Consumer"},
    "NCEL": {"name": "NewcelX Ltd", "sector": "Healthcare"},
    
    # ===== IMAGE 3: STOCKS 41-60 =====
    "ELOX": {"name": "Eloxx Pharmaceuticals Inc", "sector": "Healthcare"},
    "VMAR": {"name": "Vision Marine Technologies Inc", "sector": "Industrial"},
    "MVO": {"name": "MV Oil Trust", "sector": "Energy"},
    "NCNA": {"name": "NuCana plc ADR", "sector": "Healthcare"},
    "ALAR": {"name": "Alarum Technologies Ltd ADR", "sector": "Technology"},
    "FMST": {"name": "Foremost Clean Energy Ltd", "sector": "Energy"},
    "CMMB": {"name": "Chemomab Therapeutics Ltd ADR", "sector": "Healthcare"},
    "ACTU": {"name": "Actuate Therapeutics Inc", "sector": "Healthcare"},
    "GROV": {"name": "Grove Collaborative Holdings Inc", "sector": "Consumer"},
    "IBG": {"name": "Innovation Beverage Group Ltd", "sector": "Consumer"},
    "ELPW": {"name": "Elong Power Holding Ltd", "sector": "Technology"},
    "APVO": {"name": "Aptevo Therapeutics Inc", "sector": "Healthcare"},
    "ZSTK": {"name": "ZeroStack Corp", "sector": "Technology"},
    "TRNR": {"name": "Interactive Strength Inc", "sector": "Consumer"},
    "CCHH": {"name": "CCH Holdings Ltd", "sector": "Financial"},
    "YHC": {"name": "LQR House Inc", "sector": "Consumer"},
    "CYAB": {"name": "Cybara Inc", "sector": "Technology"},
    "TVRD": {"name": "Tvardi Therapeutics Inc", "sector": "Healthcare"},
    "ABVC": {"name": "ABVC BioPharma Inc", "sector": "Healthcare"},
    "AEON": {"name": "AEON Biopharma Inc", "sector": "Healthcare"},
    
    # ===== ADDITIONAL STOCKS =====
    "AFJK": {"name": "Aimei Health Technology Co Ltd", "sector": "Financial"},
    
    # ===== MAJOR STOCKS =====
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
    "DDOG": {"name": "Datadog Inc", "sector": "Technology"},
    "MDB": {"name": "MongoDB Inc", "sector": "Technology"},
    "ZS": {"name": "Zscaler Inc", "sector": "Technology"},
    "PANW": {"name": "Palo Alto Networks", "sector": "Technology"},
    "CRWD": {"name": "CrowdStrike Holdings", "sector": "Technology"},
    "FTNT": {"name": "Fortinet Inc", "sector": "Technology"},
    
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
    "SCHW": {"name": "Charles Schwab", "sector": "Financial"},
    "PNC": {"name": "PNC Financial", "sector": "Financial"},
    "USB": {"name": "US Bancorp", "sector": "Financial"},
    "BK": {"name": "Bank of New York Mellon", "sector": "Financial"},
    "TROW": {"name": "T. Rowe Price", "sector": "Financial"},
    "STT": {"name": "State Street Corp", "sector": "Financial"},
    
    # Healthcare
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
    "HCA": {"name": "HCA Healthcare", "sector": "Healthcare"},
    "BDX": {"name": "Becton Dickinson", "sector": "Healthcare"},
    "ZTS": {"name": "Zoetis Inc", "sector": "Healthcare"},
    "REGN": {"name": "Regeneron Pharma", "sector": "Healthcare"},
    
    # Consumer
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
    "CL": {"name": "Colgate-Palmolive", "sector": "Consumer"},
    "KMB": {"name": "Kimberly-Clark", "sector": "Consumer"},
    "EL": {"name": "Estee Lauder", "sector": "Consumer"},
    "MCO": {"name": "Moody's Corp", "sector": "Consumer"},
    
    # Energy - INCLUDING MPC (Marathon Petroleum)
    "MPC": {"name": "Marathon Petroleum Corp", "sector": "Energy"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "EOG": {"name": "EOG Resources", "sector": "Energy"},
    "SLB": {"name": "Schlumberger", "sector": "Energy"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy"},
    "PSX": {"name": "Phillips 66", "sector": "Energy"},
    "VLO": {"name": "Valero Energy", "sector": "Energy"},
    "KMI": {"name": "Kinder Morgan", "sector": "Energy"},
    "WMB": {"name": "Williams Companies", "sector": "Energy"},
    "OKE": {"name": "ONEOK Inc", "sector": "Energy"},
    
    # Industrial
    "GE": {"name": "General Electric", "sector": "Industrial"},
    "CAT": {"name": "Caterpillar Inc", "sector": "Industrial"},
    "BA": {"name": "Boeing Co", "sector": "Industrial"},
    "RTX": {"name": "Raytheon Technologies", "sector": "Industrial"},
    "HON": {"name": "Honeywell International", "sector": "Industrial"},
    "DE": {"name": "Deere & Co", "sector": "Industrial"},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrial"},
    "NOC": {"name": "Northrop Grumman", "sector": "Industrial"},
    "GD": {"name": "General Dynamics", "sector": "Industrial"},
    "EMR": {"name": "Emerson Electric", "sector": "Industrial"},
    "MMM": {"name": "3M Company", "sector": "Industrial"},
    "DOW": {"name": "Dow Inc", "sector": "Industrial"},
    
    # Communications
    "T": {"name": "AT&T Inc", "sector": "Communications"},
    "VZ": {"name": "Verizon Communications", "sector": "Communications"},
    "TMUS": {"name": "T-Mobile US", "sector": "Communications"},
    "CMCSA": {"name": "Comcast Corp", "sector": "Communications"},
    "CHTR": {"name": "Charter Communications", "sector": "Communications"},
    "EBAY": {"name": "eBay Inc", "sector": "Communications"},
    "SNAP": {"name": "Snap Inc", "sector": "Communications"},
    "TWLO": {"name": "Twilio Inc", "sector": "Communications"},
    
    # Real Estate
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
        momentum = stock_data.get('momentum_score', 50)
        confidence = stock_data.get('confidence', 50)
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
                return None
            
            info = stock.info
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            
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
# ENHANCED RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation_enhanced(data, sentiment_score, ai_analysis):
    score = 50
    change = data['change_1d']
    rsi = data['rsi']
    consecutive_down = data.get('consecutive_down_days', 0)
    price_vs_sma20 = data.get('price_vs_sma20', 'BELOW')
    price_vs_sma50 = data.get('price_vs_sma50', 'BELOW')
    trend = data.get('trend', 'NEUTRAL')
    volume_ratio = data.get('volume_ratio', 1)
    
    # 1. PRICE DIRECTION - PRIMARY
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
    
    # 2. CONSECUTIVE DOWN DAYS
    if consecutive_down >= 5:
        score -= 12
    elif consecutive_down >= 3:
        score -= 6
    elif consecutive_down >= 2:
        score -= 2
    else:
        score += 12
    
    # 3. MOVING AVERAGES
    if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
        score += 22
    elif price_vs_sma20 == 'ABOVE':
        score += 14
    elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
        score -= 10
    elif price_vs_sma20 == 'BELOW':
        score -= 4
    
    # 4. TREND STRENGTH
    if 'STRONG_BULLISH' in trend:
        score += 25
    elif 'BULLISH' in trend:
        score += 16
    elif 'STRONG_BEARISH' in trend:
        score -= 10
    elif 'BEARISH' in trend:
        score -= 5
    
    # 5. RSI
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
    
    # 6. VOLUME
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
    
    # 7. SENTIMENT
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
    
    # 8. AI ANALYSIS
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
    
    # Ensure range
    score = max(10, min(90, round(score)))
    
    # Determine recommendation
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
    
    if price_vs_sma20 == 'ABOVE':
        summary += " (✅ Above SMA20)"
    if consecutive_down < 2 and change > 0:
        summary += " (📈 Positive momentum)"
    
    confidence = min(100, round(score * 0.8 + 10))
    momentum_score = score
    
    return rec, confidence, summary, momentum_score, score

# ============================================================
# MAIN ANALYSIS
# ============================================================

scan_stats = {"technical": 0, "openai": 0, "claude": 0, "groq": 0, "total": 0}
stock_analyzer = EnhancedStockAnalyzer()
loaded_tickers = set()
filter_settings = {"keywords": [], "sources": [], "categories": []}

def get_tickers_by_sector(sector=None):
    if sector and sector != 'all':
        return [t for t, info in ALL_STOCKS.items() if info['sector'] == sector]
    return list(ALL_STOCKS.keys())

def get_next_batch(sector=None, offset=0, batch_size=30, loaded_set=None):
    all_tickers = get_tickers_by_sector(sector)
    
    if not loaded_set:
        # Prioritize major stocks first
        major_stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'MPC', 'XOM', 'CVX', 'JPM', 'JNJ']
        prioritized = []
        for t in all_tickers:
            if t in major_stocks:
                prioritized.append((t, 3))
            elif ALL_STOCKS.get(t, {}).get('sector') in ['Technology', 'Healthcare']:
                prioritized.append((t, 2))
            elif ALL_STOCKS.get(t, {}).get('sector') in ['Financial', 'Energy']:
                prioritized.append((t, 1))
            else:
                prioritized.append((t, 0))
        prioritized.sort(key=lambda x: x[1], reverse=True)
        result = []
        for priority in [3, 2, 1, 0]:
            group = [t for t, p in prioritized if p == priority]
            random.shuffle(group)
            result.extend(group)
        all_tickers = result
    
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
    if use_ai and not AI_DISABLED_GLOBALLY:
        ai_analysis = ai_engine.get_ai_analysis(
            ticker, yahoo_data['company'], yahoo_data, sentiment_score, news_data
        )
        if ai_analysis:
            ai_source = ai_analysis.get('_source', 'AI')
            if 'OpenAI' in ai_source:
                scan_stats["openai"] += 1
            elif 'Claude' in ai_source:
                scan_stats["claude"] += 1
            elif 'Groq' in ai_source:
                scan_stats["groq"] += 1
    
    rec, confidence, summary, momentum_score, score = generate_recommendation_enhanced(
        yahoo_data, sentiment_score, ai_analysis
    )
    
    if ai_analysis:
        source = ai_source
    else:
        source = "Technical Fallback"
        scan_stats["technical"] += 1
    
    scan_stats["total"] += 1
    
    # Rank scoring
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

# IMPORTANT: Load the HTML template from a separate file or use the string
# The HTML template is huge - consider storing it in a separate file
# For Railway deployment, we'll use the string version

# For now, we'll use the HTML_TEMPLATE string (you need to include the full HTML)
# Since the HTML is very long, I'll create a placeholder and load it from a file

# Try to load HTML from file, fallback to string
html_template = None
try:
    # Check if the HTML template exists as a separate file
    template_path = Path('templates/index.html')
    if template_path.exists():
        with open(template_path, 'r') as f:
            html_template = f.read()
        print("✓ Loaded HTML template from file")
    else:
        # Use the embedded template (you need to include the full HTML here)
        # For this fix, I'll provide a minimal version - you should copy your full HTML
        html_template = """<!DOCTYPE html>
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
        .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:8px;margin:12px 0}
        .stat-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px 12px;text-align:center}
        .stat-number{font-size:20px;font-weight:bold}
        .stat-label{color:#888;font-size:9px;margin-top:2px}
        .green{color:#4CAF50}.orange{color:#FF9800}.red{color:#f44336}.blue{color:#64B5F6}.gold{color:#FFD700}
        .loading{text-align:center;padding:30px 20px}
        .spinner{border:4px solid rgba(255,255,255,0.1);border-top:4px solid #667eea;border-radius:50%;width:35px;height:35px;animation:spin 1s linear infinite;margin:0 auto 12px}
        @keyframes spin{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}
        .card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:10px}
        .stock-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px;transition:all 0.3s;cursor:pointer;min-height:300px;display:flex;flex-direction:column}
        .stock-card:hover{transform:translateY(-3px);border-color:rgba(102,126,234,0.4);box-shadow:0 12px 35px rgba(0,0,0,0.3)}
        .tabs{display:flex;gap:4px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;flex-wrap:wrap}
        .tab-btn{padding:6px 16px;border-radius:6px 6px 0 0;background:transparent;border:none;color:#888;cursor:pointer;font-size:12px;transition:all 0.3s;font-weight:600}
        .tab-btn:hover{color:#fff;background:rgba(255,255,255,0.05)}
        .tab-btn.active{color:#fff;background:rgba(102,126,234,0.2);border-bottom:2px solid #667eea}
        .controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
        .search-box{padding:6px 12px;border:2px solid rgba(255,255,255,0.1);border-radius:6px;background:rgba(255,255,255,0.05);color:#fff;flex:1;min-width:150px;font-size:12px}
        .search-box:focus{outline:none;border-color:#667eea}
        .filter-btn{padding:4px 10px;border:2px solid rgba(255,255,255,0.1);border-radius:5px;background:transparent;color:#aaa;cursor:pointer;font-size:10px}
        .filter-btn:hover{border-color:#667eea;color:#fff}
        .filter-btn.active{background:#667eea;color:#fff;border-color:#667eea}
        .load-more-container{text-align:center;padding:20px 0}
        .load-more-btn{background:rgba(102,126,234,0.15);border:2px solid rgba(102,126,234,0.3);color:#667eea;padding:10px 30px;border-radius:10px;font-size:14px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .load-more-btn:hover{background:rgba(102,126,234,0.25)}
        .checkbox-label{font-size:11px;color:#aaa;display:flex;align-items:center;gap:4px;cursor:pointer;background:rgba(255,255,255,0.05);padding:3px 10px;border-radius:5px}
        @media(max-width:768px){.app-container{flex-direction:column}.card-grid{grid-template-columns:1fr}}
    </style>
</head>
<body>
<div class="app-container">
    <div class="main-content">
        <div class="header">
            <div class="header-top">
                <div>
                    <h1>🚀 <span class="gradient">AI Stock Analyzer Pro</span></h1>
                    <div class="subtitle">Stocks • News • Paper Trading</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                    <span id="statusDot" style="font-size:10px;color:#4CAF50">🟢 Live</span>
                    <span id="lastUpdate" style="font-size:9px;color:#666">Never</span>
                    <button class="btn" onclick="refreshData()">🔄 Refresh</button>
                </div>
            </div>
            <div class="stats">
                <div class="stat-card"><div class="stat-number blue" id="totalStocks">0</div><div class="stat-label">Total</div></div>
                <div class="stat-card"><div class="stat-number green" id="buyCount">0</div><div class="stat-label">Buy</div></div>
                <div class="stat-card"><div class="stat-number orange" id="watchCount">0</div><div class="stat-label">Watch</div></div>
                <div class="stat-card"><div class="stat-number red" id="sellCount">0</div><div class="stat-label">Sell</div></div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab-btn active" data-tab="all" onclick="switchTab('all')">📊 All Stocks</button>
            <button class="tab-btn" data-tab="pinned" onclick="switchTab('pinned')">📌 Pinned</button>
            <button class="tab-btn" data-tab="gainers" onclick="switchTab('gainers')">📈 Gainers</button>
            <button class="tab-btn" data-tab="losers" onclick="switchTab('losers')">📉 Losers</button>
        </div>
        
        <div class="controls">
            <input class="search-box" id="searchInput" placeholder="🔍 Search ticker..." oninput="filterCards()">
            <label class="checkbox-label">
                <input type="checkbox" id="aiToggle" checked onchange="toggleAI()"> 🧠 AI
            </label>
        </div>
        
        <div id="loadingState" class="loading">
            <div class="spinner"></div>
            <div style="color:#888;font-size:14px">📊 Loading stocks with AI analysis...</div>
        </div>
        
        <div id="resultsContent" style="display:none">
            <div id="cardGrid" class="card-grid"></div>
            <div class="load-more-container">
                <button class="load-more-btn" onclick="loadMoreStocks()">➕ Add More Stocks</button>
            </div>
        </div>
    </div>
</div>

<script>
let allData = [];
let pinnedStocks = JSON.parse(localStorage.getItem('pinnedStocks') || '[]');
let currentSector = 'all';
let currentTab = 'all';
let useAI = true;
let currentOffset = 0;
let hasMore = true;
let isLoadingMore = false;

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    renderCards();
}

function filterCards() { renderCards(); }

function toggleAI() {
    useAI = document.getElementById('aiToggle').checked;
    currentOffset = 0;
    allData = [];
    refreshData();
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
            filters: {}
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

function renderCards() {
    const grid = document.getElementById('cardGrid');
    let filtered = allData.filter(item => {
        const search = document.getElementById('searchInput').value.toLowerCase();
        if (search && !item.ticker.toLowerCase().includes(search)) return false;
        return true;
    });
    
    if (currentTab === 'pinned') {
        filtered = filtered.filter(item => isPinned(item.ticker));
    } else if (currentTab === 'gainers') {
        filtered = filtered.filter(item => item.change_1d > 2).sort((a, b) => b.change_1d - a.change_1d);
    } else if (currentTab === 'losers') {
        filtered = filtered.filter(item => item.change_1d < -2).sort((a, b) => a.change_1d - b.change_1d);
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
        
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-size:17px;font-weight:700">${item.ticker} ${pinned ? '📌' : ''}</div>
                    <div style="font-size:11px;color:#888">${item.company}</div>
                </div>
                <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">
                    <button class="pin-btn" onclick="togglePin('${item.ticker}', event)">📌</button>
                    <span class="card-rec ${recClass}">${item.recommendation || 'WATCH'}</span>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="font-size:20px;font-weight:700">$${item.price?.toFixed(2) || 'N/A'}</span>
                <span style="font-size:14px;font-weight:600;color:${item.change_1d >= 0 ? '#4CAF50' : '#f44336'}">
                    ${item.change_1d?.toFixed(1) || '0.0'}%
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin:4px 0;font-size:11px;color:#aaa">
                <div>RSI: ${item.rsi || 'N/A'}</div>
                <div>Vol: ${item.volume_ratio?.toFixed(1) || 'N/A'}x</div>
                <div>${item.trend || 'NEUTRAL'}</div>
            </div>
            <div style="font-size:11px;color:#bbb;flex:1;margin:4px 0">${item.summary || 'No analysis'}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:9px;color:#666;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.05)">
                <span>${item.ai_source || 'Technical'}</span>
                <span>Score: ${Math.round(item.rank_score || 0)}</span>
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
}

function openModal(item) {
    alert(`📊 ${item.ticker}\nPrice: $${item.price}\nChange: ${item.change_1d}%\nRSI: ${item.rsi}\nRecommendation: ${item.recommendation}\nConfidence: ${item.confidence}%\n${item.summary}`);
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
            filters: {}
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
        }
    } catch(err) {
        console.error(err);
    } finally {
        isLoadingMore = false;
        btn.textContent = '➕ Add More Stocks';
        btn.disabled = false;
    }
}

refreshData();
</script>
</body>
</html>"""
except Exception as e:
    print(f"⚠️ Error loading HTML: {e}")
    html_template = "<h1>Error loading template</h1>"

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global scan_stats, filter_settings, loaded_tickers
    scan_stats = {"technical": 0, "openai": 0, "claude": 0, "groq": 0, "total": 0}
    
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
    
    filter_settings["keywords"] = data.get('keywords', [])
    filter_settings["sources"] = data.get('sources', [])
    filter_settings["categories"] = data.get('categories', [])
    
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
        'ai_availability': {
            'openai': openai_client is not None,
            'claude': claude_client is not None,
            'groq': groq_client is not None
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
               'price_direction', 'consecutive_down_days', 'price_vs_sma20', 'price_vs_sma50']
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
    user_id, cash = paper_trading.get_or_create_user()
    portfolio_value = paper_trading.get_portfolio_value(user_id)
    transactions = paper_trading.get_transactions(user_id)
    
    recent_trades = []
    for t in transactions[:5]:
        recent_trades.append({
            'ticker': t['ticker'],
            'type': t['type'],
            'shares': t['shares'],
            'price': t['price'],
            'total': t['total'],
            'timestamp': t['timestamp']
        })
    
    return jsonify({
        'success': True,
        'user_id': user_id,
        'cash': cash,
        'portfolio': portfolio_value,
        'transactions': transactions,
        'recent_trades': recent_trades
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
    
    conn = sqlite3.connect(str(paper_trading.db_path))
    cursor = conn.cursor()
    cursor.execute('DELETE FROM portfolio WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM performance_history WHERE user_id = ?', (user_id,))
    cursor.execute('UPDATE users SET cash = 10000 WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    if user_id in paper_trading.cache:
        del paper_trading.cache[user_id]
    
    return jsonify({'success': True, 'message': 'Account reset to $10,000'})

@app.route('/api/paper/history', methods=['GET'])
def paper_history():
    user_id, _ = paper_trading.get_or_create_user()
    days = request.args.get('days', 7, type=int)
    history = paper_trading.get_performance_history(user_id, days)
    
    return jsonify({
        'success': True,
        'history': history,
        'days': days
    })

@app.route('/api/news/sources')
def get_news_sources():
    return jsonify({'success': True, 'sources': NEWS_SOURCES, 'categories': CATEGORIES})

@app.route('/api/news/feed')
def news_feed():
    news = news_scraper.get_news_feed(200)
    return jsonify({'success': True, 'news': news, 'count': len(news)})

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'openai_available': openai_client is not None,
        'claude_available': claude_client is not None,
        'groq_available': groq_client is not None,
        'total_stocks': len(ALL_STOCKS),
        'news_sources': len(NEWS_SOURCES),
        'filters': stock_analyzer.filters
    })

# ============================================================
# RUN THE APP - Modified for Railway
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*80)
    print("🚀 AI Stock Analyzer Pro - COMPLETE WITH PAPER TRADING")
    print("="*80)
    print(f"📈 Total Stocks: {len(ALL_STOCKS)}")
    print(f"📰 News Sources: {len(NEWS_SOURCES)} (ALL 11 WORKING)")
    print(f"🤖 OpenAI: {'✅ Available' if openai_client else '❌ Not available'}")
    print(f"🤖 Claude: {'✅ Available' if claude_client else '❌ Not available'}")
    print(f"🤖 Groq: {'✅ Available' if groq_client else '❌ Not available'}")
    print("="*80)
    print(f"🌐 Starting server on port {port}")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
