"""
AI Stock Analyzer Pro - Complete Application
For GitHub and Railway Deployment
"""
import os
import sys
import json
import sqlite3
import threading
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import random
import logging
import time
import io
import re
import hashlib

# Third-party imports
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file
from bs4 import BeautifulSoup
import feedparser
from dotenv import load_dotenv
from textblob import TextBlob
from werkzeug.middleware.proxy_fix import ProxyFix

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Disable warnings
warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.ERROR)
logging.getLogger('groq').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# ============================================================
# DATABASE FOR PAPER TRADING
# ============================================================

class PaperTradingDB:
    def __init__(self):
        self.db_path = Path('paper_trading.db')
        self.init_db()
        self.cache = {}
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
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
            return False, f"Insufficient shares. You have {existing_shares}"
        
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
        self._save_performance_history(user_id, result)
        
        return result
    
    def _save_performance_history(self, user_id, portfolio_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
# AI CLIENTS
# ============================================================

openai_client = None
claude_client = None
groq_client = None
GROQ_RATE_LIMITED = False
GROQ_LIMIT_RESET_TIME = None
AI_DISABLED_GLOBALLY = False

# Initialize OpenAI
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI ready (primary AI)")
    except Exception as e:
        print(f"⚠️ OpenAI error: {str(e)[:60]}")
        openai_client = None
else:
    if not OPENAI_API_KEY:
        print("⚠️ No OpenAI API key found. Add OPENAI_API_KEY to .env file")

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
# ENHANCED NEWS SCRAPER
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
    
    def scrape_yahoo_finance(self, ticker):
        results = {'news': []}
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                for item in news[:5]:
                    headline = item.get('title', '')
                    if headline:
                        news_item = {
                            'headline': headline[:200],
                            'source': 'Yahoo Finance',
                            'link': item.get('link', ''),
                            'sentiment': self.ai_engine.get_ai_sentiment(headline) if self.ai_engine else {'label': 'NEUTRAL'}
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_google_news(self, ticker):
        results = {'news': []}
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                news_item = {
                    'headline': entry.title[:200],
                    'source': 'Google News',
                    'link': entry.link,
                    'published': entry.published if hasattr(entry, 'published') else '',
                    'sentiment': self.ai_engine.get_ai_sentiment(entry.title) if self.ai_engine else {'label': 'NEUTRAL'}
                }
                results['news'].append(news_item)
                self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_stocktwits(self, ticker):
        results = {'news': []}
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            response = self.session.get(url, timeout=3)
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
                            'created': msg.get('created_at', '')
                        }
                        results['news'].append(news_item)
                        self.news_history.append(news_item)
        except:
            pass
        return results
    
    def fetch_all_news(self, ticker):
        all_news = {}
        
        sources = {
            'finviz': self.scrape_finviz,
            'yahoo': self.scrape_yahoo_finance,
            'google_news': self.scrape_google_news,
            'stocktwits': self.scrape_stocktwits,
        }
        
        with ThreadPoolExecutor(max_workers=4) as executor:
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
# AI ANALYSIS ENGINE
# ============================================================

class AIAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 120
        self.ai_usage = {"openai": 0, "claude": 0, "groq": 0, "total": 0, "failures": 0}
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
        
        # 4. Fallback
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
        
        return f"""Analyze {ticker} ({company}) stock and provide an investment recommendation.

TECHNICAL DATA:
- Current Price: ${yahoo_data['price']}
- 1-Day Change: {yahoo_data['change_1d']}%
- RSI (14-day): {yahoo_data['rsi']}
- Trend: {yahoo_data['trend']}
- Trend Strength: {yahoo_data.get('trend_strength', 'NEUTRAL')}
- Volume Ratio: {yahoo_data['volume_ratio']}x average
- SMA 20: ${yahoo_data.get('sma20', 0)}
- SMA 50: ${yahoo_data.get('sma50', 0)}
- Price vs SMA20: {yahoo_data.get('price_vs_sma20', 'N/A')}
- Price vs SMA50: {yahoo_data.get('price_vs_sma50', 'N/A')}
- Consecutive Down Days: {yahoo_data.get('consecutive_down_days', 0)}

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
        score = 50
        change = yahoo_data['change_1d']
        rsi = yahoo_data['rsi']
        trend = yahoo_data.get('trend', 'NEUTRAL')
        trend_strength = yahoo_data.get('trend_strength', 'NEUTRAL')
        price_vs_sma20 = yahoo_data.get('price_vs_sma20', 'BELOW')
        price_vs_sma50 = yahoo_data.get('price_vs_sma50', 'BELOW')
        consecutive_down = yahoo_data.get('consecutive_down_days', 0)
        volume_ratio = yahoo_data.get('volume_ratio', 1)
        
        # Price Direction
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
        
        # Consecutive Down Days
        if consecutive_down >= 5:
            score -= 15
        elif consecutive_down >= 3:
            score -= 8
        elif consecutive_down >= 2:
            score -= 3
        else:
            score += 10
        
        # Moving Averages
        if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
            score += 20
        elif price_vs_sma20 == 'ABOVE':
            score += 12
        elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
            score -= 12
        elif price_vs_sma20 == 'BELOW':
            score -= 5
        
        # Trend Strength
        if 'STRONG_BULLISH' in trend or trend_strength == 'STRONG_BULLISH':
            score += 22
        elif 'BULLISH' in trend or trend_strength == 'BULLISH':
            score += 15
        elif 'STRONG_BEARISH' in trend or trend_strength == 'STRONG_BEARISH':
            score -= 12
        elif 'BEARISH' in trend or trend_strength == 'BEARISH':
            score -= 6
        
        # RSI
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
        
        # Volume
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
        
        # Sentiment
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
    
    # Small Cap Stocks
    "VRAX": {"name": "Virax Biolabs Group Ltd", "sector": "Healthcare"},
    "TDTH": {"name": "Trident Digital Tech Holdings Ltd ADR", "sector": "Technology"},
    "RPGL": {"name": "Republic Power Group Ltd", "sector": "Energy"},
    "SDOT": {"name": "Sadot Group Inc", "sector": "Technology"},
    "SUNE": {"name": "SUNation Energy Inc", "sector": "Energy"},
    "EVGN": {"name": "Evogene Ltd", "sector": "Healthcare"},
    "RBNE": {"name": "Robin Energy Ltd", "sector": "Energy"},
    "NDRA": {"name": "ENDRA Life Sciences Inc", "sector": "Healthcare"},
    "LGHL": {"name": "Lion Group Holding Ltd ADR", "sector": "Financial"},
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
    "SHFS": {"name": "SHF Holdings Inc", "sector": "Financial"},
    "PMA": {"name": "Ming Shing Group Holdings Ltd", "sector": "Financial"},
    "NWTG": {"name": "Newton Golf Co Inc", "sector": "Consumer"},
    "XHLD": {"name": "TEN Holdings Inc", "sector": "Technology"},
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
    "TRNR": {"name": "Interactive Strength Inc", "sector": "Consumer"},
    "CCHH": {"name": "CCH Holdings Ltd", "sector": "Financial"},
    "YHC": {"name": "LQR House Inc", "sector": "Consumer"},
    "ABVC": {"name": "ABVC BioPharma Inc", "sector": "Healthcare"},
    "AEON": {"name": "AEON Biopharma Inc", "sector": "Healthcare"},
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
    
    # Price Direction
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
    
    # Consecutive Down Days
    if consecutive_down >= 5:
        score -= 12
    elif consecutive_down >= 3:
        score -= 6
    elif consecutive_down >= 2:
        score -= 2
    else:
        score += 12
    
    # Moving Averages
    if price_vs_sma20 == 'ABOVE' and price_vs_sma50 == 'ABOVE':
        score += 22
    elif price_vs_sma20 == 'ABOVE':
        score += 14
    elif price_vs_sma20 == 'BELOW' and price_vs_sma50 == 'BELOW':
        score -= 10
    elif price_vs_sma20 == 'BELOW':
        score -= 4
    
    # Trend Strength
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
    
    # AI Analysis
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
    
    confidence = min(100, round(score * 0.8 + 10))
    momentum_score = score
    
    return rec, confidence, summary, momentum_score, score

# ============================================================
# MAIN ANALYSIS
# ============================================================

scan_stats = {"technical": 0, "openai": 0, "claude": 0, "groq": 0, "total": 0}
stock_analyzer = EnhancedStockAnalyzer()
loaded_tickers = set()

def get_tickers_by_sector(sector=None):
    if sector and sector != 'all':
        return [t for t, info in ALL_STOCKS.items() if info['sector'] == sector]
    return list(ALL_STOCKS.keys())

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

# HTML Template (simplified version - full template is in the original code)
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
        .main-content{flex:1;min-width:0}
        .header{background:rgba(255,255,255,0.05);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:18px 22px;margin-bottom:18px}
        .header-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
        h1{font-size:1.6em;display:flex;align-items:center;gap:8px}
        .gradient{background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .btn{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(102,126,234,0.4)}
        .btn-success{background:linear-gradient(135deg,#4CAF50,#2E7D32);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn-success:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(76,175,80,0.4)}
        .btn-danger{background:linear-gradient(135deg,#f44336,#c62828);color:#fff;border:none;padding:7px 18px;border-radius:8px;font-size:13px;cursor:pointer;transition:all 0.3s;font-weight:600}
        .btn-danger:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(244,67,54,0.4)}
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
        .card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:10px}
        .stock-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px;transition:all 0.3s;cursor:pointer;min-height:280px;display:flex;flex-direction:column}
        .stock-card:hover{transform:translateY(-3px);border-color:rgba(102,126,234,0.4);box-shadow:0 12px 35px rgba(0,0,0,0.3)}
        .stock-card.pinned{border-color:#FFD700;background:rgba(255,215,0,0.05)}
        .stock-card.bullish-highlight{border-color:#4CAF50;background:rgba(76,175,80,0.05)}
        .stock-card.downtrend-warning{border-color:#f44336;background:rgba(244,67,54,0.05)}
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
                    <div class="value gold" id="paperCash">$10,000</div>
                    <div class="label">Cash</div>
                </div>
                <div class="paper-stat">
                    <div class="value blue" id="paperValue">$0</div>
                    <div class="label">Holdings</div>
                </div>
                <div class="paper-stat">
                    <div class="value" id="paperTotal">$10,000</div>
                    <div class="label">Total Value</div>
                </div>
                <div class="paper-stat">
                    <div class="value" id="paperPL">+$0.00</div>
                    <div class="label">Total P&L</div>
                </div>
            </div>
            <div class="trade-form">
                <select id="tradeAction" style="width:70px">
                    <option value="buy">BUY</option>
                    <option value="sell">SELL</option>
                </select>
                <input id="tradeTicker" placeholder="Ticker" style="width:70px">
                <input id="tradeShares" placeholder="Shares" type="number" style="width:70px">
                <button class="btn-success btn-sm" onclick="executeTrade()">Execute</button>
                <button class="btn-danger btn-sm" onclick="resetPaperTrading()">Reset</button>
            </div>
            <div id="tradeMessage" style="font-size:11px;margin-top:4px;color:#888"></div>
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
    </div>
    
    <div class="main-content">
        <button onclick="document.getElementById('sidebar').classList.toggle('collapsed')" style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:8px 16px;color:#aaa;cursor:pointer;margin-bottom:15px">⚙️ Filters</button>
        
        <div class="header">
            <div class="header-top">
                <div>
                    <h1>🚀 <span class="gradient">AI Stock Analyzer Pro</span></h1>
                    <div class="subtitle" style="color:#888;font-size:12px">Paper Trading • AI Analysis • 11 News Sources</div>
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
                <div class="stat-card"><div class="stat-number gold" id="pinnedCount">0</div><div class="stat-label">📌 Pinned</div></div>
            </div>
        </div>
        
        <div class="controls">
            <input class="search-box" id="searchInput" placeholder="🔍 Search ticker or company..." oninput="filterCards()">
            <button class="filter-btn active" data-sector="all" onclick="setSector('all')">All</button>
            <button class="filter-btn" data-sector="Technology" onclick="setSector('Technology')">Tech</button>
            <button class="filter-btn" data-sector="Financial" onclick="setSector('Financial')">Fin</button>
            <button class="filter-btn" data-sector="Healthcare" onclick="setSector('Healthcare')">Health</button>
            <button class="filter-btn" data-sector="Energy" onclick="setSector('Energy')">Energy</button>
        </div>
        
        <div id="loadingState" class="loading">
            <div class="spinner"></div>
            <div style="color:#888;font-size:14px">📊 Loading stocks with AI analysis...</div>
        </div>
        
        <div id="resultsContent" style="display:none">
            <div id="cardGrid" class="card-grid"></div>
            <div class="load-more-container" style="text-align:center;padding:20px 0">
                <button class="load-more-btn" onclick="loadMoreStocks()" style="background:rgba(102,126,234,0.15);border:2px solid rgba(102,126,234,0.3);color:#667eea;padding:10px 30px;border-radius:10px;font-size:14px;cursor:pointer;transition:all 0.3s;font-weight:600">➕ Add More Stocks</button>
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
            </div>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-grid" id="modalStats"></div>
        <div class="modal-chart"><canvas id="modalChart"></canvas></div>
        <div id="modalNews" style="margin-top:10px;max-height:200px;overflow-y:auto"></div>
    </div>
</div>

<script>
let allData = [];
let currentSector = 'all';
let pinnedStocks = JSON.parse(localStorage.getItem('pinnedStocks') || '[]');
let chart = null;
let currentOffset = 0;
let hasMore = true;
let isLoadingMore = false;

async function refreshData() {
    const btn = document.querySelector('.btn');
    btn.disabled = true;
    btn.textContent = '⏳...';
    
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultsContent').style.display = 'none';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sector: currentSector !== 'all' ? currentSector : null,
                limit: 30,
                offset: 0,
                load_more: false,
                use_ai: true,
                pinned: pinnedStocks,
                filters: {}
            })
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
    
    currentOffset += 30;
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sector: currentSector !== 'all' ? currentSector : null,
                limit: 30,
                offset: currentOffset,
                load_more: true,
                use_ai: true,
                pinned: pinnedStocks,
                filters: {}
            })
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
    } catch(err) {
        console.error(err);
    } finally {
        isLoadingMore = false;
        btn.textContent = '➕ Add More Stocks';
        btn.disabled = false;
    }
}

function renderCards() {
    const grid = document.getElementById('cardGrid');
    let filtered = allData.filter(item => {
        if (currentSector !== 'all' && item.sector !== currentSector) return false;
        const search = document.getElementById('searchInput').value.toLowerCase();
        if (search && !item.ticker.toLowerCase().includes(search) && !item.company.toLowerCase().includes(search)) return false;
        return true;
    });
    
    grid.innerHTML = '';
    
    filtered.forEach((item) => {
        const card = document.createElement('div');
        const isBullish = (item.trend && item.trend.includes('BULLISH')) && item.change_1d > 0;
        const isDowntrend = item.consecutive_down_days >= 3 || (item.trend && item.trend.includes('BEARISH'));
        let cardClass = 'stock-card';
        if (pinnedStocks.includes(item.ticker)) cardClass += ' pinned';
        if (isBullish) cardClass += ' bullish-highlight';
        if (isDowntrend) cardClass += ' downtrend-warning';
        card.className = cardClass;
        card.onclick = () => openModal(item);
        
        const recClass = (item.recommendation || 'WATCH').toLowerCase().replace(' ', '-');
        const pinned = pinnedStocks.includes(item.ticker);
        const aiSource = item.ai_source || item.source || 'Technical';
        
        card.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-size:17px;font-weight:700">${item.ticker} ${pinned ? '📌' : ''}</div>
                    <div style="font-size:11px;color:#888">${item.company}</div>
                </div>
                <div style="display:flex;align-items:center;gap:4px">
                    <button class="pin-btn ${pinned ? 'pinned' : ''}" onclick="togglePin('${item.ticker}', event)">📌</button>
                    <span class="card-rec ${recClass}">${item.recommendation || 'WATCH'}</span>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="font-size:20px;font-weight:700">$${item.price?.toFixed(2) || 'N/A'}</span>
                <span style="font-size:14px;font-weight:600;color:${item.change_1d >= 0 ? '#4CAF50' : '#f44336'}">
                    ${item.change_1d?.toFixed(1) || '0.0'}% ${item.change_1d >= 0 ? '▲' : '▼'}
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin:4px 0;font-size:11px;color:#aaa">
                <div>RSI: ${item.rsi || 'N/A'}</div>
                <div>Vol: ${item.volume_ratio?.toFixed(1) || 'N/A'}x</div>
                <div>${item.trend_icon || '➡️'} ${item.trend || 'NEUTRAL'}</div>
            </div>
            ${item.consecutive_down_days >= 2 ? `<div style="font-size:10px;color:#f44336;font-weight:bold">⚠️ ${item.consecutive_down_days} down days</div>` : ''}
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#888;margin:2px 0">
                <span>Momentum: ${item.momentum_score || 50}%</span>
                <span>Conf: ${item.confidence || 0}%</span>
            </div>
            <div style="font-size:11px;color:#bbb;flex:1;margin:4px 0">${item.summary || 'No analysis'}</div>
            <div style="display:flex;justify-content:space-between;font-size:9px;color:#666;margin-top:4px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.05)">
                <span>${aiSource}</span>
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

function togglePin(ticker, event) {
    event.stopPropagation();
    const index = pinnedStocks.indexOf(ticker);
    if (index > -1) pinnedStocks.splice(index, 1);
    else pinnedStocks.push(ticker);
    localStorage.setItem('pinnedStocks', JSON.stringify(pinnedStocks));
    renderCards();
    updateStats();
}

function openModal(item) {
    document.getElementById('modalTicker').textContent = item.ticker;
    document.getElementById('modalCompany').textContent = item.company + ' • ' + item.sector;
    
    const stats = [
        { label: 'Price', value: '$' + (item.price?.toFixed(2) || 'N/A') },
        { label: 'Change', value: (item.change_1d?.toFixed(1) || '0.0') + '%' },
        { label: 'RSI', value: item.rsi || 'N/A' },
        { label: 'Volume Ratio', value: item.volume_ratio?.toFixed(2) || 'N/A' },
        { label: 'Momentum', value: (item.momentum_score || 50) + '%' },
        { label: 'Recommendation', value: item.recommendation || 'WATCH' },
        { label: 'Confidence', value: item.confidence + '%' },
        { label: 'Rank Score', value: Math.round(item.rank_score || 0) }
    ];
    
    const modalStats = document.getElementById('modalStats');
    modalStats.innerHTML = '';
    stats.forEach(s => {
        const div = document.createElement('div');
        div.className = 'modal-stat';
        div.innerHTML = `<div class="label">${s.label}</div><div class="value">${s.value}</div>`;
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
            div.innerHTML = `[${n.source}] ${n.headline?.substring(0, 150) || ''}...`;
            modalNews.appendChild(div);
        });
    } else {
        modalNews.innerHTML = '<div style="color:#666;padding:10px">No news available</div>';
    }
    
    document.getElementById('detailModal').classList.add('active');
}

function closeModal() {
    document.getElementById('detailModal').classList.remove('active');
    if (chart) { chart.destroy(); chart = null; }
}

// PAPER TRADING FUNCTIONS
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
                        <strong>${t.ticker}</strong> ${t.shares} @ $${t.price.toFixed(2)}
                        <span style="float:right">$${t.total.toFixed(2)}</span>
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

async function quickSell(ticker, maxShares) {
    const shares = prompt(`Enter number of shares to sell for ${ticker} (max ${maxShares}):`, maxShares);
    if (!shares || parseFloat(shares) <= 0) return;
    
    try {
        const response = await fetch('/api/paper/sell', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, shares: parseFloat(shares) })
        });
        const data = await response.json();
        if (data.success) {
            updatePaperStatus();
        } else {
            alert(data.message);
        }
    } catch(e) {
        alert('Error selling');
    }
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
    } catch(e) {
        alert('Error resetting');
    }
}

// Initialize
refreshData();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global scan_stats, loaded_tickers
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

def get_next_batch(sector=None, offset=0, batch_size=30, loaded_set=None):
    all_tickers = get_tickers_by_sector(sector)
    
    if loaded_set:
        all_tickers = [t for t in all_tickers if t not in loaded_set]
    
    start = offset
    end = min(offset + batch_size, len(all_tickers))
    if start >= len(all_tickers):
        return []
    return all_tickers[start:end]

@app.route('/api/export', methods=['POST'])
def export_data():
    data = request.get_json() or {}
    results = data.get('results', [])
    if not results:
        return jsonify({'success': False, 'error': 'No data'}), 400
    
    df = pd.DataFrame(results)
    columns = ['ticker', 'company', 'sector', 'price', 'change_1d', 'rsi', 'volume_ratio', 
               'trend', 'trend_strength', 'recommendation', 'confidence', 'rank_score', 'news_count', 
               'sentiment_aggregate', 'source', 'ai_source', 'momentum_score', 'technical_score']
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
    
    conn = sqlite3.connect(paper_trading.db_path)
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

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'openai_available': openai_client is not None,
        'claude_available': claude_client is not None,
        'groq_available': groq_client is not None,
        'total_stocks': len(ALL_STOCKS),
        'filters': stock_analyzer.filters
    })

# ============================================================
# RUN THE APP
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*80)
    print("🚀 AI Stock Analyzer Pro - Railway Deployment")
    print("="*80)
    print(f"📈 Total Stocks: {len(ALL_STOCKS)}")
    print(f"🤖 OpenAI: {'✅ Available' if openai_client else '❌ Not available'}")
    print(f"🤖 Claude: {'✅ Available' if claude_client else '❌ Not available'}")
    print(f"🤖 Groq: {'✅ Available' if groq_client else '❌ Not available'}")
    print("="*80)
    print(f"🌐 http://localhost:{port}")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)
