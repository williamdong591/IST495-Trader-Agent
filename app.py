@@ -14,7 +14,6 @@
import feedparser
from bs4 import BeautifulSoup
import re
import hashlib
import numpy as np
from collections import deque
import random
@@ -24,133 +23,248 @@
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Set up logging for Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI imports with graceful failure
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI not installed")









warnings.filterwarnings('ignore')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.ERROR)
logging.getLogger('groq').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)


# Load environment variables
load_dotenv()

# Determine if running on Railway






IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None






PORT = int(os.environ.get('PORT', 5000))

app = Flask(__name__)


# ============================================================
# ALPHA VANTAGE API CONFIG
# ============================================================





ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# ============================================================
# DATABASE FOR PAPER TRADING - FIXED FOR RAILWAY
































































































# ============================================================

class PaperTradingDB:
    def __init__(self):
        # Use /tmp for Railway (ephemeral storage)
        if IS_RAILWAY:
            self.db_path = Path('/tmp/paper_trading.db')
        else:
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
                total_profit REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
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
                profit_loss REAL DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Performance history
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
        
        # Fix existing tables - add missing columns
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT user_id, cash, total_profit, total_trades, winning_trades FROM users WHERE username = ?', (username,))
@@ -172,15 +286,15 @@ def get_or_create_user(self, username='default'):
        return user_id, cash, total_profit, total_trades, winning_trades

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
            SELECT ticker, type, shares, price, total, profit_loss, timestamp 
@@ -194,7 +308,7 @@ def get_transactions(self, user_id, limit=50):
        return [{'ticker': r[0], 'type': r[1], 'shares': r[2], 'price': r[3], 'total': r[4], 'profit_loss': r[5], 'timestamp': r[6]} for r in results]

    def buy_stock(self, user_id, ticker, shares, price):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total_cost = shares * price
@@ -246,7 +360,7 @@ def buy_stock(self, user_id, ticker, shares, price):
        return True, f"Bought {shares} shares of {ticker} at ${price:.2f} (Total: ${total_cost:.2f})"

    def sell_stock(self, user_id, ticker, shares, price):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT shares, avg_price FROM portfolio WHERE user_id = ? AND ticker = ?', (user_id, ticker))
@@ -300,7 +414,7 @@ def get_portfolio_value(self, user_id):

        portfolio = self.get_portfolio(user_id)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash, total_profit, total_trades, winning_trades FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
@@ -316,36 +430,29 @@ def get_portfolio_value(self, user_id):

        for item in portfolio:
            try:
                price_data = get_alpha_vantage_price(item['ticker'])
                if price_data and price_data.get('price'):
                    current_price = price_data['price']
                else:
                    stock = yf.Ticker(item['ticker'])
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                    else:
                        current_price = item['avg_price']
                
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
@@ -396,7 +503,7 @@ def _empty_portfolio(self):

    def _save_performance_history(self, user_id, portfolio_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            current_minute = datetime.now().strftime('%Y-%m-%d %H:%M:00')
@@ -418,7 +525,7 @@ def _save_performance_history(self, user_id, portfolio_data):
            pass

    def get_performance_history(self, user_id, days=7):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT total_value, cash, holdings_value, total_profit, timestamp 
@@ -442,12 +549,11 @@ def get_performance_history(self, user_id, days=7):
# ============================================================

def get_alpha_vantage_price(ticker):
    """Get current price from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return None

    try:
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
@@ -464,12 +570,11 @@ def get_alpha_vantage_price(ticker):
    return None

def get_alpha_vantage_historical(ticker, days=60):
    """Get historical data from Alpha Vantage"""
    if not ALPHA_VANTAGE_API_KEY:
        return None

    try:
        url = f"{ALPHA_VANTAGE_BASE_URL}?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
@@ -494,70 +599,27 @@ def get_alpha_vantage_historical(ticker, days=60):
    return None

# ============================================================
# API KEYS FOR AI
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# AI CLIENTS (with graceful failure)
# ============================================================

openai_client = None
claude_client = None
groq_client = None
GROQ_RATE_LIMITED = False
GROQ_LIMIT_RESET_TIME = None
AI_DISABLED_GLOBALLY = False

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✓ OpenAI ready (primary AI)")
    except Exception as e:
        logger.warning(f"⚠️ OpenAI error: {str(e)[:60]}")
        openai_client = None
else:
    if not OPENAI_API_KEY:
        logger.info("⚠️ No OpenAI API key found.")

try:
    from anthropic import Anthropic
    if ANTHROPIC_API_KEY:
        claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("✓ Claude AI ready (fallback)")
except:
    claude_client = None

try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✓ Groq AI ready (fallback)")
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
        logger.warning(f"⚠️ Groq rate limited. AI disabled until {GROQ_LIMIT_RESET_TIME.strftime('%H:%M:%S')}")

# ============================================================
# PRICE PREDICTION ENGINE
@@ -629,31 +691,33 @@ def predict_next_day(self, ticker, historical_data):
            return None

# ============================================================
# ENHANCED NEWS SCRAPER - ALL 11 SOURCES
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
@@ -663,13 +727,246 @@ def _safe_scrape(self, url):
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
@@ -682,14 +979,14 @@ def scrape_finviz(self, ticker):
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
@@ -699,123 +996,56 @@ def scrape_finviz(self, ticker):
                                results['news'].append(news_item)
                                self.news_history.append(news_item)
            self.scrape_cache[cache_key] = (datetime.now(), results)
        except:
            pass
        return results
    
    def scrape_marketwatch(self, ticker):
        results = {'news': []}
        try:
            html = self._safe_scrape(f"https://www.marketwatch.com/investing/stock/{ticker}")
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('div', class_=re.compile('article__content|element--article'))[:3]:
                    headline_elem = article.find('h3') or article.find('a')
                    if headline_elem:
                        text = headline_elem.text.strip()
                        if text and len(text) > 10:
                            news_item = {
                                'headline': text[:200],
                                'source': 'MarketWatch',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                            }
                            results['news'].append(news_item)
                            self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_tradingview(self, ticker):
        results = {'news': []}
        try:
            url = f"https://www.tradingview.com/symbols/{ticker}/"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for item in soup.find_all('div', class_=re.compile('news|article|item|tv-news-item'))[:3]:
                    headline_elem = item.find('a') or item.find('h3') or item.find('div', class_=re.compile('title|headline'))
                    if headline_elem:
                        text = headline_elem.text.strip()
                        if text and len(text) > 10:
                            news_item = {
                                'headline': text[:200],
                                'source': 'TradingView',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                            }
                            results['news'].append(news_item)
                            self.news_history.append(news_item)
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
    
    def scrape_seeking_alpha(self, ticker):
        results = {'news': []}
        try:
            url = f"https://seekingalpha.com/symbol/{ticker}/news"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('article', class_=re.compile('article|post'))[:3]:
                    headline_elem = article.find('h3') or article.find('a')
                    if headline_elem:
                        text = headline_elem.text.strip()
                        if text:
                            news_item = {
                                'headline': text[:200],
                                'source': 'Seeking Alpha',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
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
@@ -826,136 +1056,34 @@ def scrape_stocktwits(self, ticker):
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

    def scrape_bloomberg(self, ticker):
        results = {'news': []}
        try:
            url = f"https://www.bloomberg.com/search?query={ticker}"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('article')[:3]:
                    headline = article.find('h3') or article.find('a')
                    if headline:
                        text = headline.text.strip()
                        if text:
                            news_item = {
                                'headline': text[:200],
                                'source': 'Bloomberg',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                            }
                            results['news'].append(news_item)
                            self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_cnbc(self, ticker):
        results = {'news': []}
        try:
            url = f"https://www.cnbc.com/search/?query={ticker}"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('div', class_=re.compile('card|result|search-result'))[:3]:
                    headline = article.find('a') or article.find('h3')
                    if headline:
                        text = headline.text.strip()
                        if text:
                            news_item = {
                                'headline': text[:200],
                                'source': 'CNBC',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                            }
                            results['news'].append(news_item)
                            self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_reuters(self, ticker):
        results = {'news': []}
        try:
            url = f"https://www.reuters.com/search/news?blob={ticker}"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('div', class_=re.compile('search-result|article'))[:3]:
                    headline = article.find('h3') or article.find('a')
                    if headline:
                        text = headline.text.strip()
                        if text:
                            news_item = {
                                'headline': text[:200],
                                'source': 'Reuters',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
                            }
                            results['news'].append(news_item)
                            self.news_history.append(news_item)
        except:
            pass
        return results
    
    def scrape_benzinga(self, ticker):
        results = {'news': []}
        try:
            url = f"https://www.benzinga.com/search/?q={ticker}"
            html = self._safe_scrape(url)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                for article in soup.find_all('div', class_=re.compile('post|article|news-item'))[:3]:
                    headline = article.find('h3') or article.find('a')
                    if headline:
                        text = headline.text.strip()
                        if text:
                            news_item = {
                                'headline': text[:200],
                                'source': 'Benzinga',
                                'sentiment': self.ai_engine.get_ai_sentiment(text) if self.ai_engine else {'label': 'NEUTRAL'}
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

@@ -969,8 +1097,8 @@ def get_news_feed(self, limit=200):
class AIAnalysisEngine:
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 120
        self.ai_usage = {"openai": 0, "claude": 0, "groq": 0, "total": 0, "failures": 0}
        self.last_ai_source = None

    def get_ai_sentiment(self, text):
@@ -1008,38 +1136,63 @@ def get_ai_analysis(self, ticker, company, yahoo_data, sentiment_score, news_dat
        result = None
        ai_source = None

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


























        if not result:
            self.ai_usage["failures"] += 1
@@ -1128,10 +1281,21 @@ def _get_openai_analysis(self, prompt):
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400

            )
            text = response.choices[0].message.content
            return self._parse_ai_response(text, "OpenAI")










        except:
            return None

@@ -1140,27 +1304,31 @@ def _get_claude_analysis(self, prompt):
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
@@ -1388,107 +1556,30 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
            "_source": "Technical Fallback"
        }

# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

ai_engine = AIAnalysisEngine()
news_scraper = EnhancedNewsScraper()
news_scraper.set_ai_engine(ai_engine)
paper_trading = PaperTradingDB()
prediction_engine = PricePredictionEngine()

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
# STOCKS DATABASE (kept as-is)
# ============================================================

ALL_STOCKS = {
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
    "AFJK": {"name": "Aimei Health Technology Co Ltd", "sector": "Financial"},
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc", "sector": "Technology"},
@@ -1511,12 +1602,6 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "SNOW": {"name": "Snowflake Inc", "sector": "Technology"},
    "PLTR": {"name": "Palantir Technologies", "sector": "Technology"},
    "UBER": {"name": "Uber Technologies", "sector": "Technology"},
    "DDOG": {"name": "Datadog Inc", "sector": "Technology"},
    "MDB": {"name": "MongoDB Inc", "sector": "Technology"},
    "ZS": {"name": "Zscaler Inc", "sector": "Technology"},
    "PANW": {"name": "Palo Alto Networks", "sector": "Technology"},
    "CRWD": {"name": "CrowdStrike Holdings", "sector": "Technology"},
    "FTNT": {"name": "Fortinet Inc", "sector": "Technology"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Financial"},
    "BAC": {"name": "Bank of America", "sector": "Financial"},
    "WFC": {"name": "Wells Fargo", "sector": "Financial"},
@@ -1527,12 +1612,6 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "MA": {"name": "Mastercard Inc", "sector": "Financial"},
    "PYPL": {"name": "PayPal Holdings", "sector": "Financial"},
    "AXP": {"name": "American Express", "sector": "Financial"},
    "SCHW": {"name": "Charles Schwab", "sector": "Financial"},
    "PNC": {"name": "PNC Financial", "sector": "Financial"},
    "USB": {"name": "US Bancorp", "sector": "Financial"},
    "BK": {"name": "Bank of New York Mellon", "sector": "Financial"},
    "TROW": {"name": "T. Rowe Price", "sector": "Financial"},
    "STT": {"name": "State Street Corp", "sector": "Financial"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc", "sector": "Healthcare"},
@@ -1545,10 +1624,6 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "GILD": {"name": "Gilead Sciences", "sector": "Healthcare"},
    "AMGN": {"name": "Amgen Inc", "sector": "Healthcare"},
    "CVS": {"name": "CVS Health Corp", "sector": "Healthcare"},
    "HCA": {"name": "HCA Healthcare", "sector": "Healthcare"},
    "BDX": {"name": "Becton Dickinson", "sector": "Healthcare"},
    "ZTS": {"name": "Zoetis Inc", "sector": "Healthcare"},
    "REGN": {"name": "Regeneron Pharma", "sector": "Healthcare"},
    "KO": {"name": "Coca-Cola Co", "sector": "Consumer"},
    "PEP": {"name": "PepsiCo Inc", "sector": "Consumer"},
    "COST": {"name": "Costco Wholesale", "sector": "Consumer"},
@@ -1561,11 +1636,6 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "NKE": {"name": "Nike Inc", "sector": "Consumer"},
    "DIS": {"name": "Walt Disney", "sector": "Consumer"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer"},
    "CL": {"name": "Colgate-Palmolive", "sector": "Consumer"},
    "KMB": {"name": "Kimberly-Clark", "sector": "Consumer"},
    "EL": {"name": "Estee Lauder", "sector": "Consumer"},
    "MCO": {"name": "Moody's Corp", "sector": "Consumer"},
    "MPC": {"name": "Marathon Petroleum Corp", "sector": "Energy"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
@@ -1574,9 +1644,6 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy"},
    "PSX": {"name": "Phillips 66", "sector": "Energy"},
    "VLO": {"name": "Valero Energy", "sector": "Energy"},
    "KMI": {"name": "Kinder Morgan", "sector": "Energy"},
    "WMB": {"name": "Williams Companies", "sector": "Energy"},
    "OKE": {"name": "ONEOK Inc", "sector": "Energy"},
    "GE": {"name": "General Electric", "sector": "Industrial"},
    "CAT": {"name": "Caterpillar Inc", "sector": "Industrial"},
    "BA": {"name": "Boeing Co", "sector": "Industrial"},
@@ -1586,17 +1653,11 @@ def _get_enhanced_fallback_analysis(self, ticker, company, yahoo_data, sentiment
    "LMT": {"name": "Lockheed Martin", "sector": "Industrial"},
    "NOC": {"name": "Northrop Grumman", "sector": "Industrial"},
    "GD": {"name": "General Dynamics", "sector": "Industrial"},
    "EMR": {"name": "Emerson Electric", "sector": "Industrial"},
    "MMM": {"name": "3M Company", "sector": "Industrial"},
    "DOW": {"name": "Dow Inc", "sector": "Industrial"},
    "T": {"name": "AT&T Inc", "sector": "Communications"},
    "VZ": {"name": "Verizon Communications", "sector": "Communications"},
    "TMUS": {"name": "T-Mobile US", "sector": "Communications"},
    "CMCSA": {"name": "Comcast Corp", "sector": "Communications"},
    "CHTR": {"name": "Charter Communications", "sector": "Communications"},
    "EBAY": {"name": "eBay Inc", "sector": "Communications"},
    "SNAP": {"name": "Snap Inc", "sector": "Communications"},
    "TWLO": {"name": "Twilio Inc", "sector": "Communications"},
    "AMT": {"name": "American Tower", "sector": "Real Estate"},
    "PLD": {"name": "Prologis Inc", "sector": "Real Estate"},
    "SPG": {"name": "Simon Property Group", "sector": "Real Estate"},
@@ -1615,9 +1676,9 @@ class EnhancedStockAnalyzer:
    def __init__(self):
        self.stock_cache = {}
        self.cache_ttl = 60
        self.ai_engine = ai_engine
        self.news_scraper = news_scraper
        self.prediction_engine = prediction_engine
        self.loaded_tickers = set()
        self.filters = {
            'min_price': 0,
@@ -1631,6 +1692,15 @@ def __init__(self):
            'trend_filter': 'all'
        }










    def set_filters(self, filters):
        self.filters.update(filters)

@@ -1643,8 +1713,6 @@ def apply_filters(self, stock_data):
        rsi = stock_data.get('rsi', 50)
        change = stock_data.get('change_1d', 0)
        sentiment = stock_data.get('sentiment_aggregate', 0)
        momentum = stock_data.get('momentum_score', 50)
        confidence = stock_data.get('confidence', 50)
        trend = stock_data.get('trend', 'NEUTRAL')

        if price < self.filters.get('min_price', 0) or price > self.filters.get('max_price', 10000):
@@ -1677,144 +1745,6 @@ def get_stock_data(self, ticker):
                return data

        try:
            av_price = get_alpha_vantage_price(ticker)
            av_historical = get_alpha_vantage_historical(ticker, 60)
            
            if av_price and av_historical:
                current_price = av_price['price']
                change_1d = av_price['change_pct']
                volume_ratio = av_price['volume'] / 1000000 if av_price['volume'] > 0 else 1
                
                prices = av_historical['prices']
                if len(prices) >= 14:
                    gains = []
                    losses = []
                    for i in range(1, len(prices)):
                        diff = prices[i] - prices[i-1]
                        if diff > 0:
                            gains.append(diff)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(diff))
                    
                    avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
                    avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.001
                    rs = avg_gain / avg_loss if avg_loss > 0 else 0
                    current_rsi = 100 - (100 / (1 + rs))
                else:
                    current_rsi = 50
                
                if len(prices) >= 20:
                    sma20 = sum(prices[-20:]) / 20
                else:
                    sma20 = current_price
                if len(prices) >= 50:
                    sma50 = sum(prices[-50:]) / 50
                else:
                    sma50 = current_price
                
                price_vs_sma20 = 'ABOVE' if current_price > sma20 else 'BELOW'
                price_vs_sma50 = 'ABOVE' if current_price > sma50 else 'BELOW'
                
                macd_bullish = False
                if len(prices) >= 26:
                    ema12 = self._calculate_ema(prices, 12)
                    ema26 = self._calculate_ema(prices, 26)
                    macd_line = ema12 - ema26
                    signal_line = self._calculate_ema(macd_line, 9)
                    macd_bullish = macd_line[-1] > signal_line[-1]
                
                adx = 0
                if len(prices) >= 28:
                    adx = 25 + (random.random() - 0.5) * 20
                
                breakout = False
                if len(prices) >= 20:
                    highest20 = max(prices[-21:-1]) if len(prices) >= 21 else max(prices[:-1])
                    breakout = current_price > highest20
                
                relative_strength = (change_1d * 2) + (random.random() - 0.5) * 4
                
                boll_signal = "NORMAL"
                if len(prices) >= 20:
                    sma = sum(prices[-20:]) / 20
                    std = np.std(prices[-20:])
                    upper = sma + std * 2
                    lower = sma - std * 2
                    if current_price < lower:
                        boll_signal = "OVERSOLD"
                    elif current_price > upper:
                        boll_signal = "OVERBOUGHT"
                
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
                
                consecutive_down = 0
                for i in range(2, min(10, len(prices))):
                    if prices[-i] < prices[-(i+1)]:
                        consecutive_down += 1
                    else:
                        break
                
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
                        "dates": av_historical['dates'],
                        "prices": av_historical['prices'],
                        "volumes": av_historical['volumes']
                    },
                    "pe_ratio": None,
                    "target_price": None,
                    "current_volume": av_price['volume'],
                    "after_hours_price": current_price,
                    "after_hours_pct": 0,
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
            
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2mo", timeout=2)
            if hist.empty:
@@ -1927,8 +1857,8 @@ def get_stock_data(self, ticker):
            relative_strength = 0
            try:
                if len(hist) >= 20:
                    spy = yf.download("SPY", period="2mo", progress=False)
                    if not spy.empty and len(spy) >= 20:
                        stock_return = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-20]) - 1)
                        spy_return = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-20]) - 1)
                        relative_strength = float((stock_return - spy_return) * 100)
@@ -1993,33 +1923,13 @@ def get_stock_data(self, ticker):
            logger.error(f"⚠️ Error for {ticker}: {e}")
            return self._get_fallback_data(ticker)

    def _calculate_ema(self, prices, period):
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def _get_fallback_data(self, ticker):
        sector_info = ALL_STOCKS.get(ticker, {})
        price = 10 + random.random() * 200
        change = (random.random() - 0.3) * 4
        rsi = 35 + random.random() * 30
        after_hours = (random.random() - 0.1) * 2

        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        prices = []
        current_price = price * (1 - change / 100)
        for i in range(30):
            if i == 29:
                prices.append(round(price, 2))
            else:
                daily_change = (random.random() - 0.5) * 0.06
                current_price = current_price * (1 + daily_change)
                prices.append(round(current_price, 2))
        
        return {
            "ticker": ticker,
            "company": sector_info.get('name', ticker),
@@ -2036,11 +1946,7 @@ def _get_fallback_data(self, ticker):
            "price_vs_sma20": "ABOVE" if change > 0 else "BELOW",
            "price_vs_sma50": "ABOVE" if change > -0.5 else "BELOW",
            "consecutive_down_days": 0 if change > 0 else random.randint(1, 3),
            "historical": {
                "dates": dates,
                "prices": prices,
                "volumes": [int(500000 + random.random() * 2000000) for _ in range(30)]
            },
            "pe_ratio": round(15 + random.random() * 20, 2),
            "target_price": round(price * (1 + random.random() * 0.2), 2),
            "current_volume": int(500000 + random.random() * 2000000),
@@ -2056,6 +1962,16 @@ def _get_fallback_data(self, ticker):
        }

    def get_news_sentiment(self, ticker):










        news_data = self.news_scraper.fetch_all_news(ticker)
        sentiment_scores = {'BULLISH': 0, 'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0, 'BEARISH': 0}
        total_news = 0
@@ -2307,42 +2223,30 @@ def generate_recommendation_enhanced(data, sentiment_score, ai_analysis, predict
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
        return [t for t, info in ALL_STOCKS.items() if info.get('sector', '') == sector]
    return list(ALL_STOCKS.keys())

def get_next_batch(sector=None, offset=0, batch_size=30, loaded_set=None):
    all_tickers = get_tickers_by_sector(sector)

    if not loaded_set:
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

@@ -2372,7 +2276,7 @@ def analyze_stock_complete(ticker, use_ai=True):

    ai_analysis = None
    ai_source = "Technical"
    if use_ai and not AI_DISABLED_GLOBALLY:
        ai_analysis = ai_engine.get_ai_analysis(
            ticker, yahoo_data['company'], yahoo_data, sentiment_score, news_data, prediction_data
        )
@@ -2384,6 +2288,8 @@ def analyze_stock_complete(ticker, use_ai=True):
                scan_stats["claude"] += 1
            elif 'Groq' in ai_source:
                scan_stats["groq"] += 1



    rec, confidence, summary, momentum_score, score = generate_recommendation_enhanced(
        yahoo_data, sentiment_score, ai_analysis, prediction_data
@@ -2528,14 +2434,14 @@ def index():

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
@@ -2544,10 +2450,6 @@ def analyze():
    if filters:
        stock_analyzer.set_filters(filters)

    filter_settings["keywords"] = data.get('keywords', [])
    filter_settings["sources"] = data.get('sources', [])
    filter_settings["categories"] = data.get('categories', [])
    
    if not tickers:
        if load_more:
            new_tickers = get_next_batch(sector, offset, limit, loaded_tickers)
@@ -2596,9 +2498,10 @@ def analyze():
        'elapsed': elapsed,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ai_availability': {
            'openai': openai_client is not None,
            'claude': claude_client is not None,
            'groq': groq_client is not None

        },
        'filters': stock_analyzer.filters
    })
@@ -2635,9 +2538,10 @@ def export_data():
@app.route('/api/paper/status', methods=['GET'])
def paper_status():
    try:
        user_id, cash, total_profit, total_trades, winning_trades = paper_trading.get_or_create_user()
        portfolio_value = paper_trading.get_portfolio_value(user_id)
        transactions = paper_trading.get_transactions(user_id)


        portfolio_value['total_profit'] = total_profit
        portfolio_value['total_trades'] = total_trades
@@ -2664,21 +2568,21 @@ def paper_buy():
        if not ticker or shares <= 0:
            return jsonify({'success': False, 'error': 'Invalid input'}), 400

        price_data = get_alpha_vantage_price(ticker)
        if price_data and price_data.get('price'):
            price = price_data['price']
        else:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                return jsonify({'success': False, 'error': f'Could not get price for {ticker}'}), 400
            price = float(hist['Close'].iloc[-1])



        user_id, _, _, _, _ = paper_trading.get_or_create_user()
        success, message = paper_trading.buy_stock(user_id, ticker, shares, price)


        if success:
            portfolio = paper_trading.get_portfolio_value(user_id)
            return jsonify({
                'success': True,
                'message': message,
@@ -2703,21 +2607,21 @@ def paper_sell():
        if not ticker or shares <= 0:
            return jsonify({'success': False, 'error': 'Invalid input'}), 400

        price_data = get_alpha_vantage_price(ticker)
        if price_data and price_data.get('price'):
            price = price_data['price']
        else:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                return jsonify({'success': False, 'error': f'Could not get price for {ticker}'}), 400
            price = float(hist['Close'].iloc[-1])



        user_id, _, _, _, _ = paper_trading.get_or_create_user()
        success, message = paper_trading.sell_stock(user_id, ticker, shares, price)


        if success:
            portfolio = paper_trading.get_portfolio_value(user_id)
            return jsonify({
                'success': True,
                'message': message,
@@ -2735,9 +2639,10 @@ def paper_sell():
@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    try:
        user_id, _, _, _, _ = paper_trading.get_or_create_user()


        conn = sqlite3.connect(paper_trading.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM portfolio WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
@@ -2746,8 +2651,8 @@ def paper_reset():
        conn.commit()
        conn.close()

        if user_id in paper_trading.cache:
            del paper_trading.cache[user_id]

        return jsonify({'success': True, 'message': 'Account reset to $10,000'})
    except Exception as e:
@@ -2756,9 +2661,10 @@ def paper_reset():
@app.route('/api/paper/history', methods=['GET'])
def paper_history():
    try:
        user_id, _, _, _, _ = paper_trading.get_or_create_user()

        days = request.args.get('days', 7, type=int)
        history = paper_trading.get_performance_history(user_id, days)

        return jsonify({
            'success': True,
@@ -2785,17 +2691,17 @@ def performance_stats():
def status():
    return jsonify({
        'status': 'online',
        'openai_available': openai_client is not None,
        'claude_available': claude_client is not None,
        'groq_available': groq_client is not None,
        'alpha_vantage_available': ALPHA_VANTAGE_API_KEY is not None,
        'total_stocks': len(ALL_STOCKS),
        'news_sources': len(NEWS_SOURCES),
        'filters': stock_analyzer.filters
    })

# ============================================================
# HTML TEMPLATE (Shortened for brevity - uses the same as before)
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
@@ -2887,6 +2793,7 @@ def status():
        .ai-badge.openai{background:rgba(255,152,0,0.2);color:#FFB74D}
        .ai-badge.claude{background:rgba(156,39,176,0.2);color:#CE93D8}
        .ai-badge.groq{background:rgba(76,175,80,0.2);color:#81C784}

        .ai-badge.technical{background:rgba(255,255,255,0.05);color:#888}
        .news-feed{max-height:400px;overflow-y:auto}
        .news-item{padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05)}
@@ -2910,29 +2817,30 @@ def status():
        .prediction-bullish{background:rgba(76,175,80,0.2);color:#4CAF50}
        .prediction-bearish{background:rgba(244,67,54,0.2);color:#f44336}
        .prediction-neutral{background:rgba(255,152,0,0.2);color:#FFB74D}
        .paper-trading-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;margin-bottom:15px}
        .paper-trading-panel h3{font-size:14px;margin-bottom:8px;color:#888}
        .paper-stats{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}
        .paper-stat{text-align:center;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px}
        .paper-stat .value{font-size:18px;font-weight:bold}
        .paper-stat .label{font-size:9px;color:#888}
        .paper-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0;padding:8px;background:rgba(255,215,0,0.05);border-radius:6px}
        .paper-row .item{text-align:center}
        .paper-row .item .value{font-size:16px;font-weight:bold}
        .paper-row .item .label{font-size:8px;color:#888}
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
@@ -2948,43 +2856,43 @@ def status():
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
            <div class="paper-row">
                <div class="item">
                    <span class="gold" id="paperWinRate" style="font-size:16px;font-weight:bold">0%</span>
                    <div class="label">Win Rate</div>
                </div>
                <div class="item">
                    <span class="gold" id="paperTrades" style="font-size:16px;font-weight:bold">0</span>
                    <div class="label">Total Trades</div>
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
@@ -3007,6 +2915,7 @@ def status():
                <span id="aiOpenAI" style="color:#FFB74D">● OpenAI</span>
                <span id="aiClaude" style="color:#CE93D8">● Claude</span>
                <span id="aiGroq" style="color:#4CAF50">● Groq</span>

            </div>
        </div>
        
@@ -3089,7 +2998,7 @@ def status():
            <div class="header-top">
                <div>
                    <h1>🚀 <span class="gradient">AI Stock Analyzer Pro</span></h1>
                    <div class="subtitle">200+ Stocks • 11 News Sources • Paper Trading • After Hours • AI Predictions</div>
                </div>
                <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
                    <span id="statusDot" style="font-size:10px;color:#4CAF50">🟢 Live</span>
@@ -3228,7 +3137,6 @@ def status():
</div>

<script>
// JavaScript code - same as before (kept for brevity, but you can copy the full script from the previous version)
let allData = [];
let currentSector = 'all';
let currentTab = 'all';
@@ -3258,17 +3166,14 @@ def status():
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
@@ -3286,7 +3191,6 @@ def status():
            } else {
                list.innerHTML = '<div style="color:#666;font-size:11px;padding:8px 0">No holdings</div>';
            }
            
            const transList = document.getElementById('transactionList');
            if (data.transactions && data.transactions.length > 0) {
                transList.innerHTML = data.transactions.slice(0, 10).map(t => `
@@ -3312,26 +3216,21 @@ def status():
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
@@ -3347,9 +3246,7 @@ def status():
    if (!currentModalTicker) return;
    const shares = prompt(`Enter number of shares to ${action} for ${currentModalTicker}:`, '1');
    if (!shares || parseFloat(shares) <= 0) return;
    
    const endpoint = action === 'buy' ? '/api/paper/buy' : '/api/paper/sell';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
@@ -3359,9 +3256,7 @@ def status():
        const data = await response.json();
        document.getElementById('quickTradeInfo').textContent = data.message;
        document.getElementById('quickTradeInfo').style.color = data.success ? '#4CAF50' : '#f44336';
        if (data.success) {
            updatePaperStatus();
        }
    } catch(e) {
        document.getElementById('quickTradeInfo').textContent = '⚠️ Error';
        document.getElementById('quickTradeInfo').style.color = '#f44336';
@@ -3375,22 +3270,16 @@ def status():
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
@@ -3403,9 +3292,7 @@ def status():
            document.getElementById('tradeMessage').textContent = '✅ Account reset';
            document.getElementById('tradeMessage').style.color = '#4CAF50';
        }
    } catch(e) {
        alert('Error resetting');
    }
}

function switchTab(tab) {
@@ -3428,12 +3315,8 @@ def status():

function setSort(sort) {
    const btn = document.querySelector(`.sort-btn[data-sort="${sort}"]`);
    if (currentSort === sort) {
        sortDescending = !sortDescending;
    } else {
        currentSort = sort;
        sortDescending = true;
    }
    document.querySelectorAll('.sort-btn').forEach(b => {
        b.classList.remove('active');
        if (b.dataset.sort === sort) {
@@ -3532,9 +3415,7 @@ def status():
                    <span>${src.icon} ${src.name}</span>
                `;
                grid.appendChild(div);
                if (src.enabled !== false && !selectedSources.includes(key)) {
                    selectedSources.push(key);
                }
            }
        }
    } catch(e) { console.error(e); }
@@ -3564,7 +3445,7 @@ def status():
    keywords = [];
    renderKeywords();
    document.querySelectorAll('#sourceGrid input[type="checkbox"]').forEach(cb => cb.checked = true);
    selectedSources = ['finviz', 'marketwatch', 'tradingview', 'yahoo', 'seekingalpha', 'google_news', 'stocktwits', 'bloomberg', 'cnbc', 'reuters', 'benzinga'];
    document.getElementById('keywordInput').value = '';
    currentOffset = 0;
    allData = [];
@@ -3577,11 +3458,10 @@ def status():
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
@@ -3590,20 +3470,17 @@ def status():
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
            
            let totalNews = 0, totalMomentum = 0, macdCount = 0, breakoutCount = 0, afterHoursCount = 0, predictionCount = 0;
            allData.forEach(item => {
                if (item.news) {
@@ -3624,24 +3501,21 @@ def status():
            document.getElementById('breakoutCount').textContent = breakoutCount;
            document.getElementById('afterHoursCount').textContent = afterHoursCount;
            document.getElementById('predictionCount').textContent = predictionCount;
            
            const aiCount = res.stats ? (res.stats.openai || 0) + (res.stats.claude || 0) + (res.stats.groq || 0) : 0;
            document.getElementById('aiCount').textContent = aiCount;
            
            if (res.ai_availability) {
                document.getElementById('aiOpenAI').style.color = res.ai_availability.openai ? '#FFB74D' : '#666';
                document.getElementById('aiClaude').style.color = res.ai_availability.claude ? '#CE93D8' : '#666';
                document.getElementById('aiGroq').style.color = res.ai_availability.groq ? '#4CAF50' : '#666';

            }
            
            renderCards();
            updateStats();
            updatePaperStatus();
            loadNewsFeed();
        }
    } catch(err) {
        console.error(err);
    } finally {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';
        btn.disabled = false;
@@ -3666,12 +3540,11 @@ def status():
    const btn = document.querySelector('.load-more-btn');
    btn.textContent = '⏳ Loading...';
    btn.disabled = true;
    currentOffset += 30;
    
    try {
        const payload = {
            sector: currentSector !== 'all' ? currentSector : null,
            limit: 30,
            offset: currentOffset,
            load_more: true,
            use_ai: useAI,
@@ -3680,14 +3553,12 @@ def status():
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
@@ -3697,9 +3568,8 @@ def status():
            updateStats();
            updatePaperStatus();
        }
    } catch(err) {
        console.error(err);
    } finally {
        isLoadingMore = false;
        btn.textContent = '➕ Add More Stocks';
        btn.disabled = false;
@@ -3710,24 +3580,20 @@ def status():
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
@@ -3747,31 +3613,25 @@ def status():
        filtered = filtered.filter(item => item.prediction && item.prediction.confidence > 50);
        filtered.sort((a, b) => (b.prediction?.confidence || 0) - (a.prediction?.confidence || 0));
    }
    
    if (currentSort && currentTab !== 'gainers' && currentTab !== 'losers' && currentTab !== 'afterhours' && currentTab !== 'predictions') {
        filtered.sort((a, b) => {
            let va = a[currentSort] ?? 0;
            let vb = b[currentSort] ?? 0;
            if (typeof va === 'string') {
                return sortDescending ? va.localeCompare(vb) : vb.localeCompare(va);
            }
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
@@ -3783,39 +3643,33 @@ def status():
        if (item.breakout) cardClass += ' breakout-highlight';
        card.className = cardClass;
        card.onclick = () => openModal(item);
        
        const recClass = (item.recommendation || 'WATCH').toLowerCase().replace(' ', '-');
        const pinned = isPinned(item.ticker);
        const aiSource = item.ai_source || item.source || 'Technical';
        const aiClass = aiSource.includes('OpenAI') ? 'openai' : aiSource.includes('Claude') ? 'claude' : aiSource.includes('Groq') ? 'groq' : 'technical';
        const momentum = item.momentum_score || 50;
        const sentiment = item.sentiment_aggregate || 0;
        const sentimentEmoji = sentiment > 0.3 ? '🟢' : sentiment < -0.3 ? '🔴' : '🟡';
        const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
        const directionClass = direction === 'UP' ? 'direction-up' : 'direction-down';
        const downDays = item.consecutive_down_days || 0;
        const trendClass = item.trend && item.trend.includes('BULLISH') ? 'trend-bullish' : 
                          item.trend && item.trend.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
        const macdSignal = item.macd_bullish ? '🟢' : '⚪';
        const breakoutSignal = item.breakout ? '🚀 ' : '';
        const ahPct = item.after_hours_pct || 0;
        const ahDisplay = Math.abs(ahPct) > 0.5 ? `${ahPct > 0 ? '📈' : '📉'} AH ${ahPct > 0 ? '+' : ''}${ahPct.toFixed(1)}%` : '';
        
        let predDisplay = '';
        if (item.prediction && item.prediction.confidence > 50) {
            const pred = item.prediction;
            const predClass = pred.prediction === 'BULLISH' || pred.prediction === 'STRONG BULLISH' ? 'prediction-bullish' : 
                              pred.prediction === 'BEARISH' || pred.prediction === 'STRONG BEARISH' ? 'prediction-bearish' : 'prediction-neutral';
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
@@ -3884,7 +3738,6 @@ def status():
        </div>
        <div style="display:flex;flex-direction:column;gap:4px;margin-top:6px">
    `;
    
    filtered.forEach((item, index) => {
        const pinned = isPinned(item.ticker);
        const isBullish = (item.trend && item.trend.includes('BULLISH')) && item.change_1d > 0;
@@ -3894,22 +3747,18 @@ def status():
        if (isBullish) rowClass += ' bullish-highlight';
        if (isDowntrend) rowClass += ' downtrend-warning';
        if (item.breakout) rowClass += ' breakout-highlight';
        
        const rankEmoji = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index+1}`;
        const aiSource = item.ai_source || item.source || 'Technical';
        const aiClass = aiSource.includes('OpenAI') ? 'openai' : aiSource.includes('Claude') ? 'claude' : aiSource.includes('Groq') ? 'groq' : 'technical';
        const momentum = item.momentum_score || 50;
        const direction = item.price_direction || (item.change_1d > 0 ? 'UP' : 'DOWN');
        const downDays = item.consecutive_down_days || 0;
        const trendDisplay = item.trend ? item.trend.substring(0, 12) : 'NEUTRAL';
        const trendClass = item.trend && item.trend.includes('BULLISH') ? 'trend-bullish' : 
                          item.trend && item.trend.includes('BEARISH') ? 'trend-bearish' : 'trend-neutral';
        const macdDisplay = item.macd_bullish ? '🟢BULL' : '⚪';
        const breakoutDisplay = item.breakout ? '🚀' : '';
        const ahDisplay = Math.abs(item.after_hours_pct || 0) > 0.5 ? `${item.after_hours_pct > 0 ? '🟢' : '🔴'}${(item.after_hours_pct || 0).toFixed(1)}%` : '';
        const predDisplay = item.prediction && item.prediction.confidence > 50 ? 
            `${item.prediction.expected_change > 0 ? '📈' : '📉'}${(item.prediction.expected_change || 0).toFixed(1)}%` : '';
        
        container.innerHTML += `
            <div class="${rowClass}" onclick="openModal(item)" data-ticker="${item.ticker}">
                <span style="min-width:30px;font-weight:bold;color:#667eea">${rankEmoji}</span>
@@ -3932,7 +3781,6 @@ def status():
            </div>
        `;
    });
    
    container.innerHTML += '</div>';
}

@@ -3943,36 +3791,23 @@ def status():
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
        const sourceIcon = {
            'Finviz': '📊', 'MarketWatch': '📊', 'TradingView': '📈',
            'Yahoo Finance': '💹', 'Seeking Alpha': '📰', 'Google News': '🔍',
            'StockTwits': '💬', 'Bloomberg': '📊', 'CNBC': '📺',
            'Reuters': '🌐', 'Benzinga': '📈'
        }[source] || '📰';
        
        html += `
            <div style="margin-bottom:8px;padding:6px;background:rgba(255,255,255,0.03);border-radius:6px;border-left:2px solid rgba(102,126,234,0.3)">
                <div style="font-size:10px;color:#667eea;font-weight:bold;margin-bottom:4px">${sourceIcon} ${source}</div>
        `;
        
        items.slice(0, 5).forEach(n => {
            const sentimentEmoji = n.sentiment?.label === 'BULLISH' ? '🟢' : 
                                   n.sentiment?.label === 'BEARISH' ? '🔴' : 
                                   n.sentiment?.label === 'POSITIVE' ? '🟢' : 
                                   n.sentiment?.label === 'NEGATIVE' ? '🔴' : '🟡';
            const headline = n.headline || 'No headline';
            const shortHeadline = headline.length > 120 ? headline.substring(0, 120) + '...' : headline;
            
            html += `
                <div class="news-item">
                    <div style="font-size:11px;color:#ddd">${sentimentEmoji} ${shortHeadline}</div>
@@ -3984,10 +3819,8 @@ def status():
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    html += `</div>`;
    container.innerHTML = html;
}
@@ -4021,10 +3854,7 @@ def status():
}

async function exportData() {
    if (allData.length === 0) {
        alert('No data to export.');
        return;
    }
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
@@ -4042,9 +3872,7 @@ def status():
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
    } catch(err) {
        alert('Error exporting: ' + err.message);
    }
}

function openModal(item) {
@@ -4065,7 +3893,6 @@ def status():
    const ahText = Math.abs(ahPct) > 0.1 ? `🌙 After Hours: ${ahPct > 0 ? '+' : ''}${ahPct.toFixed(2)}%` : '';
    document.getElementById('modalAfterHours').textContent = ahText;
    document.getElementById('modalAfterHours').style.color = ahPct > 0 ? '#4CAF50' : ahPct < 0 ? '#f44336' : '#888';
    
    const predBox = document.getElementById('modalPredictionBox');
    if (item.prediction && item.prediction.confidence > 50) {
        predBox.style.display = 'block';
@@ -4082,7 +3909,6 @@ def status():
        predBox.style.display = 'none';
        document.getElementById('modalPrediction').textContent = '';
    }
    
    const stats = [
        { label: 'Price', value: '$' + (item.price?.toFixed(2) || 'N/A') },
        { label: 'Change', value: (item.change_1d?.toFixed(1) || '0.0') + '%', class: item.change_1d >= 0 ? 'green' : 'red' },
@@ -4113,56 +3939,16 @@ def status():
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
                datasets: [{ 
                    label: 'Price', 
                    data: hist.prices, 
                    borderColor: '#667eea', 
                    backgroundColor: 'rgba(102,126,234,0.1)', 
                    fill: true, 
                    tension: 0.4, 
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: true,
                plugins: { 
                    legend: { 
                        display: true, 
                        labels: { color: '#888', font: { size: 10 } } 
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#666', font: { size: 9 } },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    x: {
                        ticks: { color: '#666', font: { size: 8 }, maxTicksLimit: 10 },
                        grid: { display: false }
                    }
                }
            }
        });
    }
    
    const modalNews = document.getElementById('modalNews');
    modalNews.innerHTML = '';
    if (item.news_items && item.news_items.length > 0) {
@@ -4210,18 +3996,29 @@ def status():

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 AI Stock Analyzer Pro - RAILWAY DEPLOYMENT READY")
    print("="*80)
    print(f"📈 Total Stocks: {len(ALL_STOCKS)}")
    print(f"📰 News Sources: {len(NEWS_SOURCES)}")
    print(f"🤖 OpenAI: {'✅ Available' if openai_client else '❌ Not available'}")
    print(f"🤖 Claude: {'✅ Available' if claude_client else '❌ Not available'}")
    print(f"🤖 Groq: {'✅ Available' if groq_client else '❌ Not available'}")
    print(f"📊 Alpha Vantage: {'✅ Available' if ALPHA_VANTAGE_API_KEY else '❌ Not available'}")












    print("="*80)
    print("🌐 Running on Railway with Gunicorn")
    print(f"📡 Port: {PORT}")
    print("="*80)

    # Run with Gunicorn compatible settings
    app.run(host='0.0.0.0', port=PORT, debug=False)
