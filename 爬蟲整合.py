# -*- coding: utf-8 -*-  # 指定檔案編碼為 UTF-8，確保程式碼碼中的中文字元能被正常解析
import logging
import pandas_datareader.data as web
import os    # 載入 os 模組，用於設定環境變數
import time  # 載入 time 模組，用於控制程式暫停 (sleep) 或時間計算
import random
# ==========================================
# 強制設定系統時區為台北時間 (確保雲端排程時間正確)
# ==========================================
os.environ['TZ'] = 'Asia/Taipei'
if hasattr(time, 'tzset'):  # 防呆機制：因為 Windows 不支援 tzset，只在 Linux/Mac 環境下執行
    time.tzset()
import pandas as pd  # 載入 pandas 套件並縮寫為 pd，用於強大的資料表 (DataFrame) 操作與分析
import numpy as np  # 載入 numpy 套件並縮寫為 np，用於高效的數值計算與陣列操作
import requests  # 載入 requests 套件，用於發送 HTTP 請求（爬蟲抓網頁資料用）
import requests_cache
from requests import Session
from requests_cache import CacheMixin, RedisCache
from requests_ratelimiter import LimiterMixin, LimiterSession
from pathlib import Path  # 從 pathlib 模組載入 Path，用於跨平台的檔案與目錄路徑操作
import feedparser  # 載入 feedparser 套件，用於解析 RSS 訂閱來源（這裡用來抓新聞標題）
import yfinance as yf  # 載入 yfinance 套件，用於從 Yahoo Finance 抓取全球金融歷史數據
from io import StringIO  # 從 io 模組載入 StringIO，用於將字串轉換為類似檔案的物件，方便 pandas 讀取
import warnings  # 載入 warnings 模組，用於控制警告訊息的顯示
import sqlite3  # 載入 sqlite3 模組，用於連接與操作 SQLite 本地輕量級資料庫
from functools import reduce  # 從 functools 載入 reduce，用於將一個函數連續應用到序列的元素上（這裡用來合併多個資料表）

# 🔥 NLP 套件載入 (FinBERT)  # 標示此區塊為自然語言處理 (NLP) 套件的初始化
try:  # 嘗試執行以下程式碼，若缺少套件則跳到 except 區塊
    from transformers import pipeline  # 從 Hugging Face 的 transformers 庫載入 pipeline，方便快速建立 NLP 任務管線
    import torch  # 載入 PyTorch 深度學習框架
    # 檢查是否有 GPU，有的話使用 GPU 加速  # 說明下一行的邏輯
    device = 0 if torch.cuda.is_available() else -1  # 若有 NVIDIA GPU 則設定 device 為 0，否則為 -1 (使用 CPU)
    print(f"✅ Transformers 載入成功 (Device: {'GPU' if device==0 else 'CPU'})")  # 印出成功訊息與當前使用的運算設備
    HAS_NLP = True  # 設定全域變數，標記 NLP 套件已成功載入
except ImportError:  # 如果系統沒有安裝 transformers 或 torch，捕捉 ImportError
    print("⚠️ 未安裝 transformers/torch，將降級使用規則基礎情緒分析")  # 印出警告，提示將改用簡單的替代方案
    HAS_NLP = False  # 設定全域變數，標記 NLP 套件未載入

# ==========================================
# 設定與環境初始化  # 標示此區塊為全域變數與環境設定
# ==========================================
DB_PATH = "data/oil_price.db"  # 設定 SQLite 資料庫的儲存路徑
HAS_NLP = False  # ⚠️ 這裡強制將 NLP 標記設為 False (可能是為了測試或除錯先關閉)
from requests.packages.urllib3.exceptions import InsecureRequestWarning  # 載入特定警告類別
warnings.simplefilter('ignore', InsecureRequestWarning)  # 忽略 HTTPS 請求中憑證驗證失敗的警告 (配合爬蟲 verify=False 使用)

HEADERS = {  # 設定 HTTP 請求標頭 (Headers)
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}  # 偽裝成一般的 Google Chrome 瀏覽器，降低被目標網站阻擋的機率

SEED = 42  # 設定隨機亂數種子，確保每次執行的隨機結果一致 (可重現性)
DATA_DIR = Path("data")  # 建立指向 "data" 資料夾的 Path 物件
DATA_DIR.mkdir(exist_ok=True)  # 如果 "data" 資料夾不存在就建立它，存在則不報錯

# ==========================================
# 1. 資料庫存取  # 標示此區塊為資料庫讀寫功能
# ==========================================
def load_from_db(table_name="market_data"):  # 定義函式，用於從資料庫讀取資料，預設表名為 "market_data"
    conn = sqlite3.connect(DB_PATH)  # 建立與 SQLite 資料庫的連線
    try:  # 嘗試讀取資料
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)  # 執行 SQL 語法撈取整張表的資料，並轉為 DataFrame
    except:  # 如果發生錯誤 (例如資料表不存在)
        df = pd.DataFrame(columns=['日期'])  # 建立一個只有 '日期' 欄位的空 DataFrame 作為防呆回傳值
    finally:  # 無論成功或失敗，最後一定要執行的區塊
        conn.close()  # 關閉資料庫連線，釋放資源
    return df  # 回傳讀取到的資料表 (或空表)

def save_to_db(df, table_name="market_data"):  # 定義函式，用於將資料寫入資料庫，預設表名為 "market_data"
    if df.empty: return  # 如果傳入的 DataFrame 是空的，直接結束函式不處理
    df_save = df.copy()  # 複製一份資料表，避免修改到原始物件
    if '日期' in df_save.columns:  # 檢查資料表是否有 '日期' 欄位
        if pd.api.types.is_datetime64_any_dtype(df_save['日期']):  # 檢查 '日期' 欄位是否為 pandas 的時間格式
            df_save['日期'] = df_save['日期'].dt.strftime('%Y-%m-%d')  # 將時間格式轉換為字串 (YYYY-MM-DD) 方便存入 SQLite
            
    conn = sqlite3.connect(DB_PATH)  # 建立資料庫連線
    try:  # 嘗試寫入資料
        df_save.to_sql(table_name, conn, if_exists='replace', index=False)  # 將資料寫入指定資料表，如果表已存在則「覆蓋 (replace)」，且不存入索引
        print(f"✅ 資料庫已成功更新 ({len(df_save)} 筆資料)")  # 印出成功訊息與寫入筆數
    except Exception as e:  # 捕捉寫入過程中的任何錯誤
        print(f"❌ 資料庫寫入失敗: {e}")  # 印出錯誤原因
    finally:  # 無論成功或失敗
        conn.close()  # 關閉連線

# ==========================================
# 2. 網路爬蟲函式  # 標示此區塊負責外部資料獲取
# ==========================================
def fetch_single_ticker_safe(ticker_info, start_date, max_retries=3): 
    """
    單一下載函式 (安全重試版)：遇到 Rate Limit 時會自動等待並重試
    """
    ticker, name = ticker_info
    print(f"   ⏳ 正在下載: {name} ({ticker})...")
    
    # 建立一個強化偽裝的 Session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    })

    for attempt in range(max_retries):
        try:
            # 如果是重試，增加等待時間 (指數退避)
            if attempt > 0:
                wait_time = random.uniform(5, 10) * attempt
                print(f"   ⚠️ 觸發限制或異常，等待 {wait_time:.1f} 秒後進行第 {attempt + 1} 次重試...")
                time.sleep(wait_time)

            new_data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False, timeout=15, session=session, proxy=None)
            
            # yfinance 遇到 Rate Limit 有時會直接回傳空的 DataFrame，以此觸發重試
            if new_data.empty:   
                raise ValueError("下載資料為空 (可能觸發 Rate Limit)")

            # --- 1. 欄位選取邏輯 --- 
            price_data = None 
            if isinstance(new_data.columns, pd.MultiIndex): 
                try: 
                    if 'Close' in new_data.columns.get_level_values(0): 
                        price_data = new_data['Close'] 
                    elif 'Close' in new_data.columns.get_level_values(1): 
                        price_data = new_data.xs('Close', level=1, axis=1) 
                    else: 
                        price_data = new_data.iloc[:, 0] 
                except: 
                     price_data = new_data.iloc[:, 0] 
            elif 'Close' in new_data.columns: 
                price_data = new_data['Close'] 
            elif 'Adj Close' in new_data.columns: 
                 price_data = new_data['Adj Close'] 
            else: 
                 price_data = new_data.iloc[:, 0] 

            # --- 2. 型別統一轉 DataFrame --- 
            if isinstance(price_data, pd.Series): 
                price_df = price_data.to_frame(name=name) 
            else: 
                price_df = price_data.copy() 
                if price_df.shape[1] > 1: 
                    price_df = price_df.iloc[:, :1] 
                price_df.columns = [name] 

            # 3. 索引與日期處理 
            price_df.index.name = '日期' 
            price_df = price_df.reset_index() 
            price_df['日期'] = pd.to_datetime(price_df['日期']) 
            
            # 4. 防呆檢查 
            last_val = price_df[name].iloc[-1] 
            if name == '布蘭特原油': 
                if last_val > 200 or last_val < 10: 
                    print(f"   ⚠️ {name} 數值異常 ({last_val})，捨棄")
                    return None 
            elif name == '台幣匯率': 
                if last_val > 50 or last_val < 20: 
                    print(f"   ⚠️ {name} 數值異常 ({last_val})，捨棄")
                    return None 
                
            print(f"   ✅ 下載成功: {name} (最新值: {last_val:.2f})") 
            return price_df 

        except Exception as e: 
            print(f"   ❌ 第 {attempt + 1} 次嘗試失敗 ({name}): {e}")
            if attempt == max_retries - 1:
                print(f"   🚨 {name} 已達最大重試次數，放棄下載。")
                return None
    
def fetch_asia_neighbor_prices():  # 定義函式，獲取亞洲鄰國(日韓)的最新油價相關指數
    print("🌐 執行亞鄰競爭國數據爬蟲...")  # 印出提示
    try:  # 嘗試執行
        jp_ticker = yf.Ticker("1671.T")  # 設定日本原油 ETF 或相關標的代碼
        kr_ticker = yf.Ticker("096770.KS")  # 設定韓國煉油企業 (如 SK Innovation) 的代碼作為代理變數      
        jp_hist = jp_ticker.history(period="5d")  # 獲取日本標的過去 5 天的歷史資料
        kr_hist = kr_ticker.history(period="5d")  # 獲取韓國標的過去 5 天的歷史資料
        
        if jp_hist.empty or kr_hist.empty:  # 如果其中一個抓不到資料
            return {'status': 'FAILED', 'JP_val': 0, 'KR_val': 0}  # 回傳失敗狀態與預設值

        return {  # 若成功，回傳包含最新價格與狀態的字典
            'JP_val': jp_hist['Close'].iloc[-1],  # 取出日本標的最後一天收盤價
            'KR_val': kr_hist['Close'].iloc[-1],  # 取出韓國標的最後一天收盤價
            'status': 'SUCCESS'  # 設定狀態為成功
        }
    except Exception as e:  # 若發生連線或其他錯誤
        print(f"⚠️ 亞鄰爬蟲異常: {e}")  # 印出錯誤訊息
        return {'status': 'FAILED', 'JP_val': np.nan, 'KR_val': np.nan}  # 回傳失敗狀態與空值(NaN)
    
def fetch_cpc_oil_history():  # 定義函式，爬取中油官方網站的歷史油價
    urls = [  # 設定要爬取的目標網址列表
        "https://vipmbr.cpc.com.tw/mbwebs/showhistoryprice_oil2019.aspx",  # 2019 年以後的歷史油價網頁
        "https://vipmbr.cpc.com.tw/mbwebs/showhistoryprice_oil.aspx"  # 舊版或備用的歷史油價網頁
    ]
    oil_dfs = []  # 建立一個空列表，準備存放爬取下來的資料表
    
    for url in urls:  # 針對網址列表使用迴圈逐一處理
        try:  # 嘗試爬取與解析
            print(f"   ☁️ 嘗試連線: {url} ...")  # 印出正在連線的網址
            res = requests.get(url, headers=HEADERS, verify=False, timeout=10)  # 發送 GET 請求，忽略憑證驗證，超時 10 秒
            res.encoding = "utf-8"  # 強制將回應內容編碼設為 UTF-8 避免中文亂碼
            
            dfs = pd.read_html(StringIO(res.text), flavor=['lxml', 'bs4'])  # 使用 pandas 直接解析 HTML 中的 <table> 標籤成為 DataFrame 列表
            if not dfs: continue  # 如果這個網頁找不到任何表格，跳過換下一個網址
                
            target_df = None  # 初始化目標資料表
            for df_item in dfs:  # 巡覽網頁中找到的所有表格
                if len(df_item) > 10:  # 假設油價表格通常資料列會大於 10 行
                    target_df = df_item  # 找到符合條件的表格，存入目標變數
                    break  # 找到就中斷迴圈
            
            if target_df is None: continue  # 如果還是沒找到合適的表格，跳過換下一個網址
            oil = target_df.copy()  # 複製找到的表格準備處理
            
            header_idx = -1  # 初始化標頭列的索引值位置
            for i in range(min(20, len(oil))):  # 只檢查表格的前 20 行，尋找哪一行才是真正的欄位名稱
                row_str = " ".join([str(x) for x in oil.iloc[i].values])  # 將該列的所有儲存格內容合併成一個長字串
                if "日期" in row_str and "92" in row_str:  # 如果字串中同時包含 "日期" 和 "92"
                    header_idx = i  # 認定這一行就是欄位名稱所在列
                    break  # 找到標頭就中斷迴圈
            
            if header_idx != -1:  # 如果有成功找到標頭列
                oil.columns = [str(x).strip() for x in oil.iloc[header_idx].values]  # 將那一行設定為資料表的欄位名稱，並去除字串前後空白
                oil = oil.iloc[header_idx+1:].copy()  # 將資料表內容截斷，只保留標頭列以下的實際數據
            
            rename_map = {}  # 建立一個空字典，準備做欄位重新命名對照表
            for col in oil.columns:  # 巡覽目前的欄位名稱
                if '日期' in col: rename_map[col] = '日期'  # 如果原欄位名包含 '日期'，統一改名為 '日期'
                elif '92' in col: rename_map[col] = '92'  # 如果原欄位名包含 '92'，統一改名為 '92'
                elif '95' in col: rename_map[col] = '95'  # 如果原欄位名包含 '95'，統一改名為 '95'
                elif '98' in col: rename_map[col] = '98'  # 如果原欄位名包含 '98'，統一改名為 '98'
                elif '柴' in col: rename_map[col] = '柴油'  # 如果原欄位名包含 '柴'，統一改名為 '柴油'
            
            oil = oil.rename(columns=rename_map)  # 套用對照表，重新命名欄位
            
            required_cols = ['日期', '92', '95', '98', '柴油']  # 定義我們最終需要的標準欄位清單
            for rc in required_cols:  # 檢查所需欄位是否齊全
                if rc not in oil.columns: oil[rc] = np.nan  # 如果缺漏某個欄位，就新增該欄位並填入空值(NaN)
            
            oil = oil[required_cols].copy()  # 只萃取我們需要的欄位，過濾掉不需要的雜訊欄位
            
            if '日期' in oil.columns:  # 確保 '日期' 欄位存在
                oil['日期'] = oil['日期'].astype(str).str.replace(r'\s+', '', regex=True)  # 將日期轉字串並用正規表示式去除所有空白字元
                oil['日期'] = pd.to_datetime(oil['日期'], errors='coerce')  # 轉換為標準時間格式，若轉換失敗(如非日期文字)則轉為 NaT (空值)
                oil = oil.dropna(subset=['日期'])  # 刪除日期為空值的那整列資料
                for c in ['92', '95', '98', '柴油']:  # 針對四種油價欄位
                    oil[c] = pd.to_numeric(oil[c], errors='coerce')  # 強制轉換為數值，若有無法轉換的文字則變為 NaN
                oil_dfs.append(oil)  # 將清理好的這份表格加入列表中

        except Exception as e:  # 捕捉這個網址處理過程中的任何錯誤
            print(f"   ❌ 爬蟲部分失敗: {e}")  # 印出錯誤訊息
    
    if not oil_dfs:  # 如果所有網址都跑完，列表還是空的 (完全沒抓到資料)
        print("⚠️ 無法獲取中油歷史油價，將使用全零填充")  # 印出警告
        return pd.DataFrame(columns=['日期', '92', '95', '98', '柴油'])  # 回傳帶有正確欄位結構但無資料的空 DataFrame
        
    return pd.concat(oil_dfs, ignore_index=True).drop_duplicates(subset='日期').sort_values('日期')  # 將列表中多個 DataFrame 上下合併，移除日期重複的資料，並依日期由舊到新排序後回傳

# ==========================================
# 3. 特徵工程 (已瘦身優化)  # 標示此區塊為建立機器學習輸入特徵
# ==========================================
def add_technical_features(df, price_col='布蘭特原油'): 
    df = df.copy() 
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce') 
    
    high = df['High'] if 'High' in df.columns else df[price_col] 
    low = df['Low'] if 'Low' in df.columns else df[price_col] 
    close = df[price_col]

    # --- 1. 原生計算 ATR (真實波幅) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()  # 14期

    # --- 2. 原生計算 布林通道寬度 (BB_WIDTH) ---
    sma = close.rolling(window=20).mean()
    std_dev = close.rolling(window=20).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    df['BB_WIDTH'] = (upper_band - lower_band) / sma * 100

    # --- 3. 原生計算 RSI (相對強弱指數) ---
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/14, adjust=False).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # ==========================================
    # 以下維持你原本優秀的複合特徵邏輯，完全不動
    # ==========================================
    df['Vol_Short'] = df[price_col].pct_change().rolling(5).std() 
    
    if '恐慌指數' in df.columns and '台幣匯率' in df.columns: 
        df['VIX_x_USD'] = df['恐慌指數'] * df['台幣匯率'] 
        
    if 'RSI' in df.columns: 
        vol = df['Vol_Short'] 
        df['Panic_Sell'] = vol * (100 - df['RSI']) 
    
    ma5 = df[price_col].rolling(5).mean() 
    df['MA5_Bias'] = (df[price_col] / (ma5 + 1e-9)) - 1 

    df['Momentum_Vol'] = df[price_col].diff() * (df['Vol_Short'] + 1e-9) 
    
    if not np.issubdtype(df['日期'].dtype, np.datetime64): 
        df['日期'] = pd.to_datetime(df['日期']) 
    day_of_year = df['日期'].dt.dayofyear 
    df['sin_365'] = np.sin(2 * np.pi * day_of_year / 365.25) 
    df['cos_365'] = np.cos(2 * np.pi * day_of_year / 365.25) 
    df['sin_90'] = np.sin(2 * np.pi * day_of_year / 91.3) 
    df['cos_90'] = np.cos(2 * np.pi * day_of_year / 91.3) 
    
    return df

def build_refined_dataset(asia_realtime=None):  # 定義主資料集建構函式，負責調度爬蟲與特徵
    print("🚀 啟動數據獲取程序 (安全模式 - 單線程)...")  # 印出流程開始提示
    
    # 1. 讀取現有資料庫  # 步驟說明：先拿舊資料，減少爬蟲負擔
    df_old = load_from_db()  # 從 SQLite 讀取過去存好的資料
    rename_dict = {'Date': '日期', 'index': '日期'}  # 建立可能需要統一改名的字典
    df_old = df_old.rename(columns={k: v for k, v in rename_dict.items() if k in df_old.columns})  # 將英文或舊稱統一改成 '日期'
    
    if not df_old.empty and '日期' in df_old.columns:  # 如果舊資料存在且格式正確
        df_old['日期'] = pd.to_datetime(df_old['日期'])  # 確保為時間格式
        last_date = df_old['日期'].max()  # 找出資料庫中最新的一筆日期
        start_date = last_date - pd.Timedelta(days=14)  # 設定爬蟲的起始日為最新日期的前 14 天 (重疊下載以防過去資料有修正)
        df_old = df_old[df_old['日期'] < start_date]  # 舊資料只保留到重疊日之前的部分
    else:  # 如果沒有舊資料 (第一次執行)
        last_date = pd.Timestamp('2000-01-01')  # 設定一個很早的基準日
        start_date = last_date  # 從頭開始抓取
    
    # 2. 下載新資料 (🔥 導入 Stooq 雙資料源 + 強制 Logging + Session 偽裝)
    tickers = { 
        "BZ=F": '布蘭特原油', 
        "USDTWD=X": '台幣匯率', "JPY=X": '日圓匯率', 
        "KRW=X": '韓元匯率', "^VIX": '恐慌指數'
    }
    
    stooq_map = {
        "BZ=F": "CB.F",        
        "USDTWD=X": "USDTWD.V", 
        "JPY=X": "USDJPY.V",   
        "KRW=X": "USDKRW.V",   
        "^VIX": "VIX.US"       
    }

    new_data_frames = [] 
    ticker_list = list(tickers.keys())
    
    # 建立強化的偽裝 Session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    })
    
    # 改用 logging 確保日誌能在 GitHub Actions 正常輸出
    logging.info("📦 嘗試批量下載 Yahoo Finance 首選資料 (已掛載偽裝 Session)...")
    
    try:
        # 👉 關鍵修復：這裡補上 session=session，繞過預設的機器人阻擋
        bulk_data = yf.download(ticker_list, start=start_date, progress=False, timeout=20, session=session)
    except Exception as e:
        logging.error(f"❌ Yahoo 批量下載發生異常: {e}")
        bulk_data = pd.DataFrame() # 若完全崩潰，產生空表讓程式順利進入備援機制

    for ticker_code, name in tickers.items():
        price_df = None
        
        # --- 嘗試 1：從 Yahoo 擷取資料 ---
        if not bulk_data.empty:
            try:
                if isinstance(bulk_data.columns, pd.MultiIndex):
                    if 'Close' in bulk_data.columns.get_level_values(0):
                        price_series = bulk_data['Close'][ticker_code]
                    elif 'Close' in bulk_data.columns.get_level_values(1):
                        price_series = bulk_data.xs('Close', level=1, axis=1)[ticker_code]
                    else:
                        price_series = pd.Series(dtype=float)
                else:
                    price_series = bulk_data[ticker_code] 
                
                price_series = price_series.dropna()
                if not price_series.empty:
                    price_df = price_series.to_frame(name=name)
                    price_df.index.name = '日期'
                    price_df = price_df.reset_index()
                    logging.info(f"   ✅ Yahoo 下載成功: {name}")
            except Exception:
                pass 
                
        # --- 嘗試 2：啟動 Stooq 備援機制 ---
        if price_df is None or price_df.empty:
            stooq_code = stooq_map.get(ticker_code)
            logging.warning(f"   ⚠️ Yahoo 失敗，切換 Stooq 備用源: {name} ({stooq_code})...")
            try:
                df_stooq = web.DataReader(stooq_code, 'stooq', start=start_date)
                if not df_stooq.empty:
                    df_stooq = df_stooq.sort_index()
                    price_df = df_stooq[['Close']].copy()
                    price_df.columns = [name]
                    price_df.index.name = '日期'
                    price_df = price_df.reset_index()
                    logging.info(f"   ✅ Stooq 備用源救援成功: {name}")
            except Exception as e:
                logging.error(f"   ❌ Stooq 備用源也失敗: {e}")
                
        # --- 3. 防呆與收尾 ---
        if price_df is not None and not price_df.empty:
            price_df['日期'] = pd.to_datetime(price_df['日期'])
            last_val = price_df[name].iloc[-1]
            
            if name == '布蘭特原油' and (last_val > 200 or last_val < 10):
                logging.warning(f"   ⚠️ {name} 數值異常 ({last_val})，自動捨棄")
                continue
            if name == '台幣匯率' and (last_val > 50 or last_val < 20):
                logging.warning(f"   ⚠️ {name} 數值異常 ({last_val})，自動捨棄")
                continue
                
            new_data_frames.append(price_df)
        else:
            logging.error(f"   🚨 {name} 所有資料源皆獲取失敗。")
        
    # 3. 合併資料  # 步驟說明：將各種不同來源的商品資料表根據日期合併
    if new_data_frames:  # 如果有成功抓到新資料
        df_new_period = reduce(lambda left, right: pd.merge(left, right, on='日期', how='outer'), new_data_frames)  # 使用 reduce 和 merge 連續執行「外部合併」(outer join)，對齊相同日期的資料
        
        if df_old.empty:  # 如果原本沒有舊資料
            df_full = df_new_period  # 全新資料就是完整資料
        else:  # 如果有舊資料
            df_full = pd.concat([df_old, df_new_period], axis=0, ignore_index=True)  # 將舊資料與新資料上下拼接在一起
            
        if not df_full.empty and '日期' in df_full.columns:  # 合併後再次確保格式
            df_full = df_full.sort_values('日期').drop_duplicates(subset=['日期'], keep='last')  # 依日期排序，若有重複日期保留最後一筆(最新抓的覆蓋舊的)
            save_to_db(df_full)  # 將最新完整的資料表存回 SQLite 資料庫備份
        else:
            print("⚠️ 合併後資料為空，無法儲存")  # 錯誤提示
    else:  # 如果完全沒抓到新資料 (例如週末沒開盤)
        print("⚠️ 無新資料可更新，使用現有資料庫")  # 提示訊息
        df_full = df_old  # 直接使用現有舊資料

    if df_full.empty or '日期' not in df_full.columns:  # 最終防呆
        print("❌ 警告：數據集嚴重缺失，嘗試回傳空結構")  # 發生嚴重錯誤
        return pd.DataFrame(columns=['日期', '布蘭特原油'])  # 回傳一個有基本欄位的空表避免後續程式當掉

    # ==============================================================================
    # 4. 後處理 (修正版：讓亞鄰數據更逼真，製造壓力訊號)  # 步驟說明：處理外匯極端值並補齊中油歷史
    # ==============================================================================
    if '日圓匯率' in df_full.columns:  # 檢查是否有日圓欄位
        valid_jpy = df_full['日圓匯率'].dropna()  # 剔除空值
        if not valid_jpy.empty and valid_jpy.tail(500).median() > 500:  # 檢查最近 500 筆的中位數，如果大於 500 (可能是抓到每 100 日圓兌美金的報價)
            df_full['日圓匯率'] = df_full['日圓匯率'] / 100  # 將數值除以 100，修正回正常的 USD/JPY 或 JPY/TWD 比例範圍

    df_cpc = fetch_cpc_oil_history()  # 呼叫爬蟲獲取台灣中油歷史油價
    df_full = df_full.sort_values('日期')  # 確保總表依日期排序
    
    if not df_cpc.empty:  # 如果有成功抓到中油歷史
        df_full = pd.merge_asof(df_full, df_cpc.sort_values('日期'), on='日期')  # 使用 merge_asof (近似合併)，將中油油價對齊到最近期的全球市場資料日

    # 亞鄰數據初始化  # 建立日韓代理變數的欄位
    if 'JP_Price_Proxy' not in df_full.columns: df_full['JP_Price_Proxy'] = np.nan  # 初始化日本代理價為 NaN
    if 'KR_Price_Proxy' not in df_full.columns: df_full['KR_Price_Proxy'] = np.nan  # 初始化韓國代理價為 NaN
    
    if asia_realtime and asia_realtime.get('status') == 'SUCCESS':  # 如果參數有傳入即時抓到的日韓資料且成功
        df_full.loc[df_full.index[-1], 'JP_Price_Proxy'] = float(asia_realtime['JP_val'])  # 將最新的一筆資料填入日本價格
        df_full.loc[df_full.index[-1], 'KR_Price_Proxy'] = float(asia_realtime['KR_val'])  # 將最新的一筆資料填入韓國價格
    
    # 防呆：處理異常極端值
    mask_jp_huge = df_full['JP_Price_Proxy'] > 3000   # 標記出日本油價大於 3000 的異常列 (可能是單位錯誤)
    df_full.loc[mask_jp_huge, 'JP_Price_Proxy'] = np.nan # 將這些異常值清空為 NaN，等待稍後的機制重新填補
    
    # --- 🔥 [關鍵修改] 讓亞鄰價格 "貼近" 成本，製造壓力 ---  # 步驟說明：如果歷史資料缺乏真實的日韓油價，就用原油成本來「模擬」合理的日韓價格
    df_full['temp_cost'] = df_full['布蘭特原油'] * df_full['台幣匯率']  # 建立臨時欄位：計算純原油的台幣成本
    
    # 1. 計算歷史比例 (如果有的話)
    if 'JP_Price_Proxy' in df_full.columns:
        historical_ratio = df_full['JP_Price_Proxy'] / (df_full['temp_cost'] + 1e-6)  # 計算有資料時，日本零售價是純原油成本的幾倍
        
        # ⚠️ 修改：原本 fillna(2.0) 太高了，改成 1.05 (只高 5%)
        # 這樣一來，只要油價稍微波動，就會撞到天花板  # 開發者註解說明政策模擬意圖
        base_ratio = historical_ratio.rolling(52, min_periods=1).mean().ffill().fillna(1.05)  # 取 52 週的移動平均，並用前值填充 (ffill)，最後沒得填就預設 1.05 倍
        
        # ⚠️ 修改：加入隨機波動 (Noise)，模擬真實市場的不確定性
        # 讓倍數在 0.95 ~ 1.15 之間跳動
        noise = np.random.normal(0, 0.05, size=len(df_full))  # 產生平均為 0、標準差 0.05 的常態分佈隨機雜訊
        dynamic_ratio = base_ratio + noise  # 將雜訊疊加到基礎倍率上，讓模擬價格不要太過平滑失真
        
        # 填補缺失值
        df_full['JP_Price_Proxy'] = df_full['JP_Price_Proxy'].fillna(df_full['temp_cost'] * dynamic_ratio)  # 針對原本是 NaN 的日本欄位，用模擬的「原油成本 * 動態倍率」填補回去
        
        # 韓國同理
        if 'KR_Price_Proxy' in df_full.columns:
            # 韓國通常比日本高一點，設 1.1 倍
            kr_ratio = 1.1 + np.random.normal(0, 0.05, size=len(df_full))  # 產生韓國專用的隨機倍率 (中心值為 1.1)
            df_full['KR_Price_Proxy'] = df_full['KR_Price_Proxy'].fillna(df_full['temp_cost'] * kr_ratio)  # 填補韓國的缺失值
            
    df_full.drop(columns=['temp_cost'], inplace=True)  # 模擬完成，把這個臨時計算用的成本欄位刪除

    # ==============================================================================
    # 5. 核心特徵 (修正版：放寬判定標準)  # 步驟說明：建立政策指標與進階特徵
    # ==============================================================================
    theoretical_cost_twd = df_full['布蘭特原油'] * df_full['台幣匯率']  # 再次計算理論原油台幣成本 (未稅)
    asia_min_price = df_full[['JP_Price_Proxy', 'KR_Price_Proxy']].min(axis=1)  # 找出日韓雙方在同一天的「最低價」，作為亞鄰天花板
    
    # 🔥 [修改] Ceiling_Pressure 計算方式
    # 計算「成本」佔「天花板」的百分比。如果 > 1.0 代表撞到天花板
    df_full['Ceiling_Ratio'] = theoretical_cost_twd / (asia_min_price + 1e-9)  # 成本除以天花板的比例
    # 正值代表成本高於天花板 (需要被吸收)，負值代表安全
    df_full['Ceiling_Gap'] = theoretical_cost_twd - asia_min_price  # 絕對價差
    # 定義壓力：只要這個比率 > 0.9 (接近天花板 90%) 就開始有壓力值
    df_full['Ceiling_Pressure'] = df_full['Ceiling_Ratio'].apply(lambda x: (x - 0.9) if x > 0.9 else 0)  # 使用 lambda 函式：超過 0.9 就產出壓力值，否則為 0

    df_full['週五偏離度'] = df_full['布蘭特原油'] / (df_full['布蘭特原油'].rolling(5).mean() + 1e-9)  # 計算當日油價與過去一週平均價格的比值 (偏離度)
    df_full['oil_diff_lag1'] = df_full['布蘭特原油'].diff().shift(1)   # 取昨天的單日價格變動量 (延遲 1 期)
    df_full['oil_diff_lag2'] = df_full['布蘭特原油'].diff().shift(2)  # 取前天的單日價格變動量 (延遲 2 期)
    
    if '韓元匯率' in df_full.columns and '日圓匯率' in df_full.columns:  # 如果兩種匯率都在
         df_full['亞鄰壓力'] = (df_full['台幣匯率'] / (df_full['日圓匯率'] + 1e-6)) + (df_full['台幣匯率'] / (df_full['韓元匯率'] + 1e-6))  # 建立一個簡單的跨國匯差壓力指數
    else:
         df_full['亞鄰壓力'] = 0  # 缺少資料則設為 0

    # 🔥 [修改] 政策凍漲風險
    # 直接利用上面算好的 Ceiling_Ratio
    # 如果成本 > 天花板 (Ratio > 1.0)，則風險為 1 (高風險)
    # 如果成本接近天花板 (Ratio > 0.95)，則風險為 0.5 (中風險)
    def assess_risk(ratio):  # 定義子函式判定風險級距
        if ratio >= 1.0: return 1.0  # 破 100% 回傳 1.0
        elif ratio >= 0.95: return 0.5  # 破 95% 回傳 0.5
        else: return 0.0  # 安全區間回傳 0.0
        
    df_full['政策凍漲風險'] = df_full['Ceiling_Ratio'].apply(assess_risk)  # 將判定函式套用到欄位上產生新特徵

    df_full['weekday'] = df_full['日期'].dt.weekday  # 提取星期幾 (0 是週一，6 是週日)，對中油在週五拍板、週一實施的規則很重要
    df_full = add_technical_features(df_full, '布蘭特原油')  # 呼叫前面寫好的函式，加入所有技術指標特徵
    
    return df_full.ffill().fillna(0)  # 最後處理：先用前一天的數值填補(ffill)缺值，剩下的空值全部補 0 後回傳

def get_finbert_pipeline():  # 定義函式，用來初始化 FinBERT 模型
    """
    載入 FinBERT 模型，專門用於金融情緒分析
    """
    if not HAS_NLP: return None  # 如果開頭檢查發現沒有安裝相關套件，直接回傳空值不執行
    try:
        # 使用 ProsusAI 的 FinBERT，這是金融界公認的 Benchmark  # 開發者註解選擇此模型的原因
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)  # 下載/載入 FinBERT 預訓練模型，並指定運算裝置 (GPU/CPU)
        return pipe  # 回傳建立好的 NLP 管線物件
    except Exception as e:  # 若載入失敗 (可能因為網路或記憶體不足)
        print(f"❌ FinBERT 載入失敗: {e}")  # 印出錯誤
        return None  # 回傳空值

def build_sentiment_features(df, mode="full"):  # 定義函式，負責產生「市場情緒」特徵
    """
    建立情緒特徵：訓練時使用模擬數據，即時模式使用真實 NLP 數據
    """
    print("🧠 啟動 NLP 語意情緒分析引擎 (ProsusAI/FinBERT)...")  # 印出提示
    df['新聞情緒'] = 0.0  # 先建立一個名為 '新聞情緒' 的欄位，預設全填 0

    if mode != "realtime":  # 如果不是「即時 (realtime)」模式 (例如跑回測或歷史訓練)
        # --- 訓練模式 (歷史模擬) ---
        # 由於沒有過去 10 年的每日新聞 RSS，我們使用「價格行為」來標記情緒  # 解釋為什麼要模擬 (受限於歷史新聞資料難以取得)
        # 邏輯：大跌 = 負面情緒，大漲 = 正面情緒
        # 這讓模型學習到 "Sentiment" 特徵與價格波動的關聯
        if '布蘭特原油' in df.columns:
            # 計算 5 日波動率與方向
            pct_change = df['布蘭特原油'].pct_change()  # 算每日報酬率
            volatility = pct_change.rolling(5).std()  # 算 5 日歷史波動率
            
            # 模擬情緒分數：方向 * 波動強度 * 放大係數
            # 這樣模型會學到：當這個欄位是負的大數值時，後市看跌
            simulated_sentiment = (pct_change / (volatility + 1e-9)) * 0.5  # 波動大且下跌，分數就會是深度的負值；反之亦然。乘 0.5 稍微縮放範圍
            df['新聞情緒'] = simulated_sentiment.fillna(0).clip(-1, 1) # 將計算結果的空值補 0，並強制將數值範圍限制 (clip) 在 -1 到 1 之間
            print("   ℹ️ [歷史模式] 使用波動率模擬情緒特徵，供模型訓練使用")

    else:  # 如果是 "realtime" 模式 (要預測今天的油價)
        # --- 即時模式 (真槍實彈) ---
        # 抓取 Google News RSS，並用 FinBERT 進行真實語意分析  # 說明即時模式的做法
        rss_url = "https://news.google.com/rss/search?q=oil+price+energy+market&hl=en-US&gl=US&ceid=US:en"  # 設定抓取 Google 英文新聞 (關鍵字：油價、能源市場) 的 RSS 網址
        try:
            resp = requests.get(rss_url, headers=HEADERS, timeout=10)  # 發送網路請求獲取 RSS
            if resp.status_code == 200:  # 如果成功獲取網頁 (HTTP 狀態碼 200)
                feed = feedparser.parse(resp.content)  # 使用 feedparser 解析 RSS 內容
                # 抓取前 15 則新聞標題
                titles = [e.title for e in feed.entries[:15]]   # 使用列表推導式，萃取前 15 則新聞的 'title' 屬性
                
                if not titles:  # 如果解析出來標題是空的
                    print("   ⚠️ 無法抓取到新聞標題")
                else:
                    nlp_pipe = get_finbert_pipeline()  # 呼叫函式初始化 NLP 模型
                    sentiment_score = 0.0  # 初始化總計的情緒分數
                    
                    if nlp_pipe:  # 確保模型有成功載入
                        print(f"   📰 正在分析 {len(titles)} 則國際能源新聞...")
                        results = nlp_pipe(titles)  # 丟入 15 則標題，讓 FinBERT 進行推論，會回傳結果列表
                        
                        for res in results:  # 針對推論回傳的結果一筆一筆看
                            # FinBERT 回傳 label (positive, negative, neutral) 和 score (信心度)
                            label = res['label']  # 取出情緒標籤
                            conf = res['score']  # 取出 AI 對這項判斷的信心度 (0~1)
                            
                            if label == 'positive':  # 如果判斷為正面新聞
                                sentiment_score += conf # 總分加上其信心度
                            elif label == 'negative':  # 如果判斷為負面新聞
                                sentiment_score -= conf # 總分扣掉其信心度
                                # 負面新聞對市場影響通常大於正面 (恐慌效應)，加權 1.2 倍  # 金融行為學常理
                                sentiment_score -= (conf * 0.2)   # 額外多扣 20% 的分數作為懲罰權重
                            # neutral 不加分  # 如果是中性新聞 (neutral) 就不做分數加減
                        
                        # 正規化：將總分壓縮到 -1 ~ 1 之間
                        # 假設分析 10 則新聞，極限分數約 +/- 10，用 tanh 壓縮  # 解釋為何使用 tanh 函數
                        final_score = np.tanh(sentiment_score / 3.0)  # 使用雙曲正切函數 (tanh) 將數值平滑地壓縮在 -1 (極度悲觀) 到 1 (極度樂觀) 之間
                        
                        print(f"   📊 當前市場情緒指數: {final_score:.4f} (FinBERT Weighted)")  # 印出計算出來的最終市場情緒指標
                        
                        # 將最新一天的情緒分數填入
                        if not df.empty:
                            df.loc[df.index[-1], '新聞情緒'] = float(final_score)  # 把算出的分數塞進資料表「最後一列」(即今天) 的 '新聞情緒' 欄位
                    else:
                        print("   ⚠️ NLP 模型未載入，跳過分析")
                        
        except Exception as e:
            print(f"⚠️ 新聞情緒分析失敗: {e}")  # 捕捉抓新聞或推論時的任何錯誤

    # 計算情緒變動 (Sentiment Momentum)
    # 這通常比絕對值更有預測力  # 金融領域中，情緒「轉變的方向」往往比「當下有多高」更能預測漲跌
    df['情緒變動'] = df['新聞情緒'].diff().fillna(0)  # 計算今天和昨天情緒分數的差值，存為新特徵 '情緒變動'
    
    return df.ffill().fillna(0)  # 回傳處理完畢的資料表，缺值一律補 0

def build_and_save_features(mode="full"):  # 定義主程式執行函式，負責串接所有流程並儲存檔案
    print("🚀 建立特徵資料集...")
    asia_realtime = fetch_asia_neighbor_prices()  # 步驟 1：先去抓日韓即時油價
    df_raw = build_refined_dataset(asia_realtime=asia_realtime)  # 步驟 2：執行主要資料收集與爬蟲管線，建立基礎資料集
    
    if df_raw.empty: raise RuntimeError("❌ 數據集建立失敗")  # 如果資料是空的，拋出嚴重錯誤終止程式

    df = build_sentiment_features(df_raw, mode=mode)  # 步驟 3：把剛建好的資料集丟進去算 NLP 與情緒特徵

    if mode == "realtime":  # 如果是即時模式
        df = df.tail(365)   # 為了提升效能與減少傳輸，只保留最近一年的資料供模型推論即可

    parquet_path = DATA_DIR / f"features_{mode}.parquet"  # 設定 Parquet 格式的存檔路徑 (Parquet 讀寫速度快，適合大數據)
    df.to_parquet(parquet_path, index=False)  # 將資料表存成 Parquet 檔，不存入 index
    
    excel_path = DATA_DIR / f"features_{mode}.xlsx"  # 設定 Excel 格式的存檔路徑 (方便人類開啟檢視)
    df.to_excel(excel_path, index=False, engine="openpyxl")  # 使用 openpyxl 引擎將資料表存成 Excel 檔

    return parquet_path, excel_path  # 回傳產生的檔案路徑，方便主程式後續調用

if __name__ == "__main__":  # Python 的標準寫法，當「直接執行」這支檔案時，才會觸發以下的程式碼

    build_and_save_features(mode="full")  # 呼叫主函式，以 "full" 模式執行，重建包含完整歷史的特徵資料集





