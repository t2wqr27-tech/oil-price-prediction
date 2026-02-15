# -*- coding: utf-8 -*-  # 指定程式碼檔案編碼為 UTF-8，確保能正確讀取中文
import os  # 載入作業系統模組，用於處理檔案與目錄路徑
import glob  # 載入 glob 模組，用於尋找符合特定規則的檔案路徑
import shutil  # 載入 shutil 模組，用於進行高階檔案操作（如複製、移動檔案）
import time  # 載入 time 模組，用於處理時間與產生時間戳記
import numpy as np  # 載入 numpy 模組並縮寫為 np，用於高效的數值與矩陣運算
import pandas as pd  # 載入 pandas 模組並縮寫為 pd，用於資料表（DataFrame）的處理與分析

# ⚠️ quick_recent_mae 需要 hybrid_predict 已被 import  # 開發者的提醒註解，標示相依的模組需要先載入

def prepare_output_folder(folder):  # 定義一個函式，用來準備輸出資料夾並備份舊檔案
    os.makedirs(folder, exist_ok=True)  # 建立目標資料夾，如果資料夾已經存在則不會報錯（exist_ok=True）
    
    # 建立備份資料夾 (例如: 週五精準預測成果/History/20231027_1530)  # 說明接下來幾行的目的
    timestamp = time.strftime("%Y%m%d_%H%M")  # 取得當下時間，並格式化為 "年月日時分"（例如：20231027_1530）
    history_dir = os.path.join(folder, "History", timestamp)  # 組合出備份資料夾的完整路徑
    
    # 找出所有檔案 (排除 History 資料夾本身)  # 說明接下來幾行的目的
    files_to_move = (  # 將所有需要移動的檔案路徑整理成一個串列（List）
        glob.glob(os.path.join(folder, "*.png")) +  # 找出資料夾內所有的 .png 圖片檔
        glob.glob(os.path.join(folder, "*.jpg")) +  # 找出資料夾內所有的 .jpg 圖片檔
        glob.glob(os.path.join(folder, "*.xlsx")) +  # 找出資料夾內所有的 .xlsx Excel 檔
        glob.glob(os.path.join(folder, "*.txt"))    # 找出資料夾內所有的 .txt 文字檔
    )  # 結束串列合併
    
    if files_to_move:  # 判斷如果 files_to_move 裡面有檔案（串列不為空）
        os.makedirs(history_dir, exist_ok=True)  # 建立剛剛組合好的歷史備份資料夾
        print(f"📦 正在備份舊檔案至: {history_dir} ...")  # 在終端機印出提示訊息，告知正在備份
        for f in files_to_move:  # 使用迴圈，逐一處理串列中的每一個檔案
            try:  # 嘗試執行以下程式碼，若發生錯誤會跳到 except 處理
                # 移動檔案  # 說明下一行的動作
                shutil.move(f, os.path.join(history_dir, os.path.basename(f)))  # 將檔案移動到歷史備份資料夾內
            except Exception as e:  # 捕捉移動過程中發生的任何例外錯誤，並將錯誤訊息存入變數 e
                print(f"⚠️ 無法備份檔案 {f}: {e}")  # 如果移動失敗，印出警告訊息與錯誤原因

def price_change(new, old):  # 定義一個函式，用來計算價格的變動量與變動百分比
    if old == 0 or pd.isna(old):  # 如果舊價格為 0，或者是空值（NaN）
        return new - old, np.nan  # 回傳價差，因為舊價格無效，所以百分比回傳 NaN（避免除以零錯誤）
    return new - old, (new - old) / old * 100  # 如果舊價格正常，回傳「價差」與「變動百分比（%）」

def get_action_advice(pred, current, threshold):  # 定義一個函式，根據預測價格與當前價格給出行動建議
    diff = pred - current  # 計算預測價格與當前價格的差值
    if abs(diff) > threshold:  # 如果差值的絕對值大於設定的門檻值
        return f"⚠️ 非常態變動 {diff:.2f}"  # 回傳非常態變動的警告，並顯示差值（取小數點後兩位）
    elif diff > 0:  # 如果差值大於 0（且沒有超過門檻）
        return f"🔺小幅上漲 {diff:.2f}"  # 回傳小幅上漲的提示
    elif diff < 0:  # 如果差值小於 0（且沒有超過門檻）
        return f"🔻小幅下跌 {diff:.2f}"  # 回傳小幅下跌的提示
    else:  # 如果差值剛好等於 0
        return f"─ 波動不大 {diff:.2f}"   # 回傳波動不大的提示

def mark_abnormal_weeks(df, oil, q=0.7, pct_threshold=0.03):  # 定義函式，用來標記資料表中特定油品每週的異常波動
    if oil not in df.columns:  # 檢查指定的油品名稱是否存在於資料表的欄位中
        raise ValueError(f"{oil} 不存在於資料中")  # 如果不存在，主動拋出錯誤終止程式
    df = df.copy()  # 複製一份資料表，避免修改到原始傳入的資料（這是一個好習慣）
    
    # 週變動（絕對值）  # 說明計算週變動的區塊
    df['weekly_change'] = df[oil].diff()  # 計算該油品當期與前一期的數值差，存入新欄位
    df['weekly_abs_change'] = df['weekly_change'].abs()  # 將剛剛算出的差值取絕對值，存入新欄位
    
    # 百分比變動  # 說明計算百分比變動的區塊
    denom = df[oil].shift(1).replace(0, np.nan)  # 取得前一期的數值作為分母，並將 0 替換為 NaN 避免除以零
    df['weekly_pct_change'] = df['weekly_change'] / denom  # 計算變動百分比（變動量 / 前一期數值）
    
    # 動態門檻（歷史分位）  # 說明計算動態門檻的區塊
    df['dyn_threshold'] = rolling_threshold(df['weekly_abs_change'], q)  # 呼叫另一函式計算滾動門檻，存入新欄位
    threshold = df['dyn_threshold'].dropna().iloc[-1]  # 取得剔除空值後，最新（最後一筆）的門檻數值
    
    # 非常態判定  # 說明標記異常的邏輯區塊
    df['abnormal_flag'] = (  # 建立一個新欄位，用來存儲異常標記（1或0）
        (df['weekly_abs_change'] >= threshold) |  # 條件一：絕對變動量大於等於動態門檻，或者(|)
        (df['weekly_pct_change'].abs() >= pct_threshold)  # 條件二：變動百分比的絕對值大於等於設定的百分比門檻
    ).astype(int)  # 將布林值(True/False)轉換為整數(1/0)
    
    threshold = threshold if not np.isnan(threshold) else df['weekly_abs_change'].quantile(q)  # 如果算出的最新門檻不是空值就用它，否則用整體的 q 分位數作為備用門檻
    return df, threshold  # 回傳處理好的資料表與最後使用的門檻值

def rolling_threshold(series, q=0.7, win=104):  # 定義函式，計算時間序列的滾動分位數門檻
    """  # 多行字串的開頭，通常用於撰寫函式的說明文件 (Docstring)
    動態歷史門檻（避免未來資訊洩漏）  # 說明此函式是為了解決時間序列中，不能用到未來資料來計算門檻的問題
    """  # 多行字串的結尾
    return series.rolling(win, min_periods=win//2).quantile(q)  # 使用滾動視窗(大小為win)，至少需要一半(win//2)的資料點才計算，然後算出第 q 分位數並回傳

def cpc_formula(df_slice, oil_type=None):  # 定義函式，模擬中油的油價計算公式
    """  # 函式的說明文件開頭
    【修正版】中油 7D3B 公式代理計算  # 說明這是一個修正版本的中油 7D3B（7成杜拜、3成布蘭特）計算公式
    🔥 新增：柴油冬季裂解價差 (Diesel Crack Spread) 修正  # 特別註記加入了柴油的季節性價差考量
    解決「原油跌但柴油漲」的公式脫鉤問題  # 說明這次修正解決了什麼痛點
    """  # 函式的說明文件結尾
    
    # 1. 取得最後一筆數據  # 步驟一說明
    if len(df_slice) == 0: return np.nan  # 如果傳入的資料表切片是空的，直接回傳空值(NaN)
    row = df_slice.iloc[-1]  # 取得資料表切片的最後一列資料（最新的一筆）
    
    try:  # 嘗試執行以下取值的程式碼
        # 取得原料價格 (加入防呆，避免抓到 0)  # 說明這段是為了取得原油與匯率，並防止異常值
        brent = float(row.get('布蘭特原油', 0))  # 取得該列的布蘭特原油價格並轉為浮點數，若找不到該欄位則預設為 0
        usd_twd = float(row.get('台幣匯率', 32.0))  # 取得該列的台幣匯率並轉為浮點數，若找不到預設為 32.0
        
        # 防呆：如果數據異常 (例如油價 < 50)，嘗試往前找一期  # 說明除錯邏輯
        if brent < 50 or usd_twd < 20:  # 如果布蘭特原油低於 50 或台幣匯率低於 20，判斷為異常數值
            if len(df_slice) > 1:  # 確保資料表中還有上一筆資料可以拿
                prev = df_slice.iloc[-2]  # 取得倒數第二列的資料（前一期）
                brent = float(prev.get('布蘭特原油', brent))  # 使用前一期的原油價格，若沒有則維持原本的異常值
                usd_twd = float(prev.get('台幣匯率', usd_twd))  # 使用前一期的台幣匯率，若沒有則維持原本的異常值
    
    except Exception as e:  # 捕捉取值或轉型時可能發生的任何錯誤
        print(f"❌ 公式計算錯誤: {e}")  # 印出錯誤訊息
        return np.nan  # 發生錯誤時回傳空值，避免程式崩潰

    # 2. 7D3B 估算 (70% 布蘭特 + 30% 杜拜)  # 步驟二說明
    # 用 0.98 係數逼近 7D3B 均價 (比純 Brent 略低)  # 解釋為什麼要乘上 0.98
    raw_oil_usd = brent * 0.98   # 將布蘭特原油價格乘上 0.98，模擬出 7D3B 的美金成本
    
    # 3. 換算每公升台幣成本 (159 公升 = 1 桶)  # 步驟三說明
    # 範例：(85 * 32) / 159 = 17.1 元  # 留給開發者看的計算範例
    raw_cost_twd = (raw_oil_usd * usd_twd) / 159.0  # 將美金/桶換算成台幣/公升
    
    # 4. 加上固定成本與稅費 (貨物稅、加油站毛利、土污費等)  # 步驟四說明
    # 🔥🔥🔥 關鍵修正：針對柴油加入「季節性價差」 🔥🔥🔥  # 強調這段邏輯的重要性
    
    if oil_type == '柴油':  # 判斷目前計算的油品是否為柴油
        # 柴油基礎稅費較低 (約 9.0)，但冬季會有 Crack Spread 溢價  # 解釋柴油的特性
        # 目前冬天溢價約 3.5 元/公升 (反應國際熱燃油需求)  # 補充說明溢價金額來源
        fixed_cost = 9.0   # 設定柴油的固定稅費為 9.0 元
        seasonal_premium = 3.5 # 設定冬季的柴油季節性溢價為 3.5 元
    else:  # 如果不是柴油（例如 95 汽油）
        fixed_cost = 12.5 # 設定汽油的固定稅費為 12.5 元（因為貨物稅較高）
        seasonal_premium = 0.0 # 汽油沒有冬季的熱燃油溢價，所以設為 0
    
    # 5. 計算稅前價格並加上 5% 營業稅  # 步驟五說明
    # 公式：(原料 + 稅費 + 溢價) * 1.05  # 列出最終計算公式
    estimated_price = (raw_cost_twd + fixed_cost + seasonal_premium) * 1.05  # 將所有成本加總後，乘上 1.05 加入營業稅
    
    return round(estimated_price, 1)  # 回傳最終計算結果，並四捨五入到小數點第一位

def analyze_extreme_events(df, oil_type='95'):  # 定義函式，用於分析資料表中出現極端波動的歷史事件
    """  # 函式的說明文件開頭
    自動識別歷史重大波動週（以 CPC 週油價為準）  # 說明此函式的功能，主要是找出中油歷史油價的重大波動
    """  # 函式的說明文件結尾
    df = df.copy()  # 複製一份資料表，避免動到原始資料
    df['日期'] = pd.to_datetime(df['日期'])  # 將「日期」欄位轉換為 pandas 的標準時間格式，方便後續時間序列操作

    # ✅ 關鍵：轉成週資料（只留週五）  # 提示這段操作的核心目的
    df_weekly = (  # 建立新的變數儲存轉換為週頻率的資料表
        df  # 從剛複製的資料表開始
        .sort_values('日期')  # 先依照「日期」欄位由舊到新排序
        .resample('W-FRI', on='日期')  # 將資料重新取樣成以「每週五」為單位的頻率
        .last()  # 取該週的最後一筆資料（通常就是週五那天的資料）
        .reset_index()  # 將日期索引重置回一般欄位，讓資料表結構恢復正常
    )

    if oil_type not in df_weekly.columns:  # 檢查指定的油品名稱是否在轉換後的資料表中
        raise ValueError(f"{oil_type} 不存在於資料中")  # 如果不在，拋出錯誤提示

    # 週變動率  # 說明接下來要算週變動率
    df_weekly['pct_change'] = df_weekly[oil_type].pct_change()  # 針對目標油品，計算本週與上週的百分比變動

    std = df_weekly['pct_change'].std()  # 計算這個變動率欄位的標準差（代表波動程度）
    
    if pd.isna(std) or std == 0:  # 如果標準差算出來是空值，或者等於 0（資料沒有波動）
        print("⚠️ 週資料不足，無法偵測重大事件")  # 印出警告訊息
        return df_weekly.iloc[0:0]  # 回傳一個沒有資料，但是欄位結構正確的空資料表

    extreme_mask = df_weekly['pct_change'].abs() > (std * 1.5)  # 建立一個布林遮罩（True/False），篩選出絕對變動率超過 1.5 倍標準差的極端值
    extreme_periods = df_weekly[extreme_mask]  # 將遮罩套用回資料表，只留下極端波動的那幾個禮拜的資料

    print(f"🚨 偵測到 {len(extreme_periods)} 個重大波動週")  # 印出找到了幾個重大波動週的結果
    return extreme_periods  # 回傳這些重大波動週的資料表

def select_decision_rows(df, mode="weekly"):  # 定義函式，用來根據不同模式篩選出要用來決策的資料列
    """  # 函式說明文件開頭
    weekly   → 只用週五（原本邏輯）  # 說明 weekly 模式的功能
    realtime → 用最新一筆資料（即時）  # 說明 realtime 模式的功能
    """  # 函式說明文件結尾
    if mode == "weekly":  # 如果傳入的模式是 "weekly"
        if 'weekday' not in df.columns:  # 檢查資料表中有沒有 'weekday' 這個標註星期幾的欄位
            raise ValueError("weekly 模式需要 weekday 欄位")  # 如果沒有，拋出錯誤要求提供
        return df[df['weekday'] == 4].copy().reset_index(drop=True)  # 篩選出星期五（weekday 為 4）的資料，複製後重置索引並回傳
    elif mode == "realtime":  # 如果傳入的模式是 "realtime"
        return df.copy().reset_index(drop=True)  # 不做特別篩選，直接複製整份資料表、重置索引後回傳
    else:  # 如果傳入的模式既不是 weekly 也不是 realtime
        raise ValueError("MODE 必須是 'weekly' 或 'realtime'")  # 拋出錯誤提示模式參數不正確



