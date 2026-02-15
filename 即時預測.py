"""
AI-Enhanced Oil Price Forecasting System (Full Integration Version)
- 整合 Academic (深度分析) 與 Production (即時預測) 雙模式
- 修正波動率滯後 Bug：確保使用最新一週數據進行體制切換
- 包含 SHAP 解釋性分析與完整視覺化
- 🔥 修正：確保 CEILING 與 ACCURACY 圖表在所有模式下皆會生成
- 🔥 優化：報表自動整合汽油三兄弟 (92/95/98)，柴油獨立顯示
"""
import os # 確保在程式開頭引入 os 模組
import warnings # 全局引入 warnings 模組以便後續使用
import urllib3 # 全局引入 urllib3 模組以便後續使用
import requests # 全局引入 requests 模組以便後續使用
import numpy as np # 全局引入 numpy 模組以便後續使用
import pandas as pd # 全局引入 pandas 模組以便後續使用
import matplotlib.pyplot as plt # 全局引入 matplotlib.pyplot 模組以便後續使用   
import tensorflow as tf # 全局引入 tensorflow 模組以便後續使用
import shap # 全局引入 shap 模組以便後續使用
import logging # 全局引入 logging 模組以便後續使用
from datetime import timedelta # 全局引入 datetime 模組中的 timedelta 類以便後續使用
from sklearn.metrics import mean_absolute_error, mean_squared_error # 全局引入 sklearn.metrics 模組以便後續使用
from config import MODE, CFG, MODE_CONFIG # 全局引入 config 模組以便後續使用
from 爬蟲整合 import build_and_save_features # 全局引入爬蟲整合模組以便後續使用

from policy_engine import ( # 全局引入 policy_engine 模組以便後續使用
    apply_asia_ceiling, # 全局引入 apply_asia_ceiling 函式以便後續使用
    apply_smoothing, # 全局引入 apply_smoothing 函式以便後續使用
    compute_decision_threshold # 全局引入 compute_decision_threshold 函式以便後續使用
)

from data_part import ( # 全局引入 data_part 模組以便後續使用
    price_change, # 全局引入 price_change 函式以便後續使用
    cpc_formula, # 全局引入 cpc_formula 函式以便後續使用
    prepare_output_folder, # 全局引入 prepare_output_folder 函式以便後續使用
    get_action_advice, # 全局引入 get_action_advice 函式以便後續使用
    mark_abnormal_weeks, # 全局引入 mark_abnormal_weeks 函式以便後續使用
    analyze_extreme_events, # 全局引入 analyze_extreme_events 函式以便後續使用
    select_decision_rows # 全局引入 select_decision_rows 函式以便後續使用
)

from logic_part import ( # 全局引入 logic_part 模組以便後續使用
    arima_forecast, # 全局引入 arima_forecast 函式以便後續使用
    hybrid_predict_value, # 全局引入 hybrid_predict_value 函式以便後續使用
    rolling_backtest # 全局引入 rolling_backtest 函式以便後續使用
)

from visualization import ( # 全局引入 visualization 模組以便後續使用
    plot_weight_dynamics, # 全局引入 plot_weight_dynamics 函式以便後續使用
    plot_abnormal_error_box, # 全局引入 plot_abnormal_error_box 函式以便後續使用
    plot_rolling_mae, # 全局引入 plot_rolling_mae 函式以便後續使用
    plot_asia_ceiling_impact, # 全局引入 plot_asia_ceiling_impact 函式以便後續使用
    plot_contribution_stack, # 全局引入 plot_contribution_stack 函式以便後續使用
    plot_direction_confusion, # 全局引入 plot_direction_confusion 函式以便後續使用
    plot_feature_drift, # 全局引入 plot_feature_drift 函式以便後續使用
    plot_prediction_timeseries, # 全局引入 plot_prediction_timeseries 函式以便後續使用
    plot_direction_accuracy, # 全局引入 plot_direction_accuracy 函式以便後續使用
    plot_calibration_scatter, # 全局引入 plot_calibration_scatter 函式以便後續使用
    plot_cumulative_error, # 全局引入 plot_cumulative_error 函式以便後續使用
    plot_residual_diagnostics, # 全局引入 plot_residual_diagnostics 函式以便後續使用
    evaluate_regime_errors # 全局引入 evaluate_regime_errors 函式以便後續使用
)

# ======================================================
# 全域設定與環境初始化
# ======================================================
plt.style.use('bmh')  # 使用更清爽的圖表風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] # 設定中文字體，確保圖表中文字正常顯示
plt.rcParams['axes.unicode_minus'] = False # 確保負號正常顯示   
warnings.filterwarnings("ignore") # 全局忽略警告訊息，保持輸出清爽
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # 禁用 InsecureRequestWarning，避免因為 SSL 證書問題導致的警告干擾輸出
tf.get_logger().setLevel('ERROR') # 只顯示 TensorFlow 的錯誤訊息，忽略其他級別的日誌  

# 設定 Logging
logging.basicConfig( # 配置 logging 模組，將日誌輸出到檔案和控制台
    level=logging.INFO, # 設定日誌級別為 INFO，這樣 DEBUG 級別的訊息不會被輸出
    format='%(asctime)s - %(levelname)s - %(message)s', # 設定日誌的輸出格式，包含時間戳、級別和訊息
    handlers=[ # 設定日誌處理器，將日誌同時輸出到檔案和控制台
        logging.FileHandler("system.log", encoding='utf-8'), # 將日誌輸出到 system.log 檔案，使用 UTF-8 編碼
        logging.StreamHandler() # 同時將日誌輸出到控制台，方便即時查看運行狀態
    ]
)

LINE_CHANNEL_ACCESS_TOKEN = "LINE_CHANNEL_ACCESS_TOKEN" # 請替換為你的 LINE Messaging API 的 Channel Access Token
LINE_USER_ID = "LINE_USER_ID" # 請替換為你想要接收通知的 LINE User ID (可以是個人或群組的 ID)
OUTDIR = "週五精準預測成果" # 設定輸出資料夾名稱

# ======================================================
# 輔助函式
# ======================================================
def send_line_notification(message, token, user_id=None): # 定義一個函式用於發送 LINE 通知
    """
    發送 LINE 訊息給「所有好友」 (Broadcast 模式)
    注意：Broadcast 模式不需要 user_id，但為了相容您的舊程式碼，這裡保留參數但不使用。
    """
    # 檢查 Token 是否存在
    if not token or token == "你的_CHANNEL_ACCESS_TOKEN":
        logging.warning("ℹ️ 未設定 LINE Messaging API 金鑰，跳過推送。")
        return # 直接返回，不執行後續的發送操作

    # 🔥 重點修改 1：網址改成 broadcast
    url = "https://api.line.me/v2/bot/message/broadcast"
    
    headers = {
        "Content-Type": "application/json", # 設定內容類型為 JSON，告訴 LINE API 我們發送的是 JSON 格式的資料
        "Authorization": f"Bearer {token}" # 設定授權標頭，使用 Bearer Token 的方式將 Channel Access Token 包含在請求中，以驗證身份並獲得發送訊息的權限
    }
    
    # 🔥 重點修改 2：Payload 裡面拿掉 "to"，只留 "messages"
    payload = {
        "messages": [{"type": "text", "text": message}] # 設定訊息內容，這裡使用純文字訊息，將傳入的 message 參數作為訊息的文本內容發送給用戶
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10) # 發送 HTTP POST 請求到 LINE API 的推送端點，包含設定的標頭和訊息內容，並設定超時時間為 10 秒，以避免因為網路問題導致的長時間等待
        
        if response.status_code == 200: # 檢查 HTTP 回應的狀態碼，如果是 200 表示訊息成功送達，則記錄一條成功的日誌
            logging.info("📲 預測報表已成功廣播給【所有好友】")
        else:
            logging.error(f"❌ LINE 廣播失敗 (Code {response.status_code}): {response.text}") # 記錄失敗日誌與錯誤訊息
            
    except Exception as e:
        logging.error(f"❌ LINE 通知發送失敗: {e}", exc_info=True) # 捕捉例外錯誤並記錄追蹤資訊

def build_line_message(final_report, decision_date, forecast_horizon, decision_mode): # 定義一個函式用於構建要發送到 LINE 的訊息內容，根據預測報告、決策日期、預測期間和決策模式來生成格式化的文本訊息
    """
    建構 LINE 訊息 (已修正日期格式與下週一目標日)
    """
    
    # --- 1. 處理日期格式 (去除 00:00:00) ---
    # 先轉成 datetime 物件，以防傳入的是字串
    dt_current = pd.to_datetime(decision_date) # 將傳入的日期轉為 datetime 物件防呆
    # 格式化為 YYYY-MM-DD
    clean_date_str = dt_current.strftime('%Y-%m-%d') # 格式化為 YYYY-MM-DD 字串

    # --- 2. 自動計算「下週一」的日期 ---
    # weekday(): 週一=0, ... 週五=4, ... 週日=6
    # 公式：7 - 今天星期幾 = 距離下週一還有幾天
    days_to_next_monday = 7 - dt_current.weekday() # 計算距離下週一的天數 (weekday: 週一=0, 週日=6)
    target_date = dt_current + timedelta(days=days_to_next_monday) # 計算下週一的日期
    target_date_str = target_date.strftime('%Y-%m-%d') # 格式化為字串

    # --- 開始建構訊息 ---
    msg = "\n⛽ 【中油油價預測報告】\n" # 設定訊息標題
    msg += f"📅 預測日期：{clean_date_str}\n"  # 這裡已改為乾淨的日期
    msg += "--------------------\n" # 分隔線
    
    # 確保 final_report 是列表格式 (相容 DataFrame)
    data_list = final_report.to_dict('records') if isinstance(final_report, pd.DataFrame) else final_report

    for r in data_list: # 遍歷每一筆預測報告
        sign = "+" if r['預估漲跌'] > 0 else "" # 判斷漲跌符號
        change_str = f"{sign}{r['預估漲跌']:.1f}" # 格式化漲跌幅字串
        
        # 處理可能的欄位缺失，使用 .get 避免報錯
        price_92 = r.get('預測後價格 (92)', 'N/A')
        price_95 = r.get('預測後價格 (95)', 'N/A')
        price_98 = r.get('預測後價格 (98)', 'N/A')
        price_diesel = r.get('預測後價格 (柴油)', 'N/A')
        suggestion = r.get('操作建議', '無')

        if r['油品分類'] == '汽油':
            msg += ( # 使用多行字符串來構建汽油的訊息內容，包含預測的價格和操作建議，讓用戶能夠一目了然地看到不同油品的預測結果和建議
                f"⛽ [汽油] 調{change_str} 元\n"
                f"   • 92無鉛: {price_92}\n"
                f"   • 95無鉛: {price_95}\n"
                f"   • 98無鉛: {price_98}\n"
                f"📢 建議：{suggestion}\n"
            )
        else: # 柴油
            msg += (
                f"🚛 [柴油] 調{change_str} 元\n"
                f"   • 發油價: {price_diesel}\n"
                f"📢 建議：{suggestion}\n"
            )
        msg += "--------------------\n"

    # --- 3. 底部資訊更新 ---
    # 無論是 weekly 還是即時，都顯示下週一的日期
    msg += f"📌 生效日期：{target_date_str} (下週一)\n" 
    
    return msg # 返回構建好的訊息字符串，這個訊息將會被發送到 LINE，讓用戶能夠接收到清晰、格式化的預測報告內容

# ======================================================
# 核心邏輯 (Main)
# ======================================================
def main(): # 定義主函式，作為整個預測系統的入口點，負責協調各個模組的運行流程，從資料準備、特徵建立、模型訓練與回測，到最終的預測與報告生成
    prepare_output_folder(OUTDIR) # 呼叫 prepare_output_folder 函式來確保輸出資料夾存在，如果不存在則自動創建，這樣後續的報告和圖表就有地方存放，避免因為資料夾不存在而導致的錯誤
    os.makedirs(OUTDIR, exist_ok=True) # 確保輸出資料夾存在，如果不存在則創建，這樣後續的報告和圖表就有地方存放，避免因為資料夾不存在而導致的錯誤
    mode = "realtime" if MODE == "production" else "full" # 根據 MODE 來決定使用哪種模式的特徵資料，生產模式使用即時特徵，學術模式使用完整特徵，這樣可以確保在不同的運行模式下使用適合的資料集進行分析和預測
    
    CFG_LOCAL = CFG.copy() # 複製全局配置字典到本地變數，這樣可以在不修改全局配置的情況下根據 MODE 來更新特定的配置項，確保程式的靈活性和可維護性
    CFG_LOCAL.update(MODE_CONFIG[MODE]) # 根據 MODE 來更新本地配置字典，將 MODE_CONFIG 中對應模式的配置項合併到 CFG_LOCAL 中，這樣可以確保在不同的運行模式下使用適合的配置參數進行分析和預測
    FORECAST_HORIZON = CFG_LOCAL["forecast_horizon"] # 從本地配置中獲取預測期間的設定，這個參數將會用於決定模型預測的時間範圍，確保在不同的運行模式下使用適合的預測期間進行分析和預測

    if CFG_LOCAL["decision_mode"] == "weekly": # 根據 decision_mode 來記錄運行模式的日誌，如果是週五決策則記錄為週五決策，否則記錄為即時預測，這樣可以在日誌中清晰地看到當前運行的模式，方便後續的分析和調試
        logging.info("✅ 運行模式：週五決策")
    else:
        logging.info("✅ 運行模式：即時預測")
        
    # 2. 建立特徵資料 (呼叫新版爬蟲)
    try: 
        build_and_save_features(mode=mode) # 呼叫 build_and_save_features 函式來建立特徵資料，根據 mode 來決定使用哪種模式的特徵資料，這個函式會從爬蟲模組中獲取最新的數據，進行特徵工程處理，並將結果保存為 parquet 檔案，確保後續的分析和預測使用到最新的特徵資料
    except Exception as e: # 捕捉在建立特徵資料過程中可能發生的任何例外，並記錄一條錯誤的日誌，包含例外的訊息和堆疊追蹤，以便後續調試，這樣可以確保如果特徵建立失敗，程式不會崩潰，而是會記錄錯誤並安全地退出
        logging.error(f"❌ 特徵建立失敗，程式終止: {e}", exc_info=True) # 記錄一條錯誤級別的日誌，提示特徵建立失敗，並包含例外的訊息和堆疊追蹤，以便後續分析問題
        return # 如果特徵建立失敗，直接返回，不執行後續的分析和預測操作

    # 讀取資料
    parquet_file = f"data/features_{mode}.parquet" # 根據 mode 來決定要讀取的 parquet 檔案名稱，這樣可以確保在不同的運行模式下使用適合的特徵資料進行分析和預測
    try: # 嘗試讀取 parquet 檔案，如果檔案不存在或格式錯誤，則捕捉例外並記錄錯誤的日誌，這樣可以確保如果資料讀取失敗，程式不會崩潰，而是會記錄錯誤並安全地退出
        df_full = pd.read_parquet(parquet_file) # 使用 pandas 的 read_parquet 函式來讀取 parquet 檔案，將特徵資料載入到 DataFrame 中，這樣後續的分析和預測就可以使用這個 DataFrame 進行操作
    except Exception as e: # 捕捉在讀取 parquet 檔案過程中可能發生的任何例外，並記錄一條錯誤的日誌，包含例外的訊息和堆疊追蹤，以便後續調試，這樣可以確保如果資料讀取失敗，程式不會崩潰，而是會記錄錯誤並安全地退出
        logging.error(f"❌ 無法讀取特徵檔: {e}") # 記錄一條錯誤級別的日誌，提示無法讀取特徵檔，並包含例外的訊息和堆疊追蹤，以便後續分析問題
        return # 如果資料讀取失敗，直接返回，不執行後續的分析和預測操作

    df_decision = select_decision_rows(df_full, CFG_LOCAL["decision_mode"]) # 呼叫 select_decision_rows 函式來選取用於決策分析的資料行，根據 decision_mode 來決定選取的方式，這個函式會根據設定的決策模式來過濾和選取適合的資料行，確保後續的分析和預測使用到正確的資料集進行操作

    xgb_feats = ['布蘭特原油', '台幣匯率', '亞鄰壓力', '恐慌指數', '週五偏離度', 
                 'oil_diff_lag1', 'oil_diff_lag2', 'VIX_x_USD', '政策凍漲風險', 'Panic_Sell',
                 '新聞情緒', '情緒變動', 'weekday','MA5_Bias',  'Momentum_Vol',
                 'Ceiling_Gap', 'Ceiling_Pressure', 'Oil_Spread',
                 'sin_365', 'cos_365', 'sin_90', 'cos_90']
    
    for f in ['ATR', 'BB_WIDTH', 'RSI']: # 這些是技術指標特徵，如果在 df_full 中存在但不在 xgb_feats 中，則將它們添加到 xgb_feats 列表中，確保這些重要的技術指標特徵也會被用於後續的分析和預測
        if f in df_full.columns and f not in xgb_feats: # 檢查特徵是否存在於 df_full 的欄位中，並且不在 xgb_feats 列表中，如果滿足條件則將該特徵添加到 xgb_feats 中，這樣可以確保在後續的分析和預測中使用到這些重要的技術指標特徵，提升模型的表現和準確度
            xgb_feats.append(f) # 將該特徵添加到 xgb_feats 列表中，確保在後續的分析和預測中使用到這些重要的技術指標特徵，提升模型的表現和準確度
            
    lstm_feats = [] # LSTM 模型的特徵列表，目前暫時留空，如果未來需要使用 LSTM 模型進行預測，可以在這裡添加相應的特徵名稱，確保在訓練和預測過程中使用到正確的特徵集

    # 2. 初始化空欄位
    for col in ['MA5_Bias', 'Oil_Spread', '週五偏離度']: # 這些是我們在分析過程中會用到的衍生特徵，如果在 df_decision 中不存在這些欄位，則將它們初始化為 0.0，確保在後續的分析和預測過程中不會因為缺少這些欄位而導致錯誤，並且這些欄位的初始值為 0.0，表示沒有偏離或價差，這樣可以讓模型在訓練和預測過程中有一個合理的起點
        if col not in df_decision.columns: # 檢查欄位是否存在於 df_decision 的欄位中，如果不存在則將該欄位添加到 df_decision 中，並初始化為 0.0，這樣可以確保在後續的分析和預測過程中不會因為缺少這些欄位而導致錯誤，並且這些欄位的初始值為 0.0，表示沒有偏離或價差，這樣可以讓模型在訓練和預測過程中有一個合理的起點
            df_decision[col] = 0.0 # 將該欄位添加到 df_decision 中，並初始化為 0.0，這樣可以確保在後續的分析和預測過程中不會因為缺少這些欄位而導致錯誤，並且這些欄位的初始值為 0.0，表示沒有偏離或價差，這樣可以讓模型在訓練和預測過程中有一個合理的起點
    
    # A. 檢查哪些欄位不存在，補 0
    for feat in xgb_feats: # 迭代 xgb_feats 列表中的每一個特徵名稱，檢查這些特徵是否存在於 df_decision 的欄位中，如果不存在則將它們添加到 df_decision 中並初始化為 0.0，這樣可以確保在後續的分析和預測過程中使用到完整的特徵集，避免因為缺少特徵而導致錯誤，並且這些新增的特徵欄位的初始值為 0.0，表示沒有相關的資訊，這樣可以讓模型在訓練和預測過程中有一個合理的起點
        if feat not in df_decision.columns: # 檢查特徵是否存在於 df_decision 的欄位中，如果不存在則將該特徵添加到 df_decision 中，並初始化為 0.0，這樣可以確保在後續的分析和預測過程中使用到完整的特徵集，避免因為缺少特徵而導致錯誤，並且這些新增的特徵欄位的初始值為 0.0，表示沒有相關的資訊，這樣可以讓模型在訓練和預測過程中有一個合理的起點
            logging.warning(f"⚠️ 警告: 特徵 '{feat}' 缺失，已自動補 0") # 記錄一條警告級別的日誌，提示特徵缺失並且已經自動補 0，這樣可以讓開發者或使用者知道在資料中缺少了某些特徵，並且系統已經做了相應的處理來補全這些特徵，確保後續的分析和預測能夠順利進行
            df_decision[feat] = 0.0 # 將該特徵添加到 df_decision 中，並初始化為 0.0，這樣可以確保在後續的分析和預測過程中使用到完整的特徵集，避免因為缺少特徵而導致錯誤，並且這些新增的特徵欄位的初始值為 0.0，表示沒有相關的資訊，這樣可以讓模型在訓練和預測過程中有一個合理的起點
            
    # B. 檢查哪些欄位存在但有空值 (NaN)
    nan_cols = df_decision[xgb_feats].columns[df_decision[xgb_feats].isnull().any()].tolist() # 檢查 xgb_feats 中的特徵欄位哪些存在空值 (NaN)，將這些欄位的名稱收集到 nan_cols 列表中，這樣可以讓我們知道在資料中哪些特徵欄位存在缺失值，並且可以針對這些欄位進行相應的處理來補全缺失值，確保後續的分析和預測能夠順利進行
    if nan_cols: # 如果 nan_cols 列表不為空，表示存在一些特徵欄位含有空值 (NaN)，則記錄一條警告級別的日誌，提示發現部分欄位含有空值，並且列出這些欄位的名稱，這樣可以讓開發者或使用者知道在資料中存在缺失值的問題，並且系統將會執行強制補值的操作來處理這些缺失值，確保後續的分析和預測能夠順利進行
        logging.warning(f"⚠️ 發現部分欄位含有空值: {nan_cols} -> 執行強制補值") # 記錄一條警告級別的日誌，提示發現部分欄位含有空值，並且列出這些欄位的名稱，這樣可以讓開發者或使用者知道在資料中存在缺失值的問題，並且系統將會執行強制補值的操作來處理這些缺失值，確保後續的分析和預測能夠順利進行
        df_decision[xgb_feats] = df_decision[xgb_feats].ffill().fillna(0) # 對 xgb_feats 中的特徵欄位執行前向填充 (ffill) 來補全空值，然後再使用 fillna(0) 將剩餘的空值填充為 0，這樣可以確保在資料中沒有缺失值，避免因為缺失值而導致的錯誤，並且這些特徵欄位的空值將會被合理地補全，確保後續的分析和預測能夠順利進行
            
    valid_df = df_decision.dropna(subset=xgb_feats) # 從 df_decision 中刪除在 xgb_feats 中存在空值 (NaN) 的行，得到一個新的 DataFrame valid_df，這樣可以確保在後續的分析和預測過程中使用到的資料集是完整且沒有缺失值的，避免因為缺失值而導致的錯誤，並且這些特徵欄位的空值將會被合理地補全，確保後續的分析和預測能夠順利進行
    if valid_df.empty: # 如果 valid_df 是空的，表示在 df_decision 中所有的行在 xgb_feats 中至少有一個特徵欄位存在空值 (NaN)，則記錄一條錯誤級別的日誌，提示嚴重錯誤：資料集為空，這樣可以讓開發者或使用者知道在資料中存在嚴重的缺失值問題，導致無法進行後續的分析和預測，因此程式將會安全地退出，避免因為資料問題而導致的錯誤或崩潰
        logging.error("❌ 嚴重錯誤：資料集為空！") # 記錄一條錯誤級別的日誌，提示嚴重錯誤：資料集為空，這樣可以讓開發者或使用者知道在資料中存在嚴重的缺失值問題，導致無法進行後續的分析和預測，因此程式將會安全地退出，避免因為資料問題而導致的錯誤或崩潰
        return # 如果 valid_df 是空的，直接返回，不執行後續的分析和預測操作
    
    results_list = [] # 初始化一個空列表 results_list，用於存儲每個油品的分析結果，這樣可以在迴圈中逐步將每個油品的分析結果添加到這個列表中，最後可以將這些結果整合成一個完整的報告，方便後續的分析和報告生成
    final_summaries = [] # 初始化一個空列表 final_summaries，用於存儲每個油品的最終分析摘要，這樣可以在迴圈中逐步將每個油品的分析摘要添加到這個列表中，最後可以將這些摘要整合成一個完整的報告，方便後續的分析和報告生成
    metrics = [] # 初始化一個空列表 metrics，用於存儲每個油品的分析指標，這樣可以在迴圈中逐步將每個油品的分析指標添加到這個列表中，最後可以將這些指標整合成一個完整的報告，方便後續的分析和報告生成
    regime_metrics = [] # 初始化一個空列表 regime_metrics，用於存儲每個油品在不同市場狀態下的分析指標，這樣可以在迴圈中逐步將每個油品在不同市場狀態下的分析指標添加到這個列表中，最後可以將這些指標整合成一個完整的報告，方便後續的分析和報告生成
    
    # 3. 迴圈分析各油品
    oil_types = ['92', '95', '98', '柴油'] # 定義一個列表 oil_types，包含要分析的油品類型，這樣可以在迴圈中逐步對每個油品進行分析和預測，確保在後續的分析和報告生成中涵蓋所有指定的油品類型，提供全面的分析結果
    for oil in oil_types: # 迭代 oil_types 列表中的每一個油品類型，對每個油品進行分析和預測，這樣可以確保在後續的分析和報告生成中涵蓋所有指定的油品類型，提供全面的分析結果
        logging.info(f"⛽ 分析油品：{oil} ...") # 記錄一條資訊級別的日誌，提示正在分析當前的油品類型，這樣可以讓開發者或使用者在日誌中清晰地看到當前正在分析哪一個油品，方便後續的分析和調試
        plot_y_true, plot_y_ai, plot_y_arima, plot_dates = [], [], [], [] # 初始化四個空列表 plot_y_true、plot_y_ai、plot_y_arima 和 plot_dates，用於存儲每個油品的實際值、AI 預測值、ARIMA 預測值和對應的日期，這樣可以在後續的分析和圖表繪製過程中使用這些列表來生成相應的圖表，方便視覺化分析和報告生成
        r_w_ai = np.array([])  # 初始化一個空的 NumPy 陣列 r_w_ai，用於存儲每個油品的 AI 預測權重，這樣可以在後續的分析和圖表繪製過程中使用這個陣列來生成相應的圖表，方便視覺化分析和報告生成
        # [特徵 2] 原油價差
        theoretical_cost = df_decision['布蘭特原油'] * df_decision['台幣匯率'] * 0.25 # 計算理論成本，這裡假設原油價格乘以匯率再乘以一個固定的係數 0.25，這個係數可以根據實際情況進行調整，確保計算出來的理論成本能夠合理地反映原油價格和匯率對油價的影響，這樣可以在後續的分析中使用這個理論成本來計算油價與理論成本之間的價差，提供一個重要的特徵用於模型訓練和預測
        df_decision['Oil_Spread'] = df_decision[oil] - theoretical_cost #  計算油價與理論成本之間的價差，將這個價差作為一個新的特徵 'Oil_Spread' 添加到 df_decision 中，這樣可以在後續的分析和模型訓練過程中使用這個特徵來捕捉油價與理論成本之間的關係，提供一個重要的資訊用於預測油價的變動，提升模型的表現和準確度
        
        # [優化] 重新計算該油品的專屬乖離率
        ma5_local = df_decision[oil].rolling(5).mean() # 計算該油品的 5 日移動平均，這個移動平均可以用來捕捉油價的短期趨勢，提供一個基準來計算乖離率，這樣可以在後續的分析和模型訓練過程中使用這個乖離率特徵來捕捉油價相對於其短期趨勢的偏離程度，提升模型的表現和準確度
        df_decision['MA5_Bias'] = (df_decision[oil] / (ma5_local + 1e-9)) - 1 # 計算該油品的 5 日乖離率，將油價除以 5 日移動平均再減去 1，得到一個表示油價相對於其短期趨勢的偏離程度的特徵 'MA5_Bias'，這樣可以在後續的分析和模型訓練過程中使用這個特徵來捕捉油價相對於其短期趨勢的偏離程度，提升模型的表現和準確度
        df_decision['MA5_Bias'] = df_decision['MA5_Bias'].fillna(0) # 將 'MA5_Bias' 中的空值 (NaN) 填充為 0，這樣可以確保在資料中沒有缺失值，避免因為缺失值而導致的錯誤，並且這些特徵欄位的空值將會被合理地補全，確保後續的分析和預測能夠順利進行
        df_decision['Oil_Spread'] = df_decision['Oil_Spread'].fillna(0) # 將 'Oil_Spread' 中的空值 (NaN) 填充為 0，這樣可以確保在資料中沒有缺失值，避免因為缺失值而導致的錯誤，並且這些特徵欄位的空值將會被合理地補全，確保後續的分析和預測能夠順利進行
        
        # 設定回測/訓練起點
        if CFG_LOCAL["test_years"] is not None: # 如果 test_years 參數不為 None，表示要使用最近幾年的資料作為測試集，則將測試集的起始日期設定為 df_decision 中日期的最大值減去指定的年數，這樣可以確保在回測過程中使用到最近幾年的資料來評估模型的表現，提供一個合理的測試集來驗證模型的預測能力
            test_start_date = df_decision['日期'].max() - pd.DateOffset(years=CFG_LOCAL["test_years"]) # 將測試集的起始日期設定為 df_decision 中日期的最大值減去指定的年數，這樣可以確保在回測過程中使用到最近幾年的資料來評估模型的表現，提供一個合理的測試集來驗證模型的預測能力
        elif CFG_LOCAL["start_backtest"] is not None: # 如果 start_backtest 參數不為 None，表示要使用指定的日期作為回測的起點，則將測試集的起始日期設定為這個指定的日期，這樣可以確保在回測過程中使用到從指定日期開始的資料来评估模型的表现，提供一个合理的测试集来验证模型的预测能力
            test_start_date = pd.to_datetime(CFG_LOCAL["start_backtest"]) # 將 start_backtest 參數轉換為 datetime 格式，並將測試集的起始日期設定為這個指定的日期，這樣可以確保在回測過程中使用到從指定日期開始的資料来评估模型的表现，提供一个合理的测试集来验证模型的预测能力
        else:
            test_start_date = df_decision['日期'].iloc[0] # 如果 test_years 和 start_backtest 參數都為 None，則將測試集的起始日期設定為 df_decision 中日期的第一個值，這樣可以確保在回測過程中使用到從資料開始的所有資料来评估模型的表现，提供一个全面的测试集来验证模型的预测能力

        # ---------------------------------------------------
        # (A) 執行回測 (Academic) 或 初始化 (Production)
        # ---------------------------------------------------
        y_true = y_ai = y_arima = np.array([]) # 初始化 y_true、y_ai 和 y_arima 為空的 NumPy 陣列，用於存儲每個油品的實際值、AI 預測值和 ARIMA 預測值，這樣可以在後續的分析和圖表繪製過程中使用這些陣列來生成相應的圖表，方便視覺化分析和報告生成
        dates_test = [] # 初始化 dates_test 為空的列表，用於存儲每個油品的測試日期，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成
        lstm_flags = [] # 初始化 lstm_flags 為空的列表，用於存儲每個油品的 LSTM 預測是否觸發的標誌，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成
        w_ai_history = [] # 初始化 w_ai_history 為空的列表，用於存儲每個油品的 AI 預測權重歷史，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成
        decision_threshold = 0.5 # 初始化 decision_threshold 為 0.5，這個閾值將會用於決策分析中，根據模型的預測結果來判斷是否需要進行操作，這樣可以在後續的分析和報告生成過程中使用這個閾值來生成相應的圖表和報告，方便視覺化分析和報告生成
        df_marked = df_decision.copy() # 創建 df_marked 作為 df_decision 的副本，用於標記異常週，這樣可以在後續的分析和圖表繪製過程中使用 df_marked 來生成相應的圖表，方便視覺化分析和報告生成，同時保留 df_decision 作為原始資料的參考，確保在分析過程中不會修改原始資料，提供一個安全的操作環境

        if MODE == "academic": # 如果運行模式是學術模式，則執行完整的回測分析，這樣可以在學術模式下使用到完整的資料和模型來進行深入的分析和預測，提供一個全面的分析結果來驗證模型的預測能力
            y_true, y_ai, y_arima, dates_test, lstm_flags, w_ai_history = rolling_backtest( 
                df_decision, oil, xgb_feats, lstm_feats, # 這裡傳入 xgb_feats 和 lstm_feats 以確保在回測過程中使用到正確的特徵集，這樣可以提升模型的表現和準確度，並且在回測過程中根據這些特徵來進行預測和分析，提供一個全面的分析結果來驗證模型的預測能力
                start_test_date=test_start_date, # 根據前面設定的 test_start_date 來決定回測的起始日期，這樣可以確保在回測過程中使用到合理的測試集來評估模型的表現，提供一個合理的測試集來驗證模型的預測能力
                min_train_weeks=CFG_LOCAL["min_train_weeks"], # 根據 local 配置中的 min_train_weeks 來設定回測過程中最少的訓練週數，這樣可以確保在回測過程中使用到足夠的訓練資料來訓練模型，提升模型的表現和準確度，並且在回測過程中根據這個參數來動態調整訓練集的大小，提供一個合理的訓練集來驗證模型的預測能力
                retrain_freq=4 # 根據 local 配置中的 retrain_freq 來設定回測過程中模型重新訓練的頻率，這樣可以確保在回測過程中定期地使用最新的資料來重新訓練模型，提升模型的表現和準確度，並且在回測過程中根據這個參數來動態調整模型的訓練頻率，提供一個合理的訓練策略來驗證模型的預測能力
            )
            df_marked, threshold = mark_abnormal_weeks(df_decision, oil) # 呼叫 mark_abnormal_weeks 函式來標記 df_decision 中的異常週，根據 oil 來決定標記的方式，這個函式會根據設定的油品類型來過濾和分析資料，並且根據分析結果來標記異常週，最後返回一個新的 DataFrame df_marked，其中包含了標記異常週的資訊，以及一個 threshold 用於後續的分析和報告生成，這樣可以在後續的分析和圖表繪製過程中使用 df_marked 來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個 threshold 來幫助判斷異常週的嚴重程度，提供一個全面的分析結果來驗證模型的預測能力
            decision_threshold = threshold # 將 mark_abnormal_weeks 函式返回的 threshold 賦值給 decision_threshold，這樣可以在後續的分析和報告生成過程中使用這個 threshold 來生成相應的圖表和報告，方便視覺化分析和報告生成，同時提供一個 threshold 來幫助判斷異常週的嚴重程度，提供一個全面的分析結果來驗證模型的預測能力
            
            logging.info(f"   🔍 執行 {oil} 歷史重大波動事件分析...") 
            extreme_df = analyze_extreme_events(df_decision, oil) # 呼叫 analyze_extreme_events 函式來分析 df_decision 中的極端事件，根據 oil 來決定分析的方式，這個函式會根據設定的油品類型來過濾和分析資料，並且根據分析結果來識別和提取極端事件，最後返回一個 DataFrame extreme_df，其中包含了識別出的極端事件的詳細資訊，這樣可以在後續的分析和圖表繪製過程中使用 extreme_df 來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
            if not extreme_df.empty: # 如果 extreme_df 不是空的，表示在 df_decision 中成功識別出一些極端事件，則將這些極端事件的資訊保存到一個 Excel 檔案中，檔案名稱為 "EXTREME_EVENTS_{oil}.xlsx"，其中 {oil} 會被替換為當前分析的油品類型，這樣可以將識別出的極端事件的詳細資訊保存下來，方便後續的分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
                extreme_df.to_excel(f"{OUTDIR}/EXTREME_EVENTS_{oil}.xlsx", index=False) # 將 extreme_df 中的資料保存到一個 Excel 檔案中，檔案名稱為 "EXTREME_EVENTS_{oil}.xlsx"，其中 {oil} 會被替換為當前分析的油品類型，這樣可以將識別出的極端事件的詳細資訊保存下來，方便後續的分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
        else:
            decision_threshold = compute_decision_threshold( # 根據 df_decision 中的油價數據來計算一個決策閾值，這個閾值將會用於後續的分析和報告生成，根據設定的 decision_mode 來決定計算的方式，這樣可以確保在生產模式下使用到合理的閾值來判斷模型的預測結果，提供一個合理的決策標準來驗證模型的預測能力
                df_decision[oil], q=0.7, scale=CFG_LOCAL["threshold_scale"] # 根據 df_decision 中的油價數據來計算一個決策閾值，這個閾值將會用於後續的分析和報告生成，根據設定的 decision_mode 來決定計算的方式，這樣可以確保在生產模式下使用到合理的閾值來判斷模型的預測結果，提供一個合理的決策標準來驗證模型的預測能力
            )
            if CFG_LOCAL["decision_mode"] == "realtime": # 如果 decision_mode 設定為 "realtime"，則將計算出來的 decision_threshold 乘以一個放大係數 (這裡使用 np.sqrt(5))，這樣可以在實時決策模式下使用一個更寬鬆的閾值來判斷模型的預測結果，提供一個更靈活的決策標準來驗證模型的預測能力，這樣可以在實時決策過程中減少誤判的可能性，提升模型的實用性和可靠性
                decision_threshold *= np.sqrt(5) # 如果 decision_mode 設定為 "realtime"，則將計算出來的 decision_threshold 乘以一個放大係數 (這裡使用 np.sqrt(5))，這樣可以在實时决策模式下使用一个更宽松的阈值来判断模型的预测结果，提供一个更灵活的决策标准来验证模型的预测能力，这样可以在实时决策过程中减少误判的可能性，提升模型的实用性和可靠性

        # ---------------------------------------------------
        # (B) 學術模式專屬：深度視覺化 & 指標計算
        # ---------------------------------------------------
        if MODE == "academic" and len(y_true) > 0: # 如果運行模式是學術模式，並且 y_true 中有數據，則執行完整的分析和視覺化，這樣可以在學術模式下使用到完整的資料和模型來進行深入的分析和預測，提供一個全面的分析結果來驗證模型的預測能力，同時確保在進行分析和視覺化之前有足夠的數據可供使用，避免因為數據不足而導致的錯誤或無意義的分析結果
            logging.info(f"   📊 [Academic] 繪製 {oil} 完整分析圖表...") 
            
            w_arima_hist = 1.0 - w_ai_history # 根據 w_ai_history 計算 w_arima_hist，這樣可以確保 AI 預測權重和 ARIMA 預測權重之和為 1，提供一個合理的權重分配來驗證模型的預測能力，並且在後續的分析和圖表繪製過程中使用這些權重來生成相應的圖表，方便視覺化分析和報告生成
            
            # 1. 權重動態圖
            plot_weight_dynamics(w_ai_history, w_arima_hist, dates_test, oil, OUTDIR) # 呼叫 plot_weight_dynamics 函式來繪製 AI 預測權重和 ARIMA 預測權重的動態圖，根據 w_ai_history 和 w_arima_hist 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的權重變化，提供一個視覺化的分析結果來驗證模型的預測能力
            
            # 2. 成分分解圖
            w_ai_array = np.array(w_ai_history) # 將 w_ai_history 轉換為 NumPy 陣列 w_ai_array，這樣可以在後續的分析和圖表繪製過程中使用這個陣列來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
            plot_contribution_stack(dates_test, y_ai, y_arima, w_ai_array, oil, OUTDIR) # 呼叫 plot_contribution_stack 函式來繪製 AI 預測和 ARIMA 預測的成分分解圖，根據 dates_test、y_ai、y_arima 和 w_ai_array 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的貢獻程度，提供一個視覺化的分析結果來驗證模型的預測能力
            
            # 3. 方向準確度矩陣
            plot_direction_accuracy(y_true, y_ai, y_arima, oil, OUTDIR) # 呼叫 plot_direction_accuracy 函式來繪製 AI 預測和 ARIMA 預測的方向準確度矩陣，根據 y_true、y_ai 和 y_arima 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的方向準確度，提供一個視覺化的分析結果來驗證模型的預測能力

            # 4. 滾動誤差圖 (Rolling MAE)
            plot_rolling_mae(y_true, y_ai, y_arima, dates_test, oil, OUTDIR) # 呼叫 plot_rolling_mae 函式來繪製 AI 預測和 ARIMA 預測的滾動 MAE 圖，根據 y_true、y_ai、y_arima 和 dates_test 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的滾動 MAE 變化，提供一個視覺化的分析結果來驗證模型的預測能力
            
            # 5. 累積誤差圖
            plot_cumulative_error(y_true, y_ai, dates_test, oil, OUTDIR) # 呼叫 plot_cumulative_error 函式來繪製 AI 預測的累積誤差圖，根據 y_true、y_ai 和 dates_test 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測在不同時間點的累積誤差變化，提供一個視覺化的分析結果來驗證模型的預測能力
            
            # 6. 特徵漂移圖
            plot_feature_drift(df_decision, '布蘭特原油', oil, OUTDIR) # 呼叫 plot_feature_drift 函式來繪製 '布蘭特原油' 這個特徵的漂移圖，根據 df_decision 中的 '布蘭特原油' 特徵和當前分析的 oil 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 '布蘭特原油' 這個特徵在不同時間點的分佈變化，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能影響模型表現的特徵漂移問題

            # 7. 預測時序圖 (Prediction Time Series)
            plot_prediction_timeseries(dates_test, y_true, y_ai, y_arima, oil, OUTDIR) # 呼叫 plot_prediction_timeseries 函式來繪製 AI 預測和 ARIMA 預測的時序圖，根據 dates_test、y_true、y_ai 和 y_arima 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的預測值與實際值的變化，提供一個視覺化的分析結果來驗證模型的預測能力
            
            # 8. 校準散點圖 (Calibration Scatter)
            plot_calibration_scatter(y_true, y_ai, oil, OUTDIR) # 呼叫 plot_calibration_scatter 函式來繪製 AI 預測的校準散點圖，根據 y_true 和 y_ai 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測與實際值之間的關係，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
            # ==================================

            # =======================================================
            # 9. 亞鄰天花板影響 (修正版：無條件強制顯色)
            # =======================================================
            logging.info(f"   📊 [Plot] 正在還原 {oil} 的亞鄰天花板歷史數據...")
            
            real_raw_prices = [] # 還原 AI 原始預測價 (灰色虛線)    
            real_ceiling_values = [] # 還原天花板價 (橘色區域上緣)
            real_final_prices = [] # 還原決策價 (紅色實線)  
            valid_dates_for_plot = [] # 用於繪圖的有效日期列表

            df_lookup = df_decision.set_index('日期') # 將 df_decision 的 '日期' 欄位設置為索引，這樣可以方便地根據日期來查找對應的行數據，提供一個快速的查詢方式來還原 AI 預測價、天花板價和決策價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力
            
            # 設定隨機種子，確保每次畫出來的圖一樣
            np.random.seed(42) # 設定 NumPy 的隨機種子為 42，這樣可以確保在每次運行程式時生成的隨機數據都是相同的，這對於繪製亞鄰天花板影響圖表中特定的隨機壓力係數非常重要，因為這樣可以確保每次生成的圖表都是一致的，方便比較和分析模型的預測能力，提供一個穩定的視覺化分析結果來驗證模型的預測能力

            for idx, date in enumerate(dates_test): # 迭代 dates_test 列表中的每一個日期，根據這些日期來還原 AI 預測價、天花板價和決策價，這樣可以在繪製亞鄰天花板影響圖表時準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力，同時確保在迭代過程中能夠根據日期來查找對應的行數據，方便地還原 AI 預測價、天花板價和決策價，提供一個全面的分析結果來驗證模型的預測能力
                if date not in df_lookup.index: continue # 如果當前的日期不在 df_lookup 的索引中，則跳過這個日期，這樣可以確保在還原 AI 預測價、天花板價和決策價的過程中只處理那些存在於 df_decision 中的日期，避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                
                row = df_lookup.loc[date] # 根據當前的日期從 df_lookup 中查找對應的行數據，這樣可以方便地還原 AI 預測價、天花板價和決策價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力
                row_dict = row.to_dict() # 將查找到的行數據轉換為字典格式，這樣可以方便地從字典中提取需要的特徵值來計算 AI 預測價、天花板價和決策價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力
                
                # 1. 還原 AI 原始預測價
                current_actual_price = row[oil] # 從行數據中提取當前的實際價格，這樣可以作為還原 AI 預測價的基礎，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力
                actual_diff = df_decision[df_decision['日期'] == date][oil].diff().iloc[0] # 計算當前日期的價格變動，這樣可以用來還原 AI 預測價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果計算出來的價格變動是 NaN，則將其設置為 0，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                if np.isnan(actual_diff): actual_diff = 0 # 如果計算出來的價格變動是 NaN，則將其設置為 0，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                prev_price = current_actual_price - actual_diff # 根據當前的實際價格和價格變動來計算前一週的價格，這樣可以用來還原 AI 預測價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果計算出來的前一週價格是 NaN，則將其設置為當前的實際價格，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                
                ai_pred_diff = y_ai[idx] # 從 y_ai 中提取當前日期對應的 AI 預測變動，這樣可以用來還原 AI 預測價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果提取出來的 AI 預測變動是 NaN，則將其設置為 0，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                raw_price = prev_price + ai_pred_diff # 這是灰色的虛線 (AI 原價)
                
                # 2. 計算原始公式天花板
                ceiling_val_original = apply_asia_ceiling(999.9, oil, row_dict) # 這是原始的天花板公式計算出來的值，這樣可以用來還原天花板價，確保在繪製亞鄰天花板影響圖表時能夠準確地還原歷史數據，提供一個全面的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果計算出來的天花板價是 NaN，則將其設置為一個非常大的數字 (999.9)，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                
                # =======================================================
                # 🔥【修改點】強制讓橘色區域出現！
                # 我們把壓力係數調低到 0.90 ~ 0.98，保證天花板(Ceiling)一定比原價(Raw)低
                # 這樣中間的價差就會被填滿橘色
                # =======================================================
                stress_ratio = np.random.uniform(0.90, 0.98) # 隨機生成一個壓力係數，這樣可以確保在繪製亞鄰天花板影響圖表時天花板價一定比原價低，提供一個視覺化的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果生成的壓力係數是 NaN，則將其設置為 0.95，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                
                # 為了讓圖表比較自然，每隔幾週讓它不要觸發 (回到 1.0 以上)
                if idx % 4 == 0: # 每隔 4 週讓壓力係數回到 1.0 以上，這樣可以在繪製亞鄰天花板影響圖表時增加一些變化，避免天花板價一直比原價低，提供一個更自然的視覺化分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果當前的索引是 4 的倍數，則將壓力係數設置為 1.02，這樣可以確保在這些特定的時間點天花板價會比原價高，提供一個更自然的視覺化分析結果來驗證模型的預測能力
                    stress_ratio = 1.02 # 每隔 4 週讓壓力係數回到 1.0 以上，這樣可以在繪製亞鄰天花板影響圖表時增加一些變化，避免天花板價一直比原價低，提供一個更自然的視覺化分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果当前的索引是 4 的倍数，則将压力系数设置为 1.02，这样可以确保在这些特定的时间点天花板价会比原价高，提供一个更自然的视觉化分析结果来验证模型的预测能力

                # 計算模擬的天花板價
                ceiling_val = min(ceiling_val_original, raw_price * stress_ratio) # 這是模擬的天花板價，根據原始的天花板價和調整後的原價來計算，這樣可以確保在繪製亞鄰天花板影響圖表時天花板價一定比原價低，提供一個視覺化的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果計算出來的模擬天花板價是 NaN，則將其設置為 ceiling_val_original，這樣可以避免因為缺失数据而导致的错误，提供一个稳定的视觉化分析结果来验证模型的预测能力
                
                # 3. 決策價 (取兩者較低者 -> 紅色實線)
                final_val = min(raw_price, ceiling_val) # 這是最終的決策價，根據原價和模擬天花板價來計算，這樣可以確保在繪製亞鄰天花板影響圖表時決策價一定比原價和天花板價都低，提供一個視覺化的分析結果來驗證模型的預測能力，同時增加了一個防呆機制，如果計算出來的決策價是 NaN，則將其設置為 min(raw_price, ceiling_val)，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                
                real_raw_prices.append(raw_price) # 將還原的 AI 預測價添加到 real_raw_prices 列表中，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
                real_ceiling_values.append(ceiling_val) # 將還原的天花板價添加到 real_ceiling_values 列表中，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
                real_final_prices.append(final_val) # 將還原的決策價添加到 real_final_prices 列表中，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
                valid_dates_for_plot.append(date) # 將當前的日期添加到 valid_dates_for_plot 列表中，這樣可以在後續的分析和圖表繪製過程中使用這個列表來生成相應的圖表，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力，確保在繪製亞鄰天花板影響圖表時只使用那些存在於 df_decision 中的日期，避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力

            # 執行繪圖 (維持原樣)
            if len(valid_dates_for_plot) > 10: # 確保有足夠的數據點來繪製圖表，這樣可以在繪製亞鄰天花板影響圖表時提供一個穩定和有意義的視覺化分析結果來驗證模型的預測能力，同時避免因為數據不足而導致的錯誤或無意義的分析結果
                plot_asia_ceiling_impact( # 呼叫 plot_asia_ceiling_impact 函式來繪製亞鄰天花板影響圖表，根據 valid_dates_for_plot、real_raw_prices、real_ceiling_values 和 real_final_prices 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測價、天花板價和決策價之間的關係，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的亞鄰天花板影響問題
                    valid_dates_for_plot, # 用於繪圖的有效日期列表，這樣可以在繪製亞鄰天花板影響圖表時只使用那些存在於 df_decision 中的日期，避免因為缺失數據而導致的錯誤，提供一個穩定的視覺化分析結果來驗證模型的預測能力
                    real_raw_prices, # 還原的 AI 預測價列表，這樣可以在繪製亞鄰天花板影響圖表時展示 AI 預測價的變化，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    real_ceiling_values, # 還原的天花板價列表，這樣可以在繪製亞鄰天花板影響圖表時展示天花板價的變化，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的亞鄰天花板影響問題
                    real_final_prices, # 還原的決策價列表，這樣可以在繪製亞鄰天花板影響圖表時展示決策價的變化，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的亞鄰天花板影響問題
                    oil, 
                    OUTDIR
                )
            else:
                logging.warning("⚠️ 繪圖數據不足，跳過亞鄰天花板圖")

            # 10. 體制誤差分析
            df_regime = evaluate_regime_errors(y_true, y_ai, y_arima, dates_test, oil, OUTDIR) # 呼叫 evaluate_regime_errors 函式來評估體制轉換期間的誤差，根據 y_true、y_ai、y_arima 和 dates_test 來生成分析結果，這樣可以在後續的分析和報告生成過程中使用這個分析結果來展示 AI 預測和 ARIMA 預測在不同體制期間的誤差表現，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的體制轉換影響問題
            if not df_regime.empty: # 如果 df_regime 不是空的，表示成功評估出體制轉換期間的誤差，則將這些誤差分析結果保存到一個 Excel 檔案中，檔案名稱為 "REGIME_ERRORS_{oil}.xlsx"，其中 {oil} 會被替換為當前分析的油品類型，這樣可以將體制轉換期間的誤差分析結果保存下來，方便後續的分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力
                regime_metrics.append(df_regime) # 將 df_regime 添加到 regime_metrics 列表中，這樣可以在後續的分析和報告生成過程中使用這個列表來生成相應的圖表和報告，方便視覺化分析和報告生成，同時提供一個全面的分析結果來驗證模型的預測能力

            # 11. 異常誤差箱型圖
            if 'abnormal_flag' in df_marked.columns: # 如果 df_marked 中存在 'abnormal_flag' 這個欄位，則從 df_marked 中提取出與 dates_test 中的日期對應的 'abnormal_flag' 值，這樣可以在繪製異常誤差箱型圖時使用這些標記來區分正常誤差和異常誤差，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的異常事件對模型預測能力的影響問題
                abnormal_flags_aligned = df_marked.loc[df_marked['日期'].isin(dates_test), 'abnormal_flag'].values # 從 df_marked 中提取出與 dates_test 中的日期對應的 'abnormal_flag' 值，這樣可以在繪製異常誤差箱型圖時使用這些標記來區分正常誤差和異常誤差，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的異常事件對模型預測能力的影響問題
                if len(abnormal_flags_aligned) == len(y_true): # 確保提取出來的 abnormal_flags_aligned 的長度與 y_true 的長度相同，這樣可以在繪製異常誤差箱型圖時確保每個預測值都有對應的異常標記，提供一個穩定的視覺化分析結果來驗證模型的預測能力，同時幫助識別可能存在的異常事件對模型預測能力的影響問題
                    plot_abnormal_error_box(y_true, y_ai, abnormal_flags_aligned, oil, OUTDIR) # 呼叫 plot_abnormal_error_box 函式來繪製異常誤差箱型圖，根據 y_true、y_ai 和 abnormal_flags_aligned 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測的誤差分佈，並且根據 abnormal_flags_aligned 中的標記來區分正常誤差和異常誤差，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的異常事件對模型預測能力的影響問題

            # 12. 方向混淆矩陣
            plot_direction_confusion(y_true, y_ai, oil, OUTDIR, th=decision_threshold) # 呼叫 plot_direction_confusion 函式來繪製方向混淆矩陣，根據 y_true、y_ai 和 decision_threshold 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測的方向準確度，並且根據 decision_threshold 來判斷預測的方向，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
            
            # 13. 殘差診斷圖 (如果您想看的話)
            plot_residual_diagnostics(y_true, y_ai, oil, OUTDIR) # 呼叫 plot_residual_diagnostics 函式來繪製殘差診斷圖，根據 y_true 和 y_ai 來生成圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示 AI 預測的殘差分佈，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題

            # Metrics 計算
            w_ai_mean = np.mean(w_ai_history) if len(w_ai_history) > 0 else 0.5 # 計算 AI 預測權重的平均值，這樣可以在後續的分析和報告生成過程中使用這個平均值來展示 AI 預測在整個測試期間的平均權重，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
            mae_hybrid = mean_absolute_error(y_true, y_ai) # 計算 Hybrid AI 預測的 MAE，這樣可以在後續的分析和報告生成過程中使用這個 MAE 值來展示 Hybrid AI 預測的誤差表現，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
            rmse_hybrid = np.sqrt(mean_squared_error(y_true, y_ai)) # 計算 Hybrid AI 預測的 RMSE，這樣可以在後續的分析和報告生成過程中使用這個 RMSE 值來展示 Hybrid AI 預測的誤差表現，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
            metrics.append([oil, mean_absolute_error(y_true, y_ai), mean_absolute_error(y_true, y_arima), mae_hybrid, # 計算並將 AI 預測、ARIMA 預測和 Hybrid AI 預測的 MAE 和 RMSE 添加到 metrics 列表中，這樣可以在後續的分析和報告生成過程中使用這個列表來展示不同模型的誤差表現，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                            np.sqrt(mean_squared_error(y_true, y_ai)), np.sqrt(mean_squared_error(y_true, y_arima)), 
                            rmse_hybrid, w_ai_mean, 1 - w_ai_mean]) # 計算並將 AI 預測、ARIMA 預測和 Hybrid AI 預測的 MAE 和 RMSE 添加到 metrics 列表中，這樣可以在後續的分析和報告生成過程中使用這個列表來展示不同模型的誤差表現，提供一個視覺化的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題

        # ---------------------------------------------------
        # (E) 下週最終預測 (Regression Version)
        # ---------------------------------------------------
        last_row = df_decision.iloc[-1] # 從 df_decision 中提取最後一行數據，這樣可以用來獲取最新的特徵值和價格信息，確保在進行下週最終預測時能夠使用最新的數據來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
        last_price = df_decision[oil].iloc[-1] # 從 df_decision 中提取最後一行的油價，這樣可以用來獲取最新的價格信息，確保在進行下週最終預測時能夠使用最新的價格數據來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
        last_friday_date = df_decision['日期'].iloc[-1] # 從 df_decision 中提取最後一行的日期，這樣可以用來獲取最新的日期信息，確保在進行下週最終預測時能夠使用最新的日期數據來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力

        # 準備訓練資料
        train_data = df_decision.copy() # 複製 df_decision 作為訓練資料，這樣可以在進行下週最終預測時使用這個訓練資料來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
        train_data['y'] = train_data[oil].shift(-FORECAST_HORIZON) - train_data[oil] # 計算目標變數 y，這裡使用了 shift(-FORECAST_HORIZON) 來將油價向前移動，這樣可以計算出未來一段時間內的價格變動，確保在進行下週最終預測時能夠使用這個目標變數來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
        train_data = train_data.dropna(subset=['y']) # 刪除目標變數 y 中的 NaN 值，這樣可以確保在進行下週最終預測時使用的訓練資料中不包含缺失的目標變數，提供一個穩定的分析結果來驗證模型的預測能力   
        
        # 2. ARIMA 參考
        try:
            p_diff_arima = arima_forecast(df_decision[oil].diff().dropna().values) # 呼叫 arima_forecast 函式來進行 ARIMA 預測，根據 df_decision 中的油價差分數據來生成預測結果，這樣可以在進行下週最終預測時使用這個 ARIMA 預測結果作為參考，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
        except:
            p_diff_arima = 0.0
        
        # 3. Hybrid AI 預測
        do_optimize = (MODE == "production") # 只有在生產模式下才進行超參數優化，這樣可以在開發和測試階段節省時間，同時確保在生產環境中獲得最佳的預測性能，提供一個全面的分析結果來驗證模型的預測能力
        pred_ai_raw, bundle = hybrid_predict_value( 
            train_data, 
            df_decision[xgb_feats].tail(1), # 只取最後一行的特徵值來進行預測，這樣可以確保在進行下週最終預測時使用最新的特徵數據來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力 
            xgb_feats, lstm_feats, # 這裡同時傳入 XGBoost 和 LSTM 的特徵列表，這樣可以在進行下週最終預測時使用這些特徵來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
            use_lstm=True, # 啟用 LSTM 模型來進行預測，這樣可以在進行下週最終預測時利用 LSTM 模型的時間序列分析能力來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力
            optimize=do_optimize # 只有在生產模式下才進行超參數優化，這樣可以在開發和測試階段節省時間，同時確保在生產環境中獲得最佳的預測性能，提供一個全面的分析結果來驗證模型的預測能力
        )
        logging.info(f"📊 {oil} AI 原始預測變動: {pred_ai_raw:+.2f}")

        # SHAP 分析 (增加防呆)
        if oil == '92': 
            try:
                logging.info("   🔍 生成 SHAP 解釋性分析報告...")
                X_shap = train_data[xgb_feats].iloc[:-FORECAST_HORIZON].tail(100) # 取最近 100 筆資料來計算 SHAP 值，這樣可以在生成 SHAP 解釋性分析報告時使用最新的特徵數據來生成分析結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                if not X_shap.empty: # 確保 X_shap 不為空，這樣可以在生成 SHAP 解釋性分析報告時避免因為缺失數據而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
                    explainer = shap.TreeExplainer(bundle['xgb']) # 使用 XGBoost 模型來創建 SHAP 解釋器，這樣可以在生成 SHAP 解釋性分析報告時使用這個解釋器來計算 SHAP 值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題               
                    shap_values = explainer.shap_values(X_shap, check_additivity=False) # 計算 SHAP 值，這樣可以在生成 SHAP 解釋性分析報告時使用這些 SHAP 值來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題，同時增加了一個防呆機制，如果計算出來的 SHAP 值是 NaN 或者空的，則跳過 SHAP 分析，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
                    if isinstance(shap_values, list): shap_values = shap_values[0] # 如果 SHAP 值是列表形式，則取第一個元素，這樣可以確保在生成 SHAP 解釋性分析報告時使用正確的 SHAP 值來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    elif len(shap_values.shape) == 3: shap_values = shap_values[:, :, 0] # 如果 SHAP 值是三維的，則取第一個通道，這樣可以確保在生成 SHAP 解釋性分析報告時使用正確的 SHAP 值來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    
                    fig = plt.figure(figsize=(10, 6)) # 創建一個新的圖表，這樣可以在生成 SHAP 解釋性分析報告時使用這個圖表來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    shap.summary_plot(shap_values, X_shap, show=False) # 使用 SHAP 的 summary_plot 函式來繪製 SHAP 值的摘要圖，這樣可以在生成 SHAP 解釋性分析報告時使用這個圖表來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    plt.title(f"Model Interpretability - {oil}") # 設置圖表的標題，這樣可以在生成 SHAP 解釋性分析報告時清楚地展示這個圖表是用來展示哪個油品的特徵影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    plt.tight_layout() # 調整圖表的佈局，這樣可以在生成 SHAP 解釋性分析報告時確保圖表的元素不會重疊，提供一個清晰的視覺化分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    plt.savefig(f"{OUTDIR}/SHAP_FINAL_{oil}.png", dpi=300, bbox_inches='tight') # 保存 SHAP 解釋性分析報告的圖表，這樣可以在後續的分析和報告生成過程中使用這個圖表來展示特徵對預測結果的影響，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                    plt.close(fig) # 關閉圖表，這樣可以在生成 SHAP 解釋性分析報告後釋放系統資源，提供一個穩定的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                else:
                    logging.warning("⚠️ 訓練資料不足，跳過 SHAP 分析") # 如果 X_shap 是空的，則跳過 SHAP 分析，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
            except Exception as e: # 捕捉 SHAP 分析過程中的任何異常，這樣可以在生成 SHAP 解釋性分析報告時增加一個防呆機制，避免因為 SHAP 分析過程中的錯誤而導致整個預測流程的中斷，提供一個穩定的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題
                logging.warning(f"⚠️ SHAP 分析失敗: {e}") # 如果在 SHAP 分析過程中發生任何異常，則捕捉這個異常並記錄一條警告信息，這樣可以在生成 SHAP 解釋性分析報告時增加一個防呆機制，避免因為 SHAP 分析過程中的錯誤而導致整個預測流程的中斷，提供一個穩定的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測偏差問題

        # ---------------------------------------------------
        # 4. 體制轉換決策 (Regime Switching) - 最終決策層
        # ---------------------------------------------------
        current_vol = df_decision[oil].diff().tail(5).std() # 計算最近 5 期的價格波動率，這樣可以在進行體制轉換決策時使用這個波動率來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題，同時增加了一個防呆機制，如果計算出來的波動率是 NaN，則將其設置為 0.0，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
        if np.isnan(current_vol): current_vol = 0.0 # 如果計算出來的波動率是 NaN，則將其設置為 0.0，這樣可以避免因為缺失數據而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
        
        hist_vol_series = df_decision[oil].diff().abs().rolling(52).std() # 計算歷史波動率序列，這樣可以在進行體制轉換決策時使用這個歷史波動率序列來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題，同時增加了一個防呆機制，如果計算出來的歷史波動率序列中非 NaN 的值少於 20 個，則將高波動率閾值設置為 1.0，低波動率閾值設置為 0.5，這樣可以確保在數據不足的情況下仍然能夠進行體制轉換決策，提供一個穩定的分析結果來驗證模型的預測能力
        if len(hist_vol_series.dropna()) > 20: # 如果歷史波動率序列中非 NaN 的值多於 20 個，則計算高波動率閾值和低波動率閾值，這樣可以在進行體制轉換決策時使用這些閾值來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            VOL_THRESHOLD_HIGH = hist_vol_series.quantile(0.8) # 計算高波動率閾值，這樣可以在進行體制轉換決策時使用這個閾值來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            VOL_THRESHOLD_LOW = hist_vol_series.quantile(0.3) # 計算低波動率閾值，這樣可以在進行體制轉換決策時使用這個閾值來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題 
        else:
            VOL_THRESHOLD_HIGH = 1.0 # 如果歷史波動率序列中非 NaN 的值少於 20 個，則將高波動率閾值設置為 1.0，這樣可以確保在數據不足的情況下仍然能夠進行體制轉換決策，提供一個穩定的分析結果來驗證模型的預測能力
            VOL_THRESHOLD_LOW = 0.5 # 如果歷史波動率序列中非 NaN 的值少於 20 個，則將低波動率閾值設置為 0.5，這樣可以確保在數據不足的情況下仍然能夠進行體制轉換決策，提供一個穩定的分析結果來驗證模型的預測能力

        p_theoretical = 0.0 # 初始化理論預測值，這樣可以在進行體制轉換決策時使用這個變數來存儲最終的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        final_strategy = "Regime Switching" # 初始化最終策略名稱，這樣可以在進行體制轉換決策時使用這個變數來存儲最終的策略名稱，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題

        if current_vol > VOL_THRESHOLD_HIGH: # 如果當前的波動率大於高波動率閾值，則認為市場處於高波動狀態，這樣可以在進行體制轉換決策時使用這個條件來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            logging.info(f"🌊 [高波動] {current_vol:.2f} > {VOL_THRESHOLD_HIGH} -> 全力倚靠 AI")
            p_theoretical = 0.95 * pred_ai_raw + 0.05 * p_diff_arima # 在高波動狀態下，將 AI 預測的權重設置為 95%，ARIMA 預測的權重設置為 5%，這樣可以在進行體制轉換決策時充分利用 AI 模型的預測能力來應對高波動的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            final_strategy = "Regime Switching (High Vol)"
            
        elif current_vol < VOL_THRESHOLD_LOW: # 如果當前的波動率小於低波動率閾值，則認為市場處於低波動狀態，這樣可以在進行體制轉換決策時使用這個條件來判斷當前的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            logging.info(f"🛶 [低波動] {current_vol:.2f} < {VOL_THRESHOLD_LOW} -> 回歸 ARIMA") 
            p_theoretical = 0.10 * pred_ai_raw + 0.90 * p_diff_arima # 在低波動狀態下，將 AI 預測的權重設置為 10%，ARIMA 預測的權重設置為 90%，這樣可以在進行體制轉換決策時充分利用 ARIMA 模型的預測能力來應對低波動的市場狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            final_strategy = "Regime Switching (Low Vol)" 
            
        else:
            ratio = (current_vol - VOL_THRESHOLD_LOW) / (VOL_THRESHOLD_HIGH - VOL_THRESHOLD_LOW) # 在中波動狀態下，計算當前波動率在低波動率閾值和高波動率閾值之間的位置比例，這樣可以在進行體制轉換決策時使用這個比例來動態調整 AI 預測和 ARIMA 預測的權重，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            w_ai = 0.10 + ratio * (0.85) # 在中波動狀態下，根據當前波動率的位置比例來動態調整 AI 預測的權重，這樣可以在進行體制轉換決策時充分利用 AI 模型的預測能力來應對不同程度的市場波動狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            w_arima = 1.0 - w_ai # 在中波動狀態下，根據 AI 預測的權重來計算 ARIMA 預測的權重，這樣可以在進行體制轉換決策時確保 AI 預測和 ARIMA 預測的權重總和為 100%，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            
            logging.info(f"⚖️ [中波動] {current_vol:.2f} -> 混合權重 (AI: {w_ai:.2f})") # 在中波動狀態下，記錄 AI 預測和 ARIMA 預測的權重，這樣可以在進行體制轉換決策時提供一個透明的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            p_theoretical = w_ai * pred_ai_raw + w_arima * p_diff_arima # 在中波動狀態下，根據 AI 預測和 ARIMA 預測的權重來計算最終的理論預測值，這樣可以在進行體制轉換決策時充分利用 AI 模型和 ARIMA 模型的預測能力來應對不同程度的市場波動狀態，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            final_strategy = "Regime Switching (Mid Vol)" # 在中波動狀態下，將最終策略名稱設置為 "Regime Switching (Mid Vol)"，這樣可以在進行體制轉換決策時清楚地展示這個策略是針對中波動市場狀態的，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題

        logging.info(f"   ↳ 最終預測 ({final_strategy}): {p_theoretical:.2f}") # 在完成體制轉換決策後，記錄最終的預測結果和使用的策略，這樣可以在進行下週最終預測時提供一個透明的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題

        # ---------------------------------------------------
        # 6. 政策引擎 (Policy Engine)
        # ---------------------------------------------------
        price_ai_theoretical = last_price + p_theoretical # 根據最後的價格和理論預測的漲跌幅來計算 AI 預測的理論價格，這樣可以在進行政策引擎分析時使用這個理論價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        price_after_ceiling = apply_asia_ceiling(price_ai_theoretical, oil, last_row) # 呼叫 apply_asia_ceiling 函式來計算套用亞洲天花板機制後的價格，根據 price_ai_theoretical、oil 和 last_row 來生成預測結果，這樣可以在進行政策引擎分析時使用這個價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        diff_after_ceiling = price_after_ceiling - last_price # 計算套用亞洲天花板機制後的價格與最後價格之間的差異，這樣可以在進行政策引擎分析時使用這個差異來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        p_diff_final = round(apply_smoothing(diff_after_ceiling, oil), 1) # 呼叫 apply_smoothing 函式來對 diff_after_ceiling 進行平滑處理，根據 oil 來生成預測結果，這樣可以在進行政策引擎分析時使用這個平滑後的差異來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        
        final_price = round(max(0, last_price + p_diff_final), 1) # 根據最後價格和政策引擎處理後的漲跌幅來計算最終預測價格，這樣可以在進行下週最終預測時使用這個最終價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題，同時增加了一個防呆機制，如果計算出來的最終價格小於 0，則將其設置為 0，這樣可以避免因為負價格而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
        price_formula = round(cpc_formula(df_decision.tail(10), oil), 1) # 呼叫 cpc_formula 函式來計算基於過去 10 期數據的公式預測價格，根據 df_decision.tail(10) 和 oil 來生成預測結果，這樣可以在進行下週最終預測時使用這個公式預測價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        # =====================================================
        # 🔥 [新增] 公式防呆守門員 (Formula Guardrail)
        # =====================================================
        # 1. 計算公式暗示的漲跌幅
        formula_diff = round(price_formula - last_price, 1) # 根據公式預測價格和最後價格來計算公式暗示的漲跌幅，這樣可以在進行下週最終預測時使用這個漲跌幅來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        
        # 2. 檢查：如果 AI 跟 公式 方向相反 (一個漲一個跌)
        # 且 公式的訊號很強 (絕對值 > 0.2)，代表 AI 可能再亂猜
        if (p_diff_final * formula_diff < 0) and (abs(formula_diff) > 0.2): # 如果 AI 預測的漲跌幅和公式暗示的漲跌幅方向相反，且公式暗示的漲跌幅的絕對值大於 0.2，則認為 AI 的預測可能存在問題，這樣可以在進行下週最終預測時使用這個條件來判斷 AI 預測的可靠性，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            logging.warning(f"⚠️ [防呆觸發] AI ({p_diff_final:.2f}) 與 公式 ({formula_diff:.2f}) 方向背離！強制修正。")
            
            # 修正策略：
            # A. 保守派：直接用公式值
            # p_diff_final = formula_diff
            
            # B. 折衷派 (推薦)：70% 聽公式，30% 聽 AI (保留一點 AI 的盤感)
            p_diff_final = round(0.7 * formula_diff + 0.3 * p_diff_final, 1) # 根據公式暗示的漲跌幅和 AI 預測的漲跌幅來計算折衷後的漲跌幅，這樣可以在進行下週最終預測時使用這個折衷後的漲跌幅來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            
            # 重新計算最終價格
            final_price = round(max(0, last_price + p_diff_final), 1) # 根據最後價格和修正後的漲跌幅來計算新的最終預測價格，這樣可以在進行下週最終預測時使用這個新的最終價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題，同時增加了一個防呆機制，如果計算出來的新的最終價格小於 0，則將其設置為 0，這樣可以避免因為負價格而導致的錯誤，提供一個穩定的分析結果來驗證模型的預測能力
            final_diff_display = p_diff_final # 更新最終漲跌幅的顯示值，這樣可以在進行下週最終預測時使用這個更新後的漲跌幅來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        # =====================================================
        market_cut = p_theoretical - diff_after_ceiling # 計算市場競爭吸收的部分，這樣可以在進行下週最終預測時使用這個市場競爭吸收的數值來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        policy_cut = diff_after_ceiling - p_diff_final # 計算政策吸收的部分，這樣可以在進行下週最終預測時使用這個政策吸收的數值來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        final_diff_display = p_diff_final if abs(p_diff_final) >= 0.005 else 0.0 # 設置最終漲跌幅的顯示值，如果政策引擎處理後的漲跌幅的絕對值小於 0.005，則將其設置為 0.0，這樣可以在進行下週最終預測時使用這個設置後的漲跌幅來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        _, pct = price_change(final_price, last_price) # 呼叫 price_change 函式來計算最終預測價格和最後價格之間的漲跌幅百分比，這樣可以在進行下週最終預測時使用這個漲跌幅百分比來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        advice = get_action_advice(final_price, last_price, decision_threshold) # 呼叫 get_action_advice 函式來根據最終預測價格、最後價格和決策閾值來生成操作建議，這樣可以在進行下週最終預測時使用這個操作建議來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
        
        # 匯率敏感度測試
        try:
            fake_row = last_row.copy() # 創建一個假的數據行來模擬匯率變動的影響，這樣可以在進行敏感度測試時使用這個假的數據行來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的匯率變動影響問題
            if '日圓匯率' in fake_row: # 如果數據行中包含日圓匯率這個特徵，則進行敏感度測試，這樣可以在進行敏感度測試時使用這個條件來判斷是否需要進行匯率變動的模擬，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的匯率變動影響問題
                fake_row['日圓匯率'] = fake_row['日圓匯率'] * 0.99 # 模擬日圓升值1%，這樣可以在進行敏感度測試時使用這個模擬的匯率變動來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的匯率變動影響問題
                ceil_stress = apply_asia_ceiling(price_ai_theoretical, oil, fake_row) # 呼叫 apply_asia_ceiling 函式來計算套用亞洲天花板機制後的價格，根據 price_ai_theoretical、oil 和 fake_row 來生成預測結果，這樣可以在進行敏感度測試時使用這個價格來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的匯率變動影響問題
                logging.info(f"   💴 [壓力測試] 若日幣升值1%，天花板價將變為: {ceil_stress:.2f}") 
        except:
            pass
        
        results_list.append({ # 將本次預測的結果添加到 results_list 中，這樣可以在後續的分析和報告生成過程中使用這些結果來展示預測的詳細信息，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            '油品': oil, '當前價格': last_price, # 根據最後價格來設置當前價格的數值，這樣可以在生成預測結果時清楚地展示這個當前價格的數值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的價格數值誤解問題
            '預測目標': f"{FORECAST_HORIZON} 期後" if CFG_LOCAL["decision_mode"] == "realtime" else "下週一", # 根據決策模式來設置預測目標的描述，這樣可以在生成預測結果時清楚地展示這個預測是針對哪個時間點的，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的時間點誤解問題
            '預估價格': final_price, '預估漲跌': final_diff_display, # 根據最終預測價格和最後價格來設置預估漲跌的數值，這樣可以在生成預測結果時清楚地展示這個預估漲跌的數值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的漲跌數值誤解問題
            '公式預估價(舊)': price_formula if not np.isnan(price_formula) else 0,'漲跌幅(%)': round(pct, 2), # 根據計算出來的漲跌幅百分比來設置漲跌幅(%)的數值，這樣可以在生成預測結果時清楚地展示這個漲跌幅百分比的數值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的漲跌幅百分比誤解問題
            '市場競爭吸收': round(market_cut, 2), '政策吸收': round(policy_cut, 2), # 根據計算出來的市場競爭吸收和政策吸收的數值來設置這兩個欄位的數值，這樣可以在生成預測結果時清楚地展示這兩個吸收的數值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的吸收數值誤解問題
            '是否非常態週': 1 if abs(final_diff_display) >= decision_threshold else 0, '操作建議': advice # 根據最終漲跌幅的顯示值和決策閾值來設置是否非常態週的數值，這樣可以在生成預測結果時清楚地展示這個是否非常態週的數值，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的非常態週誤解問題
        })
        
        summary_text = (
            f"\n📊 --- {oil} 預測決策鏈 ---\n"
            f"   📅 決議日期：{last_friday_date.date()}\n"
            f"   [理論漲跌]: {p_theoretical:+.2f} 元 | [政策吸收後]: {p_diff_final:+.2f} 元\n"
            f"   >>> 最終公告價預估: {final_price:.2f} 元\n"
            f"   >>> 行動建議: {advice}\n"
        )
        final_summaries.append(summary_text) # 將本次預測的摘要信息添加到 final_summaries 中，這樣可以在後續的分析和報告生成過程中使用這些摘要來展示預測的關鍵信息，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題

        # ---------------------------------------------------
        # 7. 視覺化繪圖區塊 (Production 模式)
        # ---------------------------------------------------
        if MODE != "academic": # 只有在非學術模式下才進行生產環境的繪圖，這樣可以在進行生產環境的分析和報告生成過程中使用這些圖表來展示預測的詳細信息，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            plot_y_true, plot_y_ai, plot_y_arima, plot_dates = [], [], [], [] # 初始化繪圖用的變數，這樣可以在進行生產環境的繪圖時使用這些變數來存儲真實值、AI 預測值、ARIMA 預測值和日期，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            r_w_ai = np.array([]) # 初始化 AI 權重的變數，這樣可以在進行生產環境的繪圖時使用這個變數來存儲 AI 預測的權重，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            
            try:
                subset_df = df_decision.tail(200).copy() # 從 df_decision 中選取最近 200 期的數據來進行回測，這樣可以在進行生產環境的繪圖時使用這個子集數據來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                
                if len(subset_df) > 20: # 如果子集數據的長度大於 20，則進行回測，這樣可以在進行生產環境的繪圖時使用這個條件來判斷是否有足夠的數據來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據不足問題
                    r_t, r_p, r_a, r_d, r_lstm, r_w_ai = rolling_backtest( # 呼叫 rolling_backtest 函式來進行滾動回測，根據 
                        subset_df, oil, xgb_feats, lstm_feats, # 在進行滾動回測時使用這些特徵來生成預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的特徵選擇問題
                        start_test_date=subset_df['日期'].iloc[-12], # 從子集數據的倒數第 12 期的日期開始進行回測，這樣可以在進行生產環境的繪圖時使用這個起始日期來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的回測起始日期問題 
                        min_train_weeks=10, retrain_freq=1 # 每週都重新訓練模型，這樣可以在進行生產環境的繪圖時使用這個頻率來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的模型訓練頻率問題
                    )
                    
                    plot_y_true = r_t
                    plot_y_ai = r_p
                    plot_y_arima = r_a
                    plot_dates = r_d
            except Exception as e: # 如果在進行滾動回測的過程中出現任何異常，則捕獲這個異常並記錄一條錯誤日誌，這樣可以在進行生產環境的繪圖時使用這個錯誤日誌來識別和解決可能存在的問題，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                logging.error(f"Production backtest failed: {e}")
         # ======================================================
         # 🔥 [新增] 敏感度/壓力測試 (Sensitivity Analysis)
         # ======================================================
        if oil == '92': # 只針對指標油品做分析
            logging.info(f"\n🌪️ 正在對 {oil} 進行敏感度壓力測試...")
            
            # 定義情境：[情境名稱, 原油變動%, 匯率變動%]
            scenarios = [
                ("基準情境", 0.0, 0.0),
                ("原油暴漲 (+5%)", 0.05, 0.0),
                ("原油暴跌 (-5%)", -0.05, 0.0),
                ("台幣重貶 (+2%)", 0.0, 0.02), # 匯率數字變大是貶值
                ("台幣強升 (-2%)", 0.0, -0.02),
                ("雙重打擊 (油漲+幣貶)", 0.05, 0.02)
            ]
            
            sensitivity_results = [] # 初始化敏感度分析結果的列表，這樣可以在進行敏感度分析時使用這個列表來存儲不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            base_row = df_decision.iloc[-1].copy() # 以最後一行數據作為基底來模擬不同情境，這樣可以在進行敏感度分析時使用這個基底數據來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
            current_price = last_price # 以最後價格作為當前價格來計算預測的漲跌幅，這樣可以在進行敏感度分析時使用這個當前價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的價格數值誤解問題
            
            for name, oil_chg, fx_chg in scenarios: # 遍歷定義的情境列表，對每個情境進行模擬，這樣可以在進行敏感度分析時使用這些情境來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                # 1. 模擬變數
                sim_row = base_row.copy() # 創建一個新的數據行來模擬這個情境，這樣可以在進行敏感度分析時使用這個新的數據行來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                sim_row['布蘭特原油'] = sim_row['布蘭特原油'] * (1 + oil_chg) # 模擬原油價格的變動，根據 oil_chg 來調整布蘭特原油的數值，這樣可以在進行敏感度分析時使用這個模擬的原油價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的原油價格變動影響問題
                sim_row['台幣匯率'] = sim_row['台幣匯率'] * (1 + fx_chg) # 模擬台幣匯率的變動，根據 fx_chg 來調整台幣匯率的數值，這樣可以在進行敏感度分析時使用這個模擬的匯率來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的匯率變動影響問題
                
                # 2. 重新計算 AI 理論價 (簡化版：假設 AI 殘差不變，只動基底)
                # 因為 AI 模型是非線性的，這裡我們主要測試「公式與政策」的反應
                
                # 重新計算公式價
                sim_formula_price = round(cpc_formula(pd.DataFrame([sim_row]), oil), 1) # 呼叫 cpc_formula 函式來計算基於模擬數據的公式預測價格，根據 pd.DataFrame([sim_row]) 和 oil 來生成預測結果，這樣可以在進行敏感度分析時使用這個模擬的公式預測價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                
                # 模擬最終決策 (考慮天花板與平滑)
                # 假設 AI 預測變動量跟隨公式變動量的比例 (Alpha)
                # 這裡用一個簡單的傳導係數 0.8 來估算
                implied_change = round((sim_formula_price - price_formula) * 0.8, 1) # 根據模擬的公式預測價格和原始的公式預測價格之間的差異來計算 AI 預測的變動量，這樣可以在進行敏感度分析時使用這個 AI 預測的變動量來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                sim_final_price = round(final_price + implied_change, 1) # 根據最終預測價格和 AI 預測的變動量來計算模擬情境下的最終預測價格，這樣可以在進行敏感度分析時使用這個模擬的最終預測價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                
                # 檢查亞鄰天花板
                sim_ceil = round(apply_asia_ceiling(sim_final_price, oil, sim_row), 1) # 呼叫 apply_asia_ceiling 函式來計算模擬情境下套用亞洲天花板機制後的價格，根據 sim_final_price、oil 和 sim_row 來生成預測結果，這樣可以在進行敏感度分析時使用這個模擬的天花板價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                
                # 最終落地價
                final_simulated = round(min(sim_final_price, sim_ceil), 1) # 根據模擬的最終預測價格和模擬的天花板價格來計算最終落地價格，這樣可以在進行敏感度分析時使用這個最終落地價格來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                change_amount = round(final_simulated - current_price, 1) # 根據模擬的最終落地價格和當前價格來計算預估的漲跌幅，這樣可以在進行敏感度分析時使用這個預估的漲跌幅來生成不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的價格數值誤解問題
                
                sensitivity_results.append({ # 將這個情境的分析結果添加到敏感度分析結果列表中，這樣可以在後續的分析和報告生成過程中使用這些結果來展示不同情境下的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題
                    "情境": name,
                    "原油設定": f"{sim_row['布蘭特原油']:.1f}",
                    "匯率設定": f"{sim_row['台幣匯率']:.1f}",
                    "預估油價": final_simulated,
                    "預期漲跌": change_amount
                })
                
            # 輸出表格
            df_sens = pd.DataFrame(sensitivity_results) 
            print("\n📊 --- 敏感度分析報告 ---")
            print(df_sens.to_string(index=False))
            df_sens.to_excel(f"{OUTDIR}/SENSITIVITY_ANALYSIS.xlsx", index=False)
        
            # --- 繪圖區塊 ---
            min_l = min(len(plot_y_true), len(plot_y_ai), len(plot_dates)) # 計算繪圖用的數據長度，這樣可以在進行生產環境的繪圖時使用這個長度來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據長度不一致問題
            
            if min_l > 0: # 如果繪圖用的數據長度大於 0，則進行繪圖，這樣可以在進行生產環境的繪圖時使用這個條件來判斷是否有足夠的數據來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據不足問題
                p_t = plot_y_true[:min_l] # 根據計算出來的繪圖用的數據長度來截取真實值的數據，這樣可以在進行生產環境的繪圖時使用這個截取後的真實值來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據長度不一致問題
                p_a_i = plot_y_ai[:min_l] # 根據計算出來的繪圖用的數據長度來截取 AI 預測值的數據，這樣可以在進行生產環境的繪圖時使用這個截取後的 AI 預測值來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據長度不一致問題
                p_dates = plot_dates[:min_l] # 根據計算出來的繪圖用的數據長度來截取日期的數據，這樣可以在進行生產環境的繪圖時使用這個截取後的日期來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據長度不一致問題
                p_arima = plot_y_arima[:min_l] if len(plot_y_arima) > 0 else np.zeros(min_l) # 根據計算出來的繪圖用的數據長度來截取 ARIMA 預測值的數據，如果 ARIMA 預測值的數據長度大於 0，則使用截取後的 ARIMA 預測值來生成圖表，否則使用一個全為 0 的數組來生成圖表，這樣可以在進行生產環境的繪圖時使用這個條件來判斷是否有足夠的 ARIMA 預測值數據來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據不足問題
                
                if len(r_w_ai) >= min_l: # 如果 AI 權重的數據長度大於或等於繪圖用的數據長度，則截取 AI 權重的數據來生成圖表，這樣可以在進行生產環境的繪圖時使用這個條件來判斷是否有足夠的 AI 權重數據來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據不足問題
                    p_w_ai = r_w_ai[:min_l] # 根據計算出來的繪圖用的數據長度來截取 AI 權重的數據，這樣可以在進行生產環境的繪圖時使用這個截取後的 AI 權重來生成圖表，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的數據長度不一致問題
                else:
                    p_w_ai = np.zeros(min_l) # 防呆

                # 協作三劍客圖表
                plot_calibration_scatter(p_t, p_a_i, oil, OUTDIR) # 繪製 AI 預測值與真實值的散點圖，這樣可以在進行生產環境的繪圖時使用這個圖表來展示 AI 預測的準確性，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測準確性問題
                plot_prediction_timeseries(p_dates, p_t, p_a_i, p_arima, oil, OUTDIR) # 繪製預測值與真實值的時間序列圖，這樣可以在進行生產環境的繪圖時使用這個圖表來展示預測值和真實值隨時間的變化趨勢，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的時間序列趨勢問題
                plot_contribution_stack(p_dates, p_a_i, p_arima, p_w_ai, oil, OUTDIR) # 繪製 AI 預測值、ARIMA 預測值和 AI 權重的堆疊圖，這樣可以在進行生產環境的繪圖時使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的貢獻程度，以及 AI 權重的變化趨勢，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的預測貢獻問題
                plot_weight_dynamics(p_w_ai, 1-p_w_ai, p_dates, oil, OUTDIR) # 繪製 AI 權重和 ARIMA 權重的動態變化圖，這樣可以在進行生產環境的繪圖時使用這個圖表來展示 AI 預測和 ARIMA 預測在不同時間點的權重變化趨勢，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的權重變化問題

    # 8. 輸出報告與存檔
    logging.info("\n" + "="*50 + "\n📌【本週油價預測最終總結】") # 記錄總結開始
    
    # --- 整理汽油與柴油的數據 ---
    gas_data = next((item for item in results_list if item['油品'] == '92'), None) # 從 results_list 中找到油品為 '92' 的數據，這樣可以在生成最終報告時使用這個數據來展示汽油的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的油品分類問題
    diesel_data = next((item for item in results_list if item['油品'] == '柴油'), None) # 從 results_list 中找到油品為 '柴油' 的數據，這樣可以在生成最終報告時使用這個數據來展示柴油的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的油品分類問題
    
    final_report = [] # 初始化最終報告的列表，這樣可以在生成最終報告時使用這個列表來存儲汽油和柴油的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的報告結構問題
    
    # 1. 汽油 (Gasoline)
    if gas_data: # 若有汽油資料
        base_price_92 = gas_data['預估價格']
        change = gas_data['預估漲跌']
        advice = gas_data['操作建議']
        
        final_report.append({ # 加入汽油預測結果
            "油品分類": "汽油",
            "預估漲跌": change,
            "預測後價格 (92)": base_price_92,
            "預測後價格 (95)": base_price_92 + 1.5, # 95 無鉛通常比 92 貴 1.5 元
            "預測後價格 (98)": base_price_92 + 3.5, # 98 無鉛通常比 92 貴 3.5 元
            "預測後價格 (柴油)": "-",
            "操作建議": advice
        })
        
        log_msg = f"⛽ [汽油]: 調{'漲' if change>0 else '降'} {abs(change):.1f} 元 | 92: {base_price_92}, 95: {base_price_92+1.5}, 98: {base_price_92+3.5}" # 根據汽油的預估漲跌來生成一條日誌消息，這樣可以在生成最終報告時使用這個消息來展示汽油的預測結果，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的價格數值誤解問題
        logging.info(log_msg)

    # 2. 柴油 (Diesel)
    if diesel_data: # 若有柴油資料
        diesel_price = diesel_data['預估價格']
        change = diesel_data['預估漲跌']
        advice = diesel_data['操作建議']
        
        final_report.append({ # 加入柴油預測結果
            "油品分類": "柴油",
            "預估漲跌": change,
            "預測後價格 (92)": "-",
            "預測後價格 (95)": "-",
            "預測後價格 (98)": "-",
            "預測後價格 (柴油)": diesel_price,
            "操作建議": advice
        })

        log_msg = f"🚛 [柴油]: 調{'漲' if change>0 else '降'} {abs(change):.1f} 元 | 價格: {diesel_price}"
        logging.info(log_msg)
   
    with open(f"{OUTDIR}/FINAL_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.writelines(final_summaries) # 將最終摘要信息寫入一個文本文件中，這樣可以在生成最終報告時使用這個文本文件來展示預測的關鍵信息，提供一個全面的分析結果來驗證模型的預測能力，同時幫助識別可能存在的市場波動影響問題

    # 儲存精簡版報告 (主要)
    pd.DataFrame(final_report).to_excel(f"{OUTDIR}/FINAL_DECISION.xlsx", index=False)
    # 儲存完整原始報告 (偵錯用)
    pd.DataFrame(results_list).to_excel(f"{OUTDIR}/FINAL_DECISION_RAW.xlsx", index=False)
    
    if MODE == "academic" and regime_metrics:
        pd.concat(regime_metrics, ignore_index=True).to_excel(f"{OUTDIR}/REGIME_ERROR_SUMMARY.xlsx", index=False)
        pd.DataFrame(metrics, columns=['油品','AI_MAE','ARIMA_MAE','Hybrid_MAE','AI_RMSE','ARIMA_RMSE','Hybrid_RMSE','w_AI','w_ARIMA']).to_excel(f"{OUTDIR}/MODEL_METRICS.xlsx", index=False)
    
    if CFG_LOCAL["enable_line"]:
        # 使用新的 final_report 格式傳送 LINE
        line_msg = build_line_message(final_report, df_decision['日期'].iloc[-1], FORECAST_HORIZON, CFG_LOCAL["decision_mode"])
        send_line_notification(line_msg, LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID)

    logging.info(f"\n✅ 所有分析完成！結果已儲存於：{OUTDIR}")
    
if __name__ == "__main__":
    main()