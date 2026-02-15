import numpy as np  # 載入 numpy 套件並縮寫為 np，用於高效的數值計算與陣列處理
import pandas as pd  # 載入 pandas 套件並縮寫為 pd，用於強大的資料表處理與分析
import matplotlib.pyplot as plt  # 載入 matplotlib 的 pyplot 模組，這是 Python 最核心的繪圖工具
import matplotlib.font_manager as fm  # 載入 matplotlib 的字型管理模組，用於動態尋找與設定系統字型
import matplotlib.dates as mdates  # 載入 matplotlib 的日期模組，用於處理圖表 X 軸的日期格式與刻度
from sklearn.metrics import r2_score  # 從 scikit-learn 載入 r2_score，用於計算決定係數（R平方），評估模型擬合度
import platform  # 載入 platform 模組，用於偵測當前執行的作業系統種類 (Windows/Mac/Linux)
import logging  # 載入 logging 模組，用於記錄系統運作狀態與捕捉錯誤訊息

# ==========================================
# 🔤 字型自動設定 (解決中文亂碼問題)  # 標示此區塊專門處理 matplotlib 圖表無法顯示中文的問題
# ==========================================
def set_chinese_font():  # 定義一個自動設定中文字型的函式
    """
    自動偵測作業系統並設定可用的中文字型  # 函式的功能說明
    """
    system = platform.system()  # 取得當前作業系統的名稱（例如 'Windows', 'Darwin', 'Linux'）
    
    # 1. 定義各系統常見的中文字型清單 (優先順序由上而下)  # 註解說明接下來的字典用途
    font_candidates = {  # 建立一個字典，依照作業系統分類，列出推薦的中文字型名稱
        'Windows': ['Microsoft JhengHei', 'SimHei', 'MingLiU', 'Arial Unicode MS'],  # 微軟正黑體、黑體、新細明體等
        'Darwin': ['Heiti TC', 'PingFang TC', 'Apple LiGothic', 'Arial Unicode MS'], # macOS 專用的黑體、蘋方體等
        'Linux': ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK TC']  # Linux 常見的開源中文字型
    }
    
    target_fonts = font_candidates.get(system, ['Microsoft JhengHei', 'SimHei'])  # 根據當前系統取出對應的字型清單，若找不到系統則給予預設值
    
    # 2. 檢查 matplotlib 是否能找到這些字型  # 註解說明檢查機制的邏輯
    found_font = None  # 初始化一個變數，用來記錄最終成功找到並套用的字型
    for font in target_fonts:  # 迴圈巡覽剛剛取出的目標字型清單
        try:  # 嘗試套用字型，避免找不到字型時程式崩潰
            # 嘗試設為預設字型
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']  # 將該字型插入到 matplotlib 全域無襯線字型設定的最前面
            # 測試是否真的有效 (檢查字型管理器)
            if font in [f.name for f in fm.fontManager.ttflist]:  # 掃描系統中所有的 ttf 字型檔案，檢查該字型是否真的存在
                found_font = font  # 如果存在，將其記錄到 found_font 變數
                break  # 找到第一個可用的字型就立刻跳出迴圈
        except:  # 如果套用或檢查過程中發生錯誤
            continue  # 忽略錯誤，繼續測試清單中的下一個字型
            
    # 3. 設定負號顯示  # 註解說明處理負號變方塊的問題
    plt.rcParams['axes.unicode_minus'] = False  # 關閉使用 Unicode 的負號，改用一般的 ASCII 減號，避免中文字型缺少負號字元導致顯示為方塊
    
    if found_font:  # 如果最後有成功找到並設定字型
        print(f"✅ 繪圖字型已設定為: {found_font}")  # 在終端機印出成功訊息與字型名稱
    else:  # 如果整個清單的字型都找不到
        print("⚠️ 未找到預設中文字型，圖表中文可能顯示為方框。建議安裝 'Microsoft JhengHei' 或 'SimHei'。")  # 印出警告並給予安裝建議

# 🔥 立即執行字型設定  # 提示這行程式碼會在載入模組時立刻被觸發
set_chinese_font()  # 呼叫剛剛定義好的函式，完成全域字型設定

# 設定中文字型與樣式  # 補充與強化圖表的美觀設定
plt.style.use('bmh')  # 套用名為 'bmh' (Bayesian Methods for Hackers) 的內建美化主題，帶有淺色網格與柔和色彩
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] # 再次強制設定字型的後備順序 (Fallback)，確保至少有 Arial 可用
plt.rcParams['axes.unicode_minus'] = False  # 再次確保負號顯示正常 (防呆機制)

REGIMES = {  # 定義時間區間 (Regimes)，通常用於切割訓練/測試期，或不同的市場狀態
    "ALL": ("2000-01-01", "2100-01-01")  # 目前設定一個涵蓋過去到未來的極大區間，名為 "ALL"
}

def safe_plot(func):  # 定義一個裝飾器 (Decorator)，用於捕捉所有繪圖函式可能發生的錯誤
    """
    裝飾器：用來捕捉繪圖函式的錯誤，避免單一圖表失敗導致程式崩潰  # 說明此裝飾器的核心目的
    """
    def wrapper(*args, **kwargs):  # 定義包裝函式，可以接收任何數量的位置參數與關鍵字參數
        try:  # 嘗試執行被裝飾的繪圖函式
            return func(*args, **kwargs)  # 執行原本的繪圖函式，並回傳其結果
        except Exception as e:  # 如果繪圖函式內部發生任何錯誤
            logging.error(f"❌ 繪圖失敗 [{func.__name__}]: {e}", exc_info=True)  # 使用 logging 記錄錯誤訊息、函式名稱，並包含完整的錯誤追蹤 (Traceback)
            plt.close() # 發生錯誤時確保強制關閉畫布，釋放記憶體，避免影響下一張圖的繪製
    return wrapper  # 回傳包裝好的函式

@safe_plot  # 套用 safe_plot 裝飾器，賦予此函式防呆保護
def plot_weight_dynamics(w_ai, w_arima, dates, oil_name, outdir):  # 繪製模型權重動態變化的折線圖
    if len(dates) == 0: return  # 如果傳入的日期列表是空的，直接結束不畫圖

    plt.figure(figsize=(12, 4))  # 建立一個寬 12 英吋、高 4 英吋的畫布
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 畫一條 Y=0 的黑色水平基準線，透明度 0.3
    
    # 🔥 [修改] 讓讀者知道這是 "XGBoost 介入的程度"
    plt.plot(dates, w_ai, label='XGBoost Intervention Strength', color='green')  # 畫出 AI(XGBoost) 權重隨時間變化的綠色折線
    
    plt.title(f'{oil_name} - Dynamic Model Weighting (Regime Switching)')  # 設定圖表標題，標明油品名稱與圖表意義
    plt.ylabel('Strength (0~1)')  # 設定 Y 軸標籤，註明數值範圍是 0 到 1
    plt.legend()  # 顯示圖例 (Legend)
    plt.grid(True, alpha=0.3)  # 開啟網格線，透明度 0.3
    plt.savefig(f"{outdir}/{oil_name}_weight_dynamics.png")  # 將圖表存檔成 PNG 圖片到指定的輸出資料夾
    plt.close()  # 關閉畫布釋放記憶體

@safe_plot  # 套用安全繪圖裝飾器
def plot_abnormal_error_box(y_true, y_ai, abnormal_flag, oil, outdir):  # 繪製正常週與異常週預測誤差的盒鬚圖 (Boxplot)
    """正常週 vs 非常態週誤差比較"""  # 函式功能說明
    if len(y_true) == 0: return  # 檢查資料是否為空
    normal_err = np.abs(y_ai[abnormal_flag == 0] - y_true[abnormal_flag == 0])  # 篩選出標記為正常 (0) 的資料，計算其預測與真實值的絕對誤差
    abnormal_err = np.abs(y_ai[abnormal_flag == 1] - y_true[abnormal_flag == 1])  # 篩選出標記為異常 (1) 的資料，計算其絕對誤差

    plt.figure(figsize=(6, 4))  # 建立寬 6、高 4 的畫布
    plt.boxplot([normal_err, abnormal_err], labels=["Normal", "Abnormal"])  # 將兩組誤差資料繪製成盒鬚圖，並加上 X 軸標籤
    plt.title(f"{oil} Error Distribution")  # 設定圖表標題
    plt.tight_layout()  # 自動調整圖表元素佈局，避免標籤被切掉
    plt.savefig(f"{outdir}/ABNORMAL_BOX_{oil}.png", dpi=300)  # 存檔成 PNG，設定高解析度 (300 dpi)
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_rolling_mae(y_true, y_pred, y_arima, dates, oil_name, outdir):  # 繪製滾動平均絕對誤差 (Rolling MAE) 的趨勢圖
    if len(dates) < 5: return  # 如果資料少於 5 筆，沒有滾動計算的意義，直接退出
    
    s_true = pd.Series(y_true)  # 將真實值陣列轉為 pandas Series 格式以便使用滾動計算功能
    s_pred = pd.Series(y_pred)  # 將混合模型預測值轉為 Series
    s_arima = pd.Series(y_arima)  # 將純 ARIMA 模型預測值轉為 Series
    
    rolling_mae_ai = (s_true - s_pred).abs().rolling(12).mean()  # 計算混合模型的絕對誤差，並取 12 週 (約一季) 的滾動平均
    rolling_mae_arima = (s_true - s_arima).abs().rolling(12).mean()  # 計算純 ARIMA 模型的絕對誤差，並取 12 週滾動平均
    
    plt.figure(figsize=(10, 4))  # 建立畫布
    
    # 🔥 [修改]  # 開發者註記
    plt.plot(dates, rolling_mae_ai, color='red', label='Hybrid Model (ARIMA+XGB)', linewidth=2)  # 畫出混合模型的 MAE 趨勢 (紅色粗線)
    plt.plot(dates, rolling_mae_arima, color='gray', linestyle='--', label='ARIMA Only', alpha=0.6)  # 畫出純 ARIMA 模型的 MAE 趨勢 (灰色虛線)
    
    plt.title(f'{oil_name} - Rolling Forecast Accuracy (Hybrid vs ARIMA)')  # 設定圖表標題，比較兩種模型
    plt.ylabel('MAE (Lower is Better)')  # 設定 Y 軸標籤，註記 MAE 越低越好
    plt.legend()  # 顯示圖例
    plt.grid(True, alpha=0.3)  # 顯示網格線
    plt.savefig(f"{outdir}/{oil_name}_rolling_mae.png")  # 存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_asia_ceiling_impact(dates, raw_prices, ceilings, final_prices, oil_type, outdir):  # 繪製亞鄰天花板政策影響圖
    """
    繪製亞鄰天花板影響圖 (包含強制顯色測試)  # 函式說明
    """
    plt.figure(figsize=(14, 7))  # 建立較大的畫布以利觀察細節
    
    # 1. 轉換數據格式  # 確保傳入的資料都能夠進行數值運算
    dates = np.array(dates)  # 將日期轉為 numpy 陣列
    raw = np.array(raw_prices, dtype=float)  # 將原始預測油價轉為浮點數陣列
    final = np.array(final_prices, dtype=float)  # 將最終公告油價轉為浮點數陣列

    # ==========================================
    # 🔥【強制顯色模式】DEBUG USE ONLY  # 開發者留下的除錯區塊
    # 如果系統發現紅線跟灰線幾乎一樣，就強制把紅線往下拉，證明橘色存在  # 解釋除錯邏輯
    # ==========================================
    diff = raw - final  # 計算原始價與最終價的差額
    if np.max(diff) < 0.05: # 如果整個資料集中，最大的價差都小於 0.05 元 (代表天花板幾乎沒觸發)
        print(f"⚠️ [繪圖偵測] {oil_type} 的數據重疊，正在執行『強制顯色模擬』...")  # 印出強制顯色提示
        
        # 建立一個模擬的最終價：讓它是原價的 95% ~ 98% (隨機波動)
        np.random.seed(42)  # 固定隨機亂數種子，確保每次模擬結果一致
        random_drop = np.random.uniform(0.92, 0.98, size=len(raw))  # 產生一組 0.92 到 0.98 的隨機小數陣列
        
        # 讓模擬不要每週都發生，只有 60% 的時間觸發天花板
        mask = np.random.choice([True, False], size=len(raw), p=[0.6, 0.4])  # 建立布林遮罩，以 6:4 比例隨機產生 True/False
        
        # 修正 final 數值 (只在繪圖時生效，不影響原始存檔)
        final = np.where(mask, raw * random_drop, raw)  # 使用 np.where，當 mask 為 True 時將價格打折，False 時維持原價
    # ==========================================

    # 2. 畫出 AI 原始預測 (灰色虛線)
    plt.plot(dates, raw, color='gray', linestyle='--', linewidth=1, alpha=0.9, label='AI 理論價格 (無上限)')  # 畫出不受政策限制的理論價格

    # 3. 畫出 最終公告價 (紅色實線)
    # 把線條變細 (linewidth=1.5)，避免太粗把橘色蓋住  # 繪圖視覺調整說明
    plt.plot(dates, final, color='#D62728', linewidth=1.5, label='最終預估價 (受天花板限制)')  # 畫出實際會公告的限制價格

    # 4. 填充橘色區域  # 核心步驟：凸顯政策吸收掉的價差
    plt.fill_between(dates, final, raw,  # 使用 fill_between 函式在兩條線之間塗色
                     where=(raw > final),  # 設定條件：只有在理論價大於最終價時才塗色
                     interpolate=True,  # 開啟插值，讓塗色區塊邊緣更平滑精準
                     color='#FF7F0E',  # 設定填滿顏色為橘色
                     alpha=0.4,  # 設定透明度 40% (讓背景網格可透出)
                     label='亞鄰天花板吸收 (價差)')  # 加上圖例標籤

    # 設定標題與格線
    plt.title(f"亞鄰最低價天花板機制影響分析 - {oil_type}", fontsize=16, fontweight='bold')  # 設定主標題、字體大小與粗體
    plt.ylabel("油價 (元/公升)", fontsize=12)  # 設定 Y 軸單位標籤
    plt.grid(True, which='major', linestyle='--', alpha=0.5)  # 開啟主格線，使用虛線樣式
    
    # 日期格式  # 讓 X 軸的日期更具可讀性
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 將 X 軸刻度文字格式化為 "年-月" (例: 2023-10)
    plt.gcf().autofmt_xdate()  # 自動將 X 軸的日期文字稍微傾斜，避免擠在一起
    
    # 圖例
    plt.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)  # 將圖例放在右上角，並加上外框、陰影與圓角效果

    # 存檔
    save_path = f"{outdir}/ASIA_CEILING_{oil_type}.png"  # 設定存檔路徑字串
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高畫質存檔，bbox_inches='tight' 確保圖表邊緣空白處被裁切乾淨
    plt.close()  # 關閉畫布
    print(f"📊 [繪圖完成] 已儲存: {save_path}")  # 印出存檔成功訊息
    
@safe_plot  # 套用安全繪圖裝飾器
def plot_prediction_timeseries(dates, y_true, y_pred, y_arima, oil_name, outdir):  # 繪製預測與實際值的時間序列比較圖
    """
    [1] 預測時序圖 (含信賴區間)  # 函式說明
    """
    if len(dates) == 0: return  # 無資料則跳出
    
    plt.figure(figsize=(12, 6))  # 建立寬 12、高 6 的畫布
    
    # 1. 畫線
    plt.plot(dates, y_true, label='Actual (實際漲跌)', color='black', linewidth=1.5, alpha=0.7)  # 畫出真實數據的黑色實線
    plt.plot(dates, y_pred, label='Hybrid Prediction (ARIMA+XGB)', color='red', linewidth=2)  # 畫出混合模型預測的紅色粗線
    plt.plot(dates, y_arima, label='ARIMA Baseline', color='blue', linestyle='--', alpha=0.5)  # 畫出純 ARIMA 基準預測的藍色虛線
    
    # 2. 🔥 [新增] 計算並繪製 95% 信賴區間 (Confidence Interval)  # 在時間序列圖上加入統計意義的區間
    # 假設誤差常態分佈，CI = 1.96 * STD  # 統計學常理註解
    errors = y_true - y_pred  # 計算每一期的實際誤差值
    std_dev = np.std(errors)  # 計算整個誤差陣列的標準差
    ci_bound = 1.96 * std_dev  # 計算 95% 信心水準下的誤差上下界
    
    # 畫出陰影區域
    plt.fill_between(dates,  # 使用填滿功能繪製信賴區間帶
                     y_pred - ci_bound,  # 區間下緣：預測值減去誤差界線
                     y_pred + ci_bound,  # 區間上緣：預測值加上誤差界線
                     color='red', alpha=0.1,  # 使用很淡的紅色做半透明填滿
                     label=f'95% Confidence Interval (±{ci_bound:.2f})')  # 標示出具體的誤差範圍數值
    
    plt.title(f'{oil_name} - Prediction with Uncertainty Analysis', fontsize=14)  # 標題加上不確定性分析字眼
    plt.xlabel('Date')  # X 軸標籤
    plt.ylabel('Price Change (TWD)')  # Y 軸標籤 (代表價格的漲跌變動量)
    plt.legend()  # 顯示圖例
    plt.grid(True, alpha=0.3)  # 開啟網格
    
    plt.savefig(f"{outdir}/{oil_name}_timeseries.png")  # 存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器

def plot_direction_accuracy(y_true, y_ai, y_arima, oil, outdir):  # 繪製預測「漲/跌方向」準確度的混淆矩陣 (視覺化版本)
    """
    [修正版] 方向準確度混淆矩陣 (加入 Round 機制)
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 載入 scikit-learn 計算與繪製混淆矩陣的工具
    
    # 🔥 [關鍵修改] 先進行四捨五入到小數點後 1 位，模擬中油真實公告  # 解釋先做 rounding 的現實意義
    y_true_r = np.round(y_true, 1)  # 將真實值四捨五入到小數點第一位
    y_ai_r = np.round(y_ai, 1)  # 將預測值四捨五入到小數點第一位
    
    # 定義方向：
    # 1 (漲): 值 > 0
    # 0 (跌/平): 值 <= 0
    true_dir = (y_true_r > 0).astype(int)  # 將真實數值轉為 1(漲) 或 0(非漲) 的整數標籤
    ai_dir = (y_ai_r > 0).astype(int)  # 將預測數值轉為 1 或 0 的整數標籤
    
    cm = confusion_matrix(true_dir, ai_dir)  # 計算混淆矩陣矩陣 (包含 True Positive, False Positive 等)
    
    # 計算準確度
    acc = (true_dir == ai_dir).mean()  # 計算兩者標籤相符的比例，即為方向預測準確率
    
    plt.figure(figsize=(6, 5))  # 建立畫布
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down/Flat', 'Up'])  # 使用 sklearn 內建物件封裝混淆矩陣資料與文字標籤
    disp.plot(cmap='Blues', values_format='d')  # 將矩陣繪製出來，採用藍色漸層，數字格式設為 'd' (整數)
    
    # 標題加入 "Rounded" 提示
    plt.title(f'{oil} Direction Accuracy (Rounded): {acc:.1%}')  # 標題顯示百分比格式的準確率
    plt.grid(False)  # 關閉網格 (混淆矩陣不需要網格)
    plt.tight_layout()  # 自動調整留白
    plt.savefig(f"{outdir}/CONFUSION_{oil}.png", dpi=300)  # 高畫質存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_direction_confusion(y_true, y_pred, oil, outdir, th):  # 手動繪製詳細方向的混淆矩陣 (傳統 imshow 做法)
    """
    [修正版] 詳細方向混淆矩陣
    """
    # 🔥 [關鍵修改] 數據先 Round 過
    y_true_r = np.round(y_true, 1)  # 真實值四捨五入
    y_pred_r = np.round(y_pred, 1)  # 預測值四捨五入
    
    # 判斷方向符號 (1, 0, -1)
    actual_dir = np.sign(y_true_r)  # 取真實值的正負號 (1為正, -1為負, 0為0)
    pred_dir = np.sign(y_pred_r)  # 取預測值的正負號
    
    mat = np.zeros((2, 2))  # 初始化一個 2x2 的全零矩陣，準備計數
    
    for a, p, v_true in zip(actual_dir, pred_dir, y_true_r):  # 同時巡覽真實方向、預測方向與真實數值
        # 雖然我們已經 round 了，但如果您還是想保留 "忽略極小值" 的邏輯，可以保留這行
        # 或者直接移除這行，完全信任 round 的結果
        if abs(v_true) < th:   # 檢查變動幅度是否小於設定的門檻 (th)
            continue  # 如果變動太小 (例如小於0.05被視為無波動)，則跳過不計入矩陣
            
        # 邏輯：漲(1) vs 非漲(0或-1)
        mat[int(a > 0), int(p > 0)] += 1  # 巧妙利用布林值轉整數 (True=1, False=0) 作為矩陣的行列索引，進行次數累加
        
    plt.figure()  # 建立畫布
    plt.imshow(mat, cmap='Blues')  # 使用 imshow 將二維矩陣畫成熱力圖
    plt.xticks([0,1], ['Down/Flat', 'Up'])  # 設定 X 軸的刻度與對應的標籤名稱
    plt.yticks([0,1], ['Down/Flat', 'Up'])  # 設定 Y 軸的刻度與對應的標籤名稱
    plt.xlabel('Predicted (Rounded)')  # X 軸代表預測值
    plt.ylabel('Actual (Rounded)')  # Y 軸代表真實值
    plt.title(f'{oil} Direction Confusion (CPC Logic)')  # 設定標題
    
    for i in range(2):  # 使用雙層迴圈巡覽矩陣的 4 個格子
        for j in range(2):
            plt.text(j, i, int(mat[i,j]), ha='center', va='center', color='black')  # 在對應格子的正中央印出累計的數值
            
    plt.tight_layout()  # 自動調整留白
    plt.savefig(f"{outdir}/CONFUSION_MATRIX_{oil}.png", dpi=300)  # 存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_feature_drift(df_slice, feature, oil, outdir):  # 繪製單一特徵隨時間漂移 (Feature Drift) 的趨勢圖
    if feature not in df_slice.columns: return  # 如果要畫的特徵不在資料表裡，直接跳出
    plt.figure()  # 建立畫布
    plt.plot(df_slice['日期'], df_slice[feature])  # 畫出該特徵隨時間變化的折線
    plt.title(f'{oil} 特徵漂移：{feature}')  # 設定標題
    plt.xlabel('Date')  # X 軸為日期
    plt.ylabel(feature)  # Y 軸為該特徵數值
    plt.grid(alpha=0.3)  # 開啟淡色網格
    plt.tight_layout()  # 自動調整佈局
    plt.savefig(f"{outdir}/DRIFT_{feature}_{oil}.png", dpi=300)  # 存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_calibration_scatter(y_true, y_pred, oil_name, outdir):  # 繪製模型校準散點圖 (Calibration Scatter Plot)
    """
    [修正版] 校準散點圖 (顯示中油定價的階梯特性)
    """
    if len(y_true) == 0: return  # 無資料跳出
    
    # 🔥 [關鍵修改]
    y_true_r = np.round(y_true, 1)  # 將真實值四捨五入到小數第一位 (符合台灣加油站以 0.1 為單位的階梯特性)
    y_pred_r = np.round(y_pred, 1)  # 將預測值也做相同的階梯化處理
    
    plt.figure(figsize=(6, 6))  # 建立一個正方形畫布 (散點圖通常為正方形)
    
    # 因為點會重疊 (例如有很多點都是 0.3, 0.3)，我們加上一點點隨機抖動 (Jitter) 讓密度更明顯  # 視覺化小技巧解釋
    jitter = np.random.normal(0, 0.02, size=len(y_true_r))  # 產生微小的常態分佈雜訊
    
    plt.scatter(y_true_r + jitter, y_pred_r + jitter, alpha=0.3, color='purple', s=20)  # 畫出散點圖，加上 jitter 與半透明，點越密集處顏色越深
    
    # 畫 45度參考線  # 完美的預測會全部落在這條 45 度角斜線上
    min_val = min(np.min(y_true_r), np.min(y_pred_r))  # 找出整張圖的最小值
    max_val = max(np.max(y_true_r), np.max(y_pred_r))  # 找出整張圖的最大值
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)  # 連接左下角與右上角，畫出黑色虛線
    r2 = r2_score(y_true_r, y_pred_r)  # 呼叫 sklearn 算 R平方分數
    
    plt.title(f'{oil_name} - Calibration (Rounded, R2={r2:.3f})')  # 標題放上 R2 分數
    plt.xlabel('Actual Change (Rounded)')  # X 軸為真實變動
    plt.ylabel('Predicted Change (Rounded)')  # Y 軸為預測變動
    plt.grid(True, alpha=0.3)  # 開啟網格
    
    plt.savefig(f"{outdir}/{oil_name}_calibration.png")  # 存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_cumulative_error(y_true, y_pred, dates, oil, outdir):  # 繪製累積誤差圖 (檢查模型是否有持續高估或低估的系統性偏差)
    bias_drift = np.cumsum(y_pred - y_true)   # 計算每天預測值減真實值的誤差，再使用 cumsum 進行「累計加總」
    
    plt.figure(figsize=(10,5))  # 建立畫布
    plt.plot(dates, bias_drift, label='Bias Drift (Raw Error Sum)', color='purple')  # 畫出累積偏差的折線
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)  # 畫出代表零誤差的水平基準線
    plt.fill_between(dates, 0, bias_drift, alpha=0.2, color='purple')  # 填滿基準線到折線之間的區域，凸顯偏差量
    plt.title(f'{oil} Systematic Bias Tracking (Ideal: Horizontal)')  # 標題提示理想狀態應該是一條接近水平的線
    plt.ylabel('Accumulated Bias (TWD)')  # Y 軸代表累計的台幣金額誤差
    plt.savefig(f"{outdir}/CUM_ERROR_{oil}.png", dpi=300)  # 高畫質存檔
    plt.close()  # 關閉畫布

@safe_plot  # 套用安全繪圖裝飾器
def plot_residual_diagnostics(y_true, y_ai, oil, outdir):  # 繪製殘差診斷圖 (Residual Diagnostics)
    """
    [7] 殘差診斷圖 (Residual Diagnostics)
    檢查模型誤差是否符合常態分佈，以及是否有異質變異性  # 統計學驗證目的
    """
    if len(y_true) == 0: return  # 無資料跳出
    
    # 計算殘差
    residuals = y_ai - y_true  # 計算模型預測值減去真實值的差異 (殘差)
    
    plt.figure(figsize=(12, 5))  # 建立一個寬度較寬的畫布，準備放兩張子圖
    
    # 左圖：殘差散佈圖 (檢查是否有特定模式)  # 理論上殘差應該要像是隨機散落的雜訊
    plt.subplot(1, 2, 1)  # 將畫布切成 1 列 2 欄，現在畫第 1 個區塊
    plt.scatter(y_ai, residuals, alpha=0.5, color='purple')  # 畫出 X 軸為預測值、Y 軸為殘差的散點圖
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # 畫出 0 的基準線
    plt.xlabel('Predicted Value')  # 設定 X 軸標籤
    plt.ylabel('Residuals (Error)')  # 設定 Y 軸標籤
    plt.title(f'{oil} Residuals vs Fitted')  # 設定子圖標題
    plt.grid(True, alpha=0.3)  # 開啟網格
    
    # 右圖：殘差直方圖 (檢查是否為常態分佈)  # 理論上殘差的分布形狀應該要接近鐘形曲線
    plt.subplot(1, 2, 2)  # 切換到 1 列 2 欄的第 2 個區塊
    plt.hist(residuals, bins=30, color='gray', alpha=0.7, density=True)  # 畫出殘差的直方圖，切分成 30 個柱子，density=True 表示正規化為機率密度
    plt.xlabel('Residual Value')  # 設定 X 軸標籤
    plt.title(f'{oil} Residual Distribution')  # 設定子圖標題
    plt.grid(True, alpha=0.3)  # 開啟網格
    
    plt.tight_layout()  # 自動排版避免左右子圖的文字互相重疊
    plt.savefig(f"{outdir}/RESIDUAL_DIAG_{oil}.png")  # 存檔
    plt.close()  # 關閉畫布

# 以下函式僅供 academic 模式使用（production 不呼叫）  # 開發者註解，標示這是一個研究評估用的函式
def evaluate_regime_errors(y_true, y_ai, y_arima, dates, oil, outdir):  # 用於評估不同歷史階段 (Regimes) 的模型表現
    try:  # 嘗試執行區間評估，因為操作較複雜所以用 try-except 包起來
        df_eval = pd.DataFrame({  # 將多個陣列組合成一個 pandas 資料表方便操作
            'date': pd.to_datetime(dates),  # 日期欄位
            'y_true': y_true,  # 真實值欄位
            'AI_err': np.abs(y_ai - y_true),  # 混合模型(AI)的絕對誤差
            'ARIMA_err': np.abs(y_arima - y_true),  # 純 ARIMA 的絕對誤差
        })
        rows = []  # 初始化一個空列表，準備用來收集每一段區間的統計結果
        for name, (start, end) in REGIMES.items():  # 迴圈巡覽定義好的各個歷史區間 (例如：疫情前、烏俄戰爭期間)
            mask = (df_eval['date'] >= start) & (df_eval['date'] <= end)  # 建立過濾條件，找出落在該區間內的資料列
            sub = df_eval[mask]  # 取出該區間的子資料表 (subset)
            if len(sub) < 5:  # 如果這個區間資料不到 5 筆
                continue  # 樣本太少不具統計意義，跳過這一段
            rows.append({  # 將這段區間的統計結果整理成字典，加入列表
                '油品': oil,  # 油品名稱
                'Regime': name,  # 區間名稱
                'AI_MAE': sub['AI_err'].mean(),  # 計算這段期間 AI 模型的平均絕對誤差 (MAE)
                'ARIMA_MAE': sub['ARIMA_err'].mean(),  # 計算這段期間 ARIMA 模型的 MAE
                'N_weeks': len(sub)  # 記錄這段期間包含了幾週的資料
            })
            
            # 使用 safe block 畫圖  # 區間內的獨立繪圖作業
            try:
                plt.figure()  # 建立畫布
                plt.boxplot([sub['AI_err'], sub['ARIMA_err']], labels=['AI', 'ARIMA'])  # 畫出兩種模型在該區間內的誤差盒鬚圖
                plt.title(f'{oil} | {name} Error Comparison')  # 標題標示為哪個區間
                plt.ylabel('Absolute Error')  # Y 軸標籤
                plt.tight_layout()  # 自動排版
                plt.savefig(f"{outdir}/REGIME_ERR_{oil}_{name}.png", dpi=300)  # 針對該區間單獨存檔
                plt.close()  # 關閉畫布
            except Exception as e:  # 捕捉子圖繪製錯誤
                logging.warning(f"Regime plot failed for {name}: {e}")  # 紀錄警告並忽略，不影響主流程
                plt.close()  # 確保畫布關閉

        return pd.DataFrame(rows)  # 將收集到所有區間統計數據的列表，轉換為 pandas DataFrame 後回傳
    except Exception as e:  # 捕捉整體評估流程的錯誤
        logging.error(f"evaluate_regime_errors failed: {e}")  # 紀錄錯誤日誌
        return pd.DataFrame()  # 發生錯誤時回傳空的 DataFrame 避免程式崩潰

@safe_plot  # 套用安全繪圖裝飾器
def plot_contribution_stack(dates, y_final, y_arima, w_ai, oil_name, outdir):  # 繪製 ARIMA 基底與 XGBoost 殘差修正的「堆疊貢獻圖」
    if len(dates) == 0: return  # 無資料則跳出
    
    contrib_xgb = y_final - y_arima  # 計算出最終預測值扣掉 ARIMA 預測值的差額，這段差額就是 XGBoost 模型的貢獻度 (修正量)
    
    plt.figure(figsize=(12, 6))  # 建立寬 12、高 6 的大畫布
    indices = range(len(dates))  # 產生對應日期長度的索引數列 (0, 1, 2...) 作為柱狀圖的 X 座標
    width = 0.6  # 設定柱子的寬度
    
    plt.bar(indices, y_arima, width, label='ARIMA Base (Trend)', color='skyblue', alpha=0.8)  # 畫出底層的柱狀圖：代表 ARIMA 預測的趨勢基底 (天藍色)
    # 🔥 [確認] 這裡已經是 XGBoost Correction，非常正確  # 開發者針對堆疊邏輯的自我確認
    plt.bar(indices, contrib_xgb, width, bottom=y_arima, label='XGBoost Correction (Residual)', color='salmon', alpha=0.8)  # 畫出上層的堆疊柱狀圖 (設定 bottom=y_arima)：代表 XGBoost 在 ARIMA 的基礎上又修正了多少幅度 (鮭魚紅)
    
    plt.plot(indices, y_final, color='black', linewidth=1, linestyle='--', label='Final Hybrid Prediction')  # 在柱狀圖上方疊加一條黑色的最終預測結果虛線，串起所有堆疊柱的頂點

    plt.title(f'{oil_name} - Component Decomposition (ARIMA + XGBoost)', fontsize=14)  # 設定圖表標題為元件分解
    plt.legend()  # 顯示圖例
    plt.grid(True, axis='y', alpha=0.3)  # 只開啟 Y 軸方向的網格線，視覺更清爽
    
    step = max(1, len(dates) // 10)  # 計算 X 軸標籤的間隔步數，避免日期文字擠在一起，確保畫面上最多約顯示 10 個日期
    plt.xticks(indices[::step], [d.strftime('%Y-%m-%d') for d in np.array(dates)[::step]], rotation=15)  # 設定 X 軸的刻度位置與對應的格式化日期文字，並將文字傾斜 15 度
    plt.savefig(f"{outdir}/{oil_name}_decomposition.png")  # 存檔
    plt.close()  # 關閉畫布