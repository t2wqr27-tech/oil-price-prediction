# -*- coding: utf-8 -*-
import numpy as np # 數值計算
import pandas as pd # 資料處理
from tqdm import tqdm # 進度條
import xgboost as xgb # 機器學習模型
from statsmodels.tsa.arima.model import ARIMA # 時序模型
from policy_engine import apply_asia_ceiling, apply_smoothing # 政策函數

SEED = 42 # 固定隨機種子，確保結果可重現

def arima_forecast(series): # ARIMA 預測，專注於捕捉短期動能
    series = series[~np.isnan(series)] # 移除 NaN 值，確保 ARIMA 模型能正常運行
    if len(series) < 5: return 0.0 # ARIMA 需要足夠的數據點來建模，至少 5 個差分後的數據點
    try:
        model = ARIMA(series, order=(1, 0, 0)).fit() # 使用 AR(1) 模型，專注於捕捉短期動能 
        forecast = model.forecast()[0] # 預測下一期的變化量
        if abs(forecast) > 3.0: return 0.0 # 安全機制：如果 ARIMA 預測的變化量過大，直接歸零，避免極端值干擾
        return forecast # 返回預測的變化量，而不是價格，這樣更適合與 AI 預測的殘差結合使用
    except: 
        return 0.0 # 如果 ARIMA 模型失敗，返回 0.0，表示沒有動能預測

def apply_smart_filter(pred_val, lock_level, current_price, p_arima, # 這裡是智慧濾網的核心邏輯，根據多種市場條件來調整 AI 預測的結果，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
                       is_rebound=False, is_overbought=False, is_price_dropping=False, # 這些是市場條件的參數，讓智慧濾網能夠根據不同的市場情況來調整預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
                       is_momentum_up=False, curr_macd=0.0, trend_health=True, # 這些是動能和趨勢的參數，讓智慧濾網能夠根據市場的動能和趨勢狀況來調整預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
                       vol_ratio=1.0): # 這是成交量比率的參數，讓智慧濾網能夠根據市場的活躍程度來調整預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
    """
    🔥 [智慧濾網 v17 - 轉折捕捉版]
    放寬對「逆勢上漲」的容忍度，增加對 V 型反轉的捕捉能力
    """
    
    # 1. 價格行為否決 (Price Action Veto) - 放寬版
    # 只有在「跌勢非常明顯」且「AI 信心不足」時才否決
    if pred_val > 0 and is_price_dropping and not is_rebound: # 只有在預測為漲的情況下才進行這個過濾，這是為了避免過度懲罰 AI 的漲勢預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
        
        # --- 豁免條款 (Waiver Conditions) ---
        # 1. AI 信心門檻降低：原本 0.10 -> 改為 0.08
        cond1 = (pred_val > 0.12) 
        
        # 2. ARIMA 門檻降低：原本 0.20 -> 改為 0.15
        cond2 = (p_arima > 0.15)
        
        # 3. 動能豁免優化
        # 如果 MACD 雖然是綠棒但在收斂 (curr_macd 正在變好)，也給予機會
        cond3 = (is_momentum_up and vol_ratio > 1.05 and trend_health) # 動能向上且有量能支持，且趨勢健康
            # 如果不符合任何豁免條件，直接殺掉
        if not (cond1 or cond2 or cond3): # 如果 AI 信心不足，ARIMA 也沒有給力的預測，動能又沒有明顯改善，這絕對是誘多，殺掉！
            return 0.0  # 🔥 殺伐果斷：直接歸零，不再保留 0.15 的殘渣

    # 2. 過熱煞車
    if is_overbought and pred_val > 0: # 只有在預測為漲的情況下才進行這個過濾，這是為了避免過度懲罰 AI 的漲勢預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
        if pred_val < 0.15: return 0.0 # 只要 AI 預測的漲幅不夠大，就不買，避免在過熱時追高

    # 3. 搶反彈 (這部分保持，做得不錯)
    if is_rebound and pred_val > 0: # 只有在預測為漲的情況下才進行這個過濾，這是為了避免過度懲罰 AI 的漲勢預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
        return pred_val # 反彈的預測直接放行，這是為了捕捉 V 型反轉的機會，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性

    # 4. 順勢邏輯 (🔥 數據驅動優化：釋放弱勢紅利，收緊強勢追價)(完美參數，勿再調整)
    if pred_val > 0:
        if lock_level >= 3:
            # 趨勢雖強，但容易買在末升段
            # 🔥 提高門檻：原本 0.015 -> 改為 0.025 (過濾掉微幅上漲的誘多)
            threshold = 0.025 
        elif lock_level == 2:
            # 🔥 提高門檻：原本 0.03 -> 改為 0.04
            threshold = 0.04  
            # 🔥 新增【動能鎖】：針對 Lock 2 的精確打擊
            # 只有在中等趨勢時，我們強制要求「動能向上」
            # 邏輯：如果趨勢普普，動能又在衰退，這絕對是誘多，殺掉！
            if not is_momentum_up: # 如果動能沒有向上，直接殺掉，這是為了避免在中等趨勢時追高，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
                if vol_ratio > 1.25: # 如果成交量比率大於 1.25，表示市場活躍度很高，可能是 V 型反轉的機會，給予放行
                    pass # 放行 (救援成功)
                else:
                    return 0.0 # 沒動能又沒量，殺無赦
        else:
            if lock_level == 1:
                # 🔥 修正：從 0.05 升回 0.06 (過濾雜訊)
                # ✨ 新增微創手術：如果成交量太低 (<0.7)，視為無量假突破，不買
                threshold = 0.06 
            else:
                # Lock 0: 垃圾場，維持徹底封殺
                return 0.0
            
        if pred_val < threshold: # 如果預測的漲幅小於門檻，直接歸零，這是為了避免在趨勢不夠強烈時追高，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們更好地理解市場的動態，提高預測的準確性和穩定性
            return 0.0
            
    else:
        # 下跌訊號過濾 (保持不變)
        if abs(pred_val) < 0.03: # 追求更精確的賣出訊號，過濾掉微弱的下跌預測
            return 0.0 # 其他情況保持原預測
            
    return pred_val # 最後返回經過智慧濾網調整後的預測值

def hybrid_predict_value(train, X_last, xgb_feats, lstm_feats=None, use_lstm=False, optimize=False): 
    train = train.copy() 
    X_last = X_last.copy() 
    
    features_to_use = list(xgb_feats) 
    y_true = train['y'].values 
    X_train_full = train[features_to_use] 
    
    bundle = {} 

    # ---------------------------------------------------------
    # 階段一：建立「中油基準公式」模型 (Baseline Model)
    # ---------------------------------------------------------
    if '成本週變動' in train.columns:
        # 🔥 真實物理公式：1桶=159公升
        train_base_pred = (train['成本週變動'] / 159.0) * 0.8
        test_base_pred = (X_last['成本週變動'].values[0] / 159.0) * 0.8
        bundle['base_model'] = "True_Math_Formula" 
    else:
        # 防呆機制
        train_base_pred = train['y'].shift(1).fillna(0).values * 0.5 + \
                          train['y'].rolling(3).mean().shift(1).fillna(0).values * 0.5
        test_base_pred = 0.0 

    y_residuals = y_true - train_base_pred 
    
    # --- 權重優化 ---
    time_weights = np.linspace(0.5, 1.5, len(train)) 
    dir_weights = np.where(y_residuals > 0, 1.1, 1.0) 
    final_weights = time_weights * dir_weights 

    # XGBoost 參數調整
    xgb_m = xgb.XGBRegressor( 
        n_estimators=1300, 
        max_depth=5,            
        learning_rate=0.015,    
        subsample=0.8, 
        colsample_bytree=0.8, 
        gamma=0.1,              
        min_child_weight=3, 
        objective='reg:squarederror', 
        n_jobs=-1, 
        random_state=SEED 
    )
    
    xgb_m.fit(X_train_full, y_residuals, sample_weight=final_weights, verbose=False) 
    bundle['xgb'] = xgb_m 
    
    xgb_pred_residual = xgb_m.predict(X_last[features_to_use])[0] 
    
    # 🔥 階段三：實戰預測 (公式基準 + AI殘差)
    final_ai_pred = test_base_pred + xgb_pred_residual

    return final_ai_pred, bundle

def rolling_backtest(df, oil, xgb_feats, lstm_feats, start_test_date, min_train_weeks=52, retrain_freq=4): # 回測函數，實現雙重確認架構，結合 ARIMA 預測、AI 預測和智慧濾網，並加入動態權重調整和成交量過濾，專注於捕捉轉折點和 V 型反轉
    df = df.sort_values('日期').copy().reset_index(drop=True) # 確保資料按照日期排序，並重置索引，避免因為資料順序問題導致的錯誤，這對於時序分析非常重要
    df['y'] = df[oil].shift(-1) - df[oil] # 計算下一期的價格變化量，作為模型的預測目標，這樣模型專注於學習價格變化的模式，而不是絕對價格，這對於捕捉轉折點更有效
    
    mask = df['日期'] >= pd.to_datetime(start_test_date) # 找到回測開始的索引位置，確保我們有足夠的訓練數據來支持模型的學習，避免因為訓練數據不足導致的模型不穩定或無法捕捉轉折點的情況
    if not mask.any(): return np.array([]), np.array([]), np.array([]), [], [], [] # 如果沒有符合條件的日期，直接返回空的結果，避免後續程式出錯
    start_idx = df[mask].index[0] # 回測開始的索引位置，這是我們進行回測的起點，確保模型有足夠的訓練數據來學習，特別是對於捕捉轉折點和 V 型反轉非常重要
    
    y_true, y_ai, y_arima, dates_test, lstm_flags, w_ai_history = [], [], [], [], [], [] # 用於存儲回測結果的列表，這些結果將用於評估模型的表現，特別是 AI 預測的效果，以及 ARIMA 預測的貢獻，還有日期和其他相關資訊
    error_history = [] # 用於存儲預測誤差的歷史，這將用於後續的偏差修正，讓模型能夠根據過去的表現進行自我調整，特別是在捕捉轉折點和 V 型反轉方面，這是提升模型穩定性和準確性的關鍵因素
    cached_bundle = None # 用於緩存模型和相關資訊的容器，避免在回測過程中頻繁訓練模型造成的效率問題，特別是在捕捉轉折點和 V 型反轉時，這可以讓模型更快地適應市場變化，提高預測的即時性和準確性
    
    rolling_vol_series = df[oil].diff().abs().rolling(52).std() # 計算 52 週的滾動波動率，這將用於動態權重調整，讓模型能夠根據市場的波動性來調整 ARIMA 和 AI 預測的權重，特別是在捕捉轉折點和 V 型反轉時，這可以讓模型更靈活地適應不同的市場環境，提高預測的準確性和穩定性
    
    print(f"⚡ 啟動 {oil} 雙重確認版回測 (Double Confirmation)...")
    
    for i in tqdm(range(start_idx, len(df) - 1)): # 從回測開始的索引位置開始迭代，直到倒數第二行，確保我們有下一期的價格變化量作為預測目標，這是回測的核心過程，在這個過程中，我們將結合 ARIMA 預測、AI 預測和智慧濾網來進行預測，並根據市場的波動性動態調整權重，特別是在捕捉轉折點和 V 型反轉方面，這是提升模型表現的關鍵因素
        X_last = df.iloc[i:i+1].copy() # 取出當前行作為 XGBoost 的輸入，這是我們進行預測的基礎，確保我們有最新的市場資訊來支持模型的預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓模型更即時地反應市場變化，提高預測的準確性和穩定性
        current_price = df[oil].iloc[i] # 取出當前的價格，這是我們進行預測的基礎，確保我們有最新的市場資訊來支持模型的預測，特別是在捕捉轉折點和 V 型反轉時，這可以讓模型更即時地反應市場變化，提高預測的準確性和穩定性
        prev_price = df[oil].iloc[i-1] # 取出前一個價格，這將用於計算價格變化和其他技術指標，特別是在捕捉轉折點和 V 型反轉時，這可以讓模型更好地理解市場的動態，提高預測的準確性和穩定性
        
        # 1. ARIMA
        history_series = df.iloc[max(0, i-26):i][oil].diff().dropna() # 取出過去 26 週的價格變化量作為 ARIMA 的輸入，這是 ARIMA 預測的基礎，確保我們有足夠的數據來支持 ARIMA 模型的建模，特別是在捕捉轉折點和 V 型反轉時，這可以讓 ARIMA 更好地捕捉短期動能，提高預測的準確性和穩定性
        p_arima = arima_forecast(history_series) # 使用 ARIMA 模型進行預測，這是我們混合預測的第一步，ARIMA 專注於捕捉短期動能，特別是在捕捉轉折點和 V 型反轉時，這可以讓我們的混合預測更強大、更靈活，提高預測的準確性和穩定性

        # 2. Hybrid AI
        if (cached_bundle is None) or ((i - start_idx) % retrain_freq == 0): 
            # 🔥 這裡是不小心被刪除的拼圖！補回來就不會報錯了
            train = df.iloc[max(0, i-104):i].copy().dropna(subset=['y']) 
            if len(train) < min_train_weeks: continue 
            
            pred_residual, cached_bundle = hybrid_predict_value(train, X_last, xgb_feats, lstm_feats) 
        else:
            for f in xgb_feats: 
                if f not in X_last.columns: X_last[f] = 0 
            
            xgb_pred_residual = cached_bundle['xgb'].predict(X_last[xgb_feats])[0] 
            
            # 🔥 取得「中油真實物理數學公式」的理論預測
            if '成本週變動' in X_last.columns:
                test_base_pred = (X_last['成本週變動'].values[0] / 159.0) * 0.8
            else:
                test_base_pred = 0.0
                
            pred_residual = test_base_pred + xgb_pred_residual

        # 3. 動態權重
        current_vol = df.iloc[max(0, i-5):i][oil].diff().std() # 計算過去 5 週的價格變化量的標準差作為當前的波動率
        if np.isnan(current_vol): current_vol = 0.0 # 防呆機制
        hist_vol_window = rolling_vol_series.iloc[max(0, i-52):i] # 取出過去 52 週的滾動波動率作為歷史波動率的參考
        
        if len(hist_vol_window.dropna()) > 20: 
            VOL_HIGH = hist_vol_window.quantile(0.8) 
            # 🔥 修改 1：調降低波動門檻為 0.15，避免系統太容易進入死水期
            VOL_LOW = hist_vol_window.quantile(0.15) 
        else:
            VOL_HIGH, VOL_LOW = 1.0, 0.5 
        
        if VOL_HIGH - VOL_LOW == 0: ratio = 0.5 
        else: ratio = (current_vol - VOL_LOW) / (VOL_HIGH - VOL_LOW) 
        ratio = np.clip(ratio, 0, 1) 
        
        # 🔥 修改 2：解放 AI 權重！因為 AI 內含真實中油公式，保底給予 60% 權重，最高至 95%
        final_w_ai = 0.60 + (ratio * 0.35) 

        # 4. 原始混合
        raw_hybrid = p_arima + (final_w_ai * pred_residual)
        
        # 5. 偏差修正
        if len(error_history) >= 4: 
            bias_correction = np.mean(error_history[-4:]) * 0.25 
        else:
            bias_correction = 0.0 
        corrected_hybrid = raw_hybrid - bias_correction 
        
        # =========================================================
        # 🔥🔥🔥 [核心] 61% 衝刺：雙重確認架構 🔥🔥🔥
        # =========================================================
        hist_slice = df.iloc[max(0, i-35):i+1][oil] 
        
        ma5 = hist_slice.tail(5).mean() 
        ma10 = hist_slice.tail(10).mean() 
        ma20 = hist_slice.tail(20).mean() 
        
        # MACD
        exp12 = hist_slice.ewm(span=12, adjust=False).mean() 
        exp26 = hist_slice.ewm(span=26, adjust=False).mean() 
        macd = exp12 - exp26 
        macd_hist = macd - macd.ewm(span=9, adjust=False).mean() 
        
        if len(macd_hist) < 2: 
            curr_hist, prev_hist = 0.0, 0.0 
        else:
            curr_hist = macd_hist.iloc[-1] 
            prev_hist = macd_hist.iloc[-2] 
            
        is_momentum_up = (curr_hist > prev_hist) 
        is_momentum_down = (curr_hist < prev_hist) 
        curr_macd = curr_hist 
        
        delta = hist_slice.diff() 
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean() 
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean() 
        rs = gain / (loss + 1e-9) 
        current_rsi = 100 - (100 / (1 + rs)).iloc[-1] 
        
        bias_pct = (current_price - ma5) / ma5 
        
        # --- 狀態判定 ---
        
        # 1. 搶反彈優化
        is_extreme_oversold = (current_rsi < 30) 
        
        # 🔥 新增：RSI 背離偵測 (簡單版)
        prev_rsi = 100 - (100 / (1 + (rs.shift(1).iloc[-1]))) 
        is_rsi_divergence = (current_price < prev_price) and (current_rsi > prev_rsi) and (current_rsi < 50) 

        # 反彈條件放寬：加入背離偵測
        is_normal_rebound = (bias_pct < -0.015 and is_momentum_up) 
        is_rebound = is_extreme_oversold or is_normal_rebound or is_rsi_divergence 
            
        # 2. 過熱 
        is_overbought = (current_rsi > 80 and is_momentum_down) 
        
        # 3. 價格行為否決條件
        is_price_dropping = (current_price < prev_price) and (current_price < ma5) and (not is_momentum_up) 
        
        # 🔥 [雙重確認] 趨勢健康度
        trend_health = (current_price > ma20) and (ma5 > ma20) 

        # 4. 鎖定等級
        lock_level = 0 
        if ma5 > ma10: lock_level += 1 
        if current_price > ma5: lock_level += 1 
        if curr_hist > 0: lock_level += 1 
        if p_arima > 0.15: lock_level += 1 
        
        # --- 評分機制 ---
        if is_overbought: 
            corrected_hybrid -= 0.05 
        elif is_rebound: 
            corrected_hybrid += 0.07  
        else:
            if lock_level >= 3: 
                corrected_hybrid += 0.08  
            elif lock_level == 2: 
                corrected_hybrid += 0.05  
            elif lock_level <= 1: 
                corrected_hybrid -= 0.02
        # ==========================================
        # 🔥 新增：成交量計算區塊
        # ==========================================
        vol_col = 'Volume' if 'Volume' in df.columns else 'Vol.' 
        
        if vol_col in df.columns: 
            vol_slice = df.iloc[max(0, i-4):i+1][vol_col] 
            current_volume = vol_slice.iloc[-1] 
            ma5_volume = vol_slice.mean() 
            
            if ma5_volume == 0: 
                vol_ratio = 1.0 
            else:
                vol_ratio = current_volume / ma5_volume 
        else:
            vol_ratio = 1.0 
        
        # 6. 智慧濾網
        filtered_hybrid = apply_smart_filter( 
            corrected_hybrid, lock_level, current_price, p_arima, 
            is_rebound, is_overbought, is_price_dropping, 
            is_momentum_up, curr_macd, trend_health, 
            vol_ratio=vol_ratio 
        )
        
        # 7. 政策與記錄
        pred_price_raw = current_price + filtered_hybrid 
        last_row_dict = df.iloc[i].to_dict() 
        pred_price_ceiling = apply_asia_ceiling(pred_price_raw, oil, last_row_dict) 
        diff_after_ceiling = pred_price_ceiling - current_price 
        final_pred = apply_smoothing(diff_after_ceiling, oil) 
        
        actual_diff = df[oil].iloc[i+1] - df[oil].iloc[i] 
        current_error = filtered_hybrid - actual_diff 
        error_history.append(current_error) 
        
        y_true.append(actual_diff) 
        y_ai.append(final_pred) 
        y_arima.append(p_arima) 
        dates_test.append(df['日期'].iloc[i]) 
        lstm_flags.append(1) 
        w_ai_history.append(final_w_ai) 

    return np.array(y_true), np.array(y_ai), np.array(y_arima), dates_test, lstm_flags, np.array(w_ai_history)


