# 🛢️ SmartFuel: AI-Driven Fuel Price Decision System (AI 驅動國際油價與國內定價預測系統)

## 📖 專案簡介 (Project Overview)
本專案旨在透過機器學習與領域知識 (Domain Knowledge)，建構一個能精準預測「台灣中油次週汽柴油公告牌價」的決策支援系統。有別於傳統單一的時序預測，本專案獨創 **「AI 混合模型 + 智慧風控濾網 + 政策模擬引擎」** 三層架構。不僅預測國際原油走勢，更精準模擬台灣政府的調價法規，提供具備真實商業價值的決策建議。

> 🏆 **專案成就：** 於 AI 大數據與 LLM 模型應用人才養成班結訓時，榮獲班級評估 **第二名**！(預測準確率達 61.2%)

## ✨ 核心技術亮點 (Key Innovations)
本專案跳脫純學術的數據擬合，解決了金融時間序列高度雜訊的痛點：

* **Hybrid AI 架構：** 使用 ARIMA 捕捉長期線性趨勢，並運用 XGBoost (極端梯度提升樹) 學習非線性殘差與外部特徵干擾。
* **Smart Filter v29 (獨創智慧濾網)：** 內建動態分級制度 (Lock Level)、動能鎖 (Momentum Lock) 與爆量救援機制，有效過濾誘多/誘空雜訊，將做多勝率提升至實戰等級。
* **政策模擬引擎 (Policy Engine)：** 程式內建台灣中油「亞鄰最低價天花板」與「油價平穩機制」雙重法規演算法，將 AI 預測還原為合規的最終零售價。
* **NLP 情緒分析：** 導入 ProsusAI/FinBERT 模型，即時抓取國際能源新聞進行語意運算，將地緣政治與市場恐慌情緒量化為特徵。

## 📐 系統架構與 API 流程 (System Architecture)

本專案整合前端介面、預測 API 與資料庫，完整模擬真實環境下的系統交握流程。

```mermaid
sequenceDiagram
    actor VIP會員
    participant 前端網頁
    participant 預測API
    participant 歷史資料庫

    VIP會員->>前端網頁: 1. 輸入預測日期與參數
    前端網頁->>預測API: 2. 發送 POST 請求 (RESTful API)
    預測API->>歷史資料庫: 3. 撈取原油收盤價與總體經濟特徵
    歷史資料庫-->>預測API: 4. 回傳歷史特徵資料
    預測API->>預測API: 5. 執行 ARIMA+XGBoost 混合運算
    預測API->>預測API: 6. 經過 Policy Engine (法規平穩機制) 轉換
    預測API-->>前端網頁: 7. 回傳最終預測牌價 (JSON)
    前端網頁-->>VIP會員: 8. 渲染視覺化預測圖表

    ## 🛠️ 技術堆疊 (Tech Stack)
    * **資料工程與爬蟲：** Python, Pandas, Numpy, Requests, yfinance, Feedparser
    * **機器學習與 NLP：** Scikit-learn, XGBoost, Statsmodels (ARIMA), Transformers (Hugging Face / FinBERT)
    * **視覺化與解釋性：** Matplotlib, Seaborn, SHAP (特徵貢獻度分析)
    * **自動化部署：** LINE Messaging API (自動廣播推播系統), Schedule 排程自動化
    
    ## 📊 資料來源與特徵工程 (Data & Feature Engineering)
    * **國際金融數據：** 布蘭特原油 (Brent)、USD/TWD、JPY、KRW 匯率、VIX 恐慌指數。
    * **技術指標擴充：** MA 乖離率、MACD 動能、RSI、ATR、布林通道帶寬、量價相對指標。
    * **防呆與極端值處理：** 動態歷史波動門檻計算、極端事件偵測與標記 (如 COVID-19、烏俄戰爭)。
    
    ## 📈 成果展示 (Results & Visualizations)
    
    <img width="864" height="432" alt="92_timeseries" src="https://github.com/user-attachments/assets/2eb00d52-dc9c-4aea-b175-48170e321a0b" />
    *▲ 圖 1：ARIMA+XGBoost 混合預測與實際油價變動之擬合狀況 (含 95% 信賴區間)*
    
    <br>
    
    <img width="3447" height="1699" alt="ASIA_CEILING_92" src="https://github.com/user-attachments/assets/6102a583-a7a6-4e20-a6f0-b429a93beed4" />
    *▲ 圖 2：政策引擎精準捕捉「亞鄰競爭國最低價」對漲幅的壓抑效應 (橘色區域)*
    
    ### 🎯 模型評估指標
    
    * **方向預測準確率 (Direction Accuracy): 61.1%** (突破傳統統計模型瓶頸)
      <br><img width="600" alt="CONFUSION_92" src="https://github.com/user-attachments/assets/9c9e09aa-cf9f-45ea-9421-657bc52a0b3b" />
    
    * **系統化誤差 (Bias Drift):** 透過動態權重與殘差修正，長期累積誤差趨近於零。
      <br><img width="800" alt="CUM_ERROR_92" src="https://github.com/user-attachments/assets/281ed13a-8c0d-447a-87b0-9be1de56491d" />
    
    ## 🚀 如何執行此專案 (Quick Start)
    
    **1. Clone 專案到本地端**
    ```bash
    git clone https://github.com/t2wqr27-tech/oil-price-prediction.git
    cd oil-price-prediction
