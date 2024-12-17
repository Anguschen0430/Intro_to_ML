import pandas as pd
import numpy as np

def ensemble_predictions(csv_files, weights=None):
    """
    Ensemble multiple model predictions from CSV files using weighted averaging.
    
    Parameters:
    csv_files (list): List of CSV file paths
    weights (list): List of weights for each model (optional)
    
    Returns:
    pandas.DataFrame: DataFrame with ensemble predictions
    """
    # 如果沒有提供權重，則使用平均權重
    if weights is None:
        weights = [1/len(csv_files)] * len(csv_files)
    
    # 確保權重和檔案數量相符
    if len(weights) != len(csv_files):
        raise ValueError("Number of weights must match number of CSV files")
    
    # 確保權重總和為1
    weights = np.array(weights) / sum(weights)
    
    # 讀取所有CSV文件
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.sort_values('filename').reset_index(drop=True)
        dfs.append(df)
    
    # 驗證所有文件的filename是否一致
    for i in range(1, len(dfs)):
        if not all(dfs[0]['filename'] == dfs[i]['filename']):
            raise ValueError(f"The filenames in {csv_files[0]} and {csv_files[i]} do not match.")
    
    # 定義概率列
    prob_columns = ['Angry_prob', 'Disgust_prob', 'Fear_prob', 'Happy_prob', 
                    'Neutral_prob', 'Sad_prob', 'Surprise_prob']
    
    # 創建結果DataFrame
    df_ensemble = dfs[0][['filename']].copy()
    
    # 計算加權平均
    for col in prob_columns:
        df_ensemble[col] = sum(df[col] * weight for df, weight in zip(dfs, weights))
    
    # 根據加權平均結果選擇最終標籤
    df_ensemble['label'] = df_ensemble[prob_columns].idxmax(axis=1).apply(lambda x: prob_columns.index(x))
    
    # 映射情緒標籤
    emotion_map = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprise'
    }
    df_ensemble['emotion'] = df_ensemble['label'].map(emotion_map)
    
    return df_ensemble

# 使用範例
if __name__ == "__main__":
    # 設定輸入文件和權重
    csv_files = [
        'output_vit.csv',
        'output_eff.csv',
        'output_base.csv',
        'output_res.csv'
    ]
    
    weights = [0.7, 0.75,0.55 ,0.4]  # 對應每個模型的權重
    
    # 執行ensemble
    try:
        df_ensemble = ensemble_predictions(csv_files, weights)
        
        # 保存結果
        output_file = 'ensemble_predictions.csv'
        df_ensemble.to_csv(output_file, index=False)
        print(f"Ensemble completed. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")