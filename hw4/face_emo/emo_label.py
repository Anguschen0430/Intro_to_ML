import pandas as pd
import os
# 建立 emotion 到 label 的對應字典
emotion_to_label = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}

# 讀取輸入 CSV 檔案
df = pd.read_csv('ensemble_predictions.csv')  # 假設你的檔名為 input.csv

# 將 emotion 欄位轉換為對應的 label
df['filename'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])
df['emotion'] = df['emotion'].apply(lambda x: x.capitalize())
df['label'] = df['emotion'].map(emotion_to_label)

# 保留 'filename' 和 'label' 欄位
df = df[['filename', 'label']]

# 輸出到新的 CSV 檔案
df.to_csv('output.csv', index=False)

print("轉換完成！已將輸出儲存至 'output.csv'")
