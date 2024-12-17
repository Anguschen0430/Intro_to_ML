import os
import pandas as pd

path = '../rafdb_large/val'
csv_file_path = '../rafdb_large/val_labels.csv'

label_mapping = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}
image_data = []

# Make sure the name of the file is partition_iteration_emotion.jpg or .png
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  
        label_name = filename.split('_')[-1].split('.')[0]
        label_name = label_name.capitalize()
        label_value = label_mapping.get(label_name)
        if label_value is not None:  
            image_data.append([filename, label_value])

df = pd.DataFrame(image_data, columns=["ImageName", "Label"])



df.to_csv(csv_file_path, index=False, header=False)

print(f"CSV file created at: {csv_file_path}")