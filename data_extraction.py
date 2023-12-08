import os
import json
import csv
import pandas as pd

original_folder = 'social_network_rumor\\train\\original-microblog'
rumor_folder = 'social_network_rumor\\train\\rumor-repost'
non_rumor_folder = 'social_network_rumor\\train\\non-rumor-repost'

# Test
original_folder = 'social_network_rumor\\test\\original-microblog'

def extract_ids(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_name, file_extension = os.path.splitext(file)
            file_name = file_name.split('_')[0]
            file_names.append(file_name)

    return file_names

rumor_ids = extract_ids(rumor_folder)
non_rumor_ids = extract_ids(non_rumor_folder)
print('Number of rumor posts', rumor_ids[:5])
print('Number of non-rumor posts', len(non_rumor_ids))

def read_original_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            extracted_data = {}
            file_name = filename.split('.')[0]
            extracted_data['id'] = file_name
            keys = ['mid', 'text', 'has_url', 'comments', 'time', 'pics', 'likes', 'reposts']
            for key in keys:
                extracted_data[key] = json_data[key]

            data.append(extracted_data)
    return data

original_data = read_original_data(original_folder)
# print(original_data[:5])

original_df = pd.DataFrame(original_data)
# original_df.to_csv('original.csv', index=False)
original_df.to_csv('test.csv', index=False)

df = pd.read_csv('original.csv')

def add_rumor_label(row):
    if row['mid'] in rumor_ids:
        return 'rumor'
    elif row['mid'] in non_rumor_ids:
        return 'non-rumor'
    else:
        return 'unknown'

df['label'] = df.apply(add_rumor_label, axis=1)
df.to_csv('train.csv', index=False)
df.to_csv('test.csv', index=False)