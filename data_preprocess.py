import re
import pandas as pd
import numpy as np

# df = pd.read_csv('train.csv')
df = pd.read_csv('test.csv')


df = df.drop(['mid','has_url','comments','time', 'pics','likes','reposts'], axis=1)
print(df.head())

# Remove urls, non-alphanumeric chars,emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
            u'(\U0001F1F2\U0001F1F4)|'       # Macau flag
            u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
            u'([\U0001F600-\U0001F64F])'     # emoticons
            "+", flags=re.UNICODE)
    
    return emoji_pattern.sub('', text)


def preprocessing(text):
    text = remove_emojis(text)
    text = re.sub("[a-zA-Z0-9]+", '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    return text


df_clean = df.dropna()
print(df[df.isnull().any(axis=1)])
print(df_clean.info())


df_clean.loc[:, "text"] = df_clean['text'].apply(preprocessing)
print(df_clean[df_clean.isnull().any(axis=1)])
df_final = df_clean[df_clean['text'].str.match(r'^\s*$')]
df_clean = df_clean.replace(to_replace=r'^\s*$', value=np.nan, regex=True)
print(df_clean.iloc[10:20])
print(df_clean[df_clean.isnull().any(axis=1)])

df_final = df_clean.dropna()
df_final.loc[df_final['label'] == 'rumor', 'label'] = 1
df_final.loc[df_final['label'] == 'non-rumor', 'label'] = 0

# print(df_final.info())
# df_final.to_csv("data\\train.csv", index=False)
df_clean.to_csv("data\\test.csv", index=False)

