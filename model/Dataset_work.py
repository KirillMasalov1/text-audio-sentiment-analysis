import pandas as pd

url = "hf://datasets/seara/ru_go_emotions/raw/train-00000-of-00001-86de8ef1d0ae28df.parquet"
# save_path = "Датасет"
# df = pd.read_parquet(url)
# df.to_csv(save_path, index=False, encoding='utf-8')

save_path = "Датасет"
df = pd.read_parquet(url)
# df = pd.read_csv(save_path)
df = df.drop(['id','author','subreddit','link_id','parent_id','created_utc','rater_id','example_very_unclear'], axis=1)
df.to_csv(save_path, index=False, encoding='utf-8')
print(df.info)
