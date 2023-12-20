import pandas as pd
import os
import datetime
import subprocess

# Gunakan Twitter search untuk mencari tweet yang di-favoritkan minimal 10000 kali dan berbahasa Indonesia
twitter_search = "Ganjar Pranowo near:Aceh within:25mi lang:id until:2023-04-18 since:2023-04-17" 

# Tentukan nama file dengan format "<kueri pencarian>_<tanggal saat ini>.json"
filename = f"{twitter_search.replace(' ', '_').replace(':', '-')}_{datetime.date.today().strftime('%Y-%m-%d')}.json"

# Ambil tweet yang cocok dengan kueri pencarian dan simpan dalam file JSON dengan nama file yang telah ditentukan
run_command= f"snscrape --jsonl twitter-search '{twitter_search}'> {filename}"
# Jalankan perintah NPX di CMD dan berikan input
process = subprocess.run(npx_command, 
                         shell=True, 
                         input="692987fd316311d07bcb56080209aee1e85be0a7\n", 
                         capture_output=True, 
                         text=True)

print(process.stdout)
print(process.returncode)

# Membaca file JSON hasil dari perintah CLI sebelumnya dan membuat dataframe pandas
tweets_df = pd.read_json(filename, lines=True) 

NAMA_FILE_CSV = 'Ganjar Pranowo.csv'

# Membuat kamus untuk mengganti nama kolom
new_columns = {
    'conversationId': 'Conv. ID',
    'url': 'URL',
    'date': 'Date',
    'rawContent': 'Tweet',
    'id': 'ID',
    'replyCount': 'Replies',
    'retweetCount': 'Retweets',
    'likeCount': 'Likes',
    'quoteCount': 'Quotes',
    'bookmarkCount': 'Bookmarks',
    'lang': 'Language',
    'links': 'Links',
    'media': 'Media',
    'retweetedTweet': 'Retweeted Tweet',
    'username': 'Username'
}

if len(tweets_df) == 0:
    print('Pencarian tidak ditemukan coba ganti keyword lain, keywordmu: ', twitter_search)
    exit()
else:
  # Memilih kolom yang akan digunakan dan mengganti nama kolom menggunakan kamus yang telah dibuat
  tweets_df = tweets_df.loc[:, ['url', 'date', 'rawContent', 'id',
                            'replyCount', 'retweetCount', 'likeCount', 'quoteCount',
                            'conversationId', 'lang', 'links',
                            'media', 'retweetedTweet', 'bookmarkCount', 'username']]
  tweets_df = tweets_df.rename(columns=new_columns)

  # Ekstrak fullUrl dari kolom media dan url dari kolom links
  tweets_df['Media'] = tweets_df['Media'].apply(lambda x: x[0]['fullUrl'] if isinstance(x, list) and x and isinstance(x[0], dict) and 'fullUrl' in x[0] else None)
  tweets_df['Links'] = tweets_df['Links'].apply(lambda x: x[0]['url'] if isinstance(x, list) and x and isinstance(x[0], dict) and 'url' in x[0] else None)

  # Simpan ke csv
  tweets_df.to_csv(NAMA_FILE_CSV, index=False)