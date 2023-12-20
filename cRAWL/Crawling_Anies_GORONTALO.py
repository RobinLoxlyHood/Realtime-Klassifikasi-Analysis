import subprocess
import os
from datetime import timedelta
import pandas as pd
from datetime import date
import warnings
warnings.filterwarnings("ignore")

path = r"C:\Skripsi Rizki\SKRIPSI"
os.chdir(path)

today = "2023-06-18"  # date.today()
yesterday = "2023-06-17"  # today - timedelta(days=1)
tokoh = "Anies Baswedan"
lokasi = "GORONTALO"
radius = 562
radius = str(radius / 2) + "km"

filename = f'{lokasi}_ANIES.csv'
twitter_search = f"{tokoh} near:{lokasi} within:{radius} lang:id until:{today} since:{yesterday}"
limit = 5
npx_command = f'npx tweet-harvest@latest -o "{filename}" -s "{twitter_search}" -l {limit} --token ""'
process = subprocess.run(npx_command, 
                         shell=True, 
                         input="692987fd316311d07bcb56080209aee1e85be0a7\n", 
                         capture_output=True, 
                         text=True,
                         encoding="utf-8")  # Explicitly set the encoding to 'utf-8'

print(process.stdout)
print(process.returncode)


direktori = "tweets-data"  # Ganti dengan path direktori yang ingin diperiksa
path_file = os.path.join(direktori, filename)

if os.path.exists(path_file):
    tweets_df = pd.read_csv(path_file, delimiter=";")
    if len(tweets_df) != 0:
        tweets_df_rename = tweets_df.rename(columns={"created_at": "date", "username": "username", "full_text": "tweet"})
        tweets_df_fix = tweets_df_rename[["date", "username", "tweet"]]
        tweets_df_fix["location"] = lokasi
        tweets_df_fix['date'] = pd.to_datetime(tweets_df_fix['date'], format='%a %b %d %H:%M:%S +0000 %Y')
        tweets_df_fix['date'] = tweets_df_fix['date'].dt.strftime('%Y-%m-%d')
        tweets_df_fix.to_csv(f"Data/Hasil Crawling/{lokasi}_ANIES.csv", index=False)
        os.remove(path_file)
        print(f"Crawling Tokoh {tokoh} pada {lokasi} Telah Selesai")
    else:
        print(f"Hasil Crawling Tokoh {tokoh} pada {lokasi} Tidak Ditemukan")
else:
    print(f"Proses Crawling Tokoh {tokoh} pada {lokasi}.")

print("Proses Crawling Twitter Tokoh Anies Baswedan Selesai")