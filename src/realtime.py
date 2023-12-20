import subprocess

## Crawling

# Definisikan tanggal yang diinginkan
start_date = "2023-07-01"
end_date = "2023-07-02"

# Crawl Data

filename = 'kota_tegal.csv'
search_keyword = f'Kota Tegal until:{end_date} since:{start_date}'
limit = 1000

npx_command = f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword}" -l {limit} --token ""'
# Jalankan perintah NPX di CMD dan berikan input
process = subprocess.run(npx_command, 
                         shell=True, 
                         input="692987fd316311d07bcb56080209aee1e85be0a7\n", 
                         capture_output=True, 
                         text=True,
                         encoding="utf-8")  # Explicitly set the encoding to 'utf-8'

print(process.stdout)
print(process.returncode)
