import requests

url = "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS"

headers = {
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(url, headers=headers)

print("STATUS:", r.status_code)
print("FIRST 300 CHARS:\n")
print(r.text[:300])