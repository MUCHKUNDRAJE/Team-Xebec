import json
import pandas as pd
from donki_api import DonkiAPI


data = []
with open("donki_data2.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
df.to_csv("donki_data.csv", index=False)
print("Data has been written to donki_data.csv")

class JsonToCsvConverter:
    @staticmethod
    def convert():
        donki_api_data = DonkiAPI('PuBnZa7c03I3c608IxEYWmOmybq15LswZARx8oyy').fetch_cme_data('2011-04-01', '2025-12-31')
        df = pd.DataFrame(donki_api_data)
        df.to_csv("donki_data.csv", index=False)