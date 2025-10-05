import requests
import json

api_key = 'PuBnZa7c03I3c608IxEYWmOmybq15LswZARx8oyy'
start_date = '2011-04-01'
end_date = '2025-12-31'
# response = requests.get(f'https://api.nasa.gov/DONKI/CMEAnalysis?startDate={start_date}&endDate={end_date}&api_key={api_key}')
# # print(response.json())

# with open("donki_data2.json", "w") as file:
#     json.dump(response.json(), file, indent=2)

class DonkiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.nasa.gov/DONKI/CMEAnalysis'

    def fetch_cme_data(self, start_date, end_date):
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'api_key': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()