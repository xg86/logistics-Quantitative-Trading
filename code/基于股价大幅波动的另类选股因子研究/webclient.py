import httpx
import json
import numpy as np
import pandas as pd
from datetime import datetime
url = 'http://localhost:8888/curve'


def get_ylds(ref_date: datetime, mat_date: datetime, code: str, dates: []):
        category = code[2:4]
        delta_mat_dates = mat_date - (dates - ref_date)
        dates_list = []
        for mat_date in delta_mat_dates.array:
                dates_list.append(mat_date.strftime('%Y-%m-%d'))
        #dates_str=json.dumps(delta_mat_dates.array,sort_keys=True, default=str)
        data = {"ref_date": str(ref_date),
                "curve_category": category,
                "mat_dates": dates_list
                }
        response = httpx.post(url, json=data)
        return response.json()['yeilds']

#print(get_ylds('2023-09-22','00', ['2024-06-17 00:00:00', '2024-06-16 00:00:00', '2024-06-15 00:00:00']))