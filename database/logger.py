# -*- coding: utf-8 -*-
import os
import csv
from datetime import datetime
from config import LOG_PATH

def log_unified(source, name, status, detail):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if is_new: writer.writerow(['时间', '来源', '姓名', '状态', '详情'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), source, name, status, detail])

def get_daily_statistics():
    if not os.path.exists(LOG_PATH): 
        return None
    try:
        import pandas as pd
        df = pd.read_csv(LOG_PATH, encoding='utf-8-sig')
        today_str = datetime.now().strftime('%Y-%m-%d')
        df['时间'] = pd.to_datetime(df['时间'])
        today_df = df[df['时间'].dt.strftime('%Y-%m-%d') == today_str]

        if today_df.empty: 
            return 0, 0, 0, 0, []
        
        total_visits = len(today_df)
        blacklist_cnt = len(today_df[today_df['状态'].str.contains('黑名单')])
        whitelist_cnt = len(today_df[today_df['状态'].str.contains('白名单')])
        top_visitors = today_df['姓名'].value_counts().head(3).to_dict()
        return total_visits, blacklist_cnt, whitelist_cnt, top_visitors
    except Exception as e:
        print(f"统计出错: {e}")
        return None