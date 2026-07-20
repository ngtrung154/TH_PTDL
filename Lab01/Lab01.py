#!/usr/bin/env python
# coding: utf-8

# # BÀI LÀM LAB 1
# ## Môn Nhập môn Phân tích dữ liệu và Học sâu
# 
# Thao tác, xử lý dữ liệu thiếu và tạo các biến mới cho dữ liệu tuyển sinh đại học.
# 
# > Notebook đã được tách riêng và có thể chạy độc lập.

# In[1]:


from pathlib import Path
import csv
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

BASE_DIR = Path.cwd()

def locate_file(filename):
    candidates = [BASE_DIR / filename, Path("/mnt/data") / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Không tìm thấy {filename}. Hãy đặt tệp cùng thư mục với notebook."
    )

DATA_SCORE = locate_file("dulieuxettuyendaihoc.csv")
print("Thư mục làm việc:", BASE_DIR)
print("Dữ liệu Lab 1:", DATA_SCORE)


# # LAB 1 — BÀI THỰC HÀNH THAO TÁC DỮ LIỆU
# 
# Dữ liệu gồm thông tin điểm trung bình các môn qua nhiều năm học, giới tính, dân tộc, khu vực, khối thi và ba điểm thi đại học.

# ## Câu 1–2. Phân loại dữ liệu và thang đo
# 
# - **Định tính định danh:** `GT`, `DT`, `KV`, `KT`; `STT` chỉ là mã nhận diện.
# - **Định lượng:** các điểm môn học, `DH1`, `DH2`, `DH3`.
# - **Biến tạo thêm:** `TBM*`, `US_TBM*` là định lượng; `XL*` là định tính thứ bậc; `KQXT` là định tính nhị phân.

# In[2]:


data_dictionary = pd.DataFrame([
    ['STT', 'Mã nhận diện', 'Định tính', 'Định danh'],
    ['T1..N6', 'Điểm các môn qua các năm', 'Định lượng liên tục', 'Khoảng'],
    ['GT', 'Giới tính', 'Định tính', 'Định danh'],
    ['DT', 'Dân tộc', 'Định tính', 'Định danh'],
    ['KV', 'Khu vực', 'Định tính', 'Định danh'],
    ['KT', 'Khối thi', 'Định tính', 'Định danh'],
    ['DH1, DH2, DH3', 'Điểm thi đại học', 'Định lượng liên tục', 'Khoảng'],
    ['TBM1, TBM2, TBM3', 'Điểm trung bình môn', 'Định lượng liên tục', 'Khoảng'],
    ['XL1, XL2, XL3', 'Xếp loại học lực', 'Định tính', 'Thứ bậc'],
    ['US_TBM1..3', 'Điểm quy đổi thang 4', 'Định lượng liên tục', 'Khoảng'],
    ['KQXT', 'Kết quả xét tuyển', 'Định tính nhị phân', 'Định danh'],
], columns=['Biến/nhóm biến', 'Ý nghĩa', 'Loại dữ liệu', 'Thang đo'])
display(data_dictionary)


# ## Câu 3. Tải dữ liệu và in 10 dòng đầu, 10 dòng cuối

# In[3]:


df_raw = pd.read_csv(DATA_SCORE)
print('Kích thước dữ liệu:', df_raw.shape)
print('\n10 dòng đầu:')
display(df_raw.head(10))
print('\n10 dòng cuối:')
display(df_raw.tail(10))


# ## Câu 4. Thống kê và xử lý dữ liệu thiếu của biến dân tộc `DT`

# In[4]:


print('Các giá trị riêng biệt của DT:', df_raw['DT'].unique())
frequency_dt_before = df_raw['DT'].value_counts(dropna=False).rename('Tan_so').to_frame()
frequency_dt_before['Tan_suat'] = frequency_dt_before['Tan_so'] / len(df_raw)
display(frequency_dt_before)
print('Số lượng thiếu DT:', int(df_raw['DT'].isna().sum()))

# Theo yêu cầu đề bài, DT thiếu được thay bằng 0.
df = df_raw.copy()
df['DT'] = df['DT'].fillna(0).astype(int)
print('Số lượng thiếu DT sau xử lý:', int(df['DT'].isna().sum()))
display(df['DT'].value_counts(dropna=False).sort_index().rename('Tan_so').to_frame())


# Trong bộ dữ liệu này, phần lớn `DT` bị trống. Theo đúng yêu cầu, bài làm thay bằng `0`. Khi lọc “dân tộc Kinh” ở Lab 2, bài làm sử dụng `DT = 0`, phù hợp với cách mã hóa thường dùng trong bộ dữ liệu thực hành này.

# ## Câu 5–6. Thống kê và xử lý dữ liệu thiếu ở các biến điểm số bằng Mean

# In[5]:


score_cols = [c for c in df.columns if re.fullmatch(r'[TLHSVXDN][1-6]', c)] + ['DH1', 'DH2', 'DH3']
missing_scores_before = df[score_cols].isna().sum().rename('So_luong_thieu').to_frame()
display(missing_scores_before[missing_scores_before['So_luong_thieu'] > 0])

print('T1 - số lượng thiếu:', int(df['T1'].isna().sum()))
print('T1 - mean dùng để thay thế nếu có thiếu:', round(df['T1'].mean(), 4))

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())

print('Tổng số ô còn thiếu trong các biến điểm:', int(df[score_cols].isna().sum().sum()))


# ## Câu 7. Tạo các biến `TBM1`, `TBM2`, `TBM3`

# In[6]:


def calculate_tbm(data, year):
    return (
        data[f'T{year}'] * 2 + data[f'L{year}'] + data[f'H{year}'] +
        data[f'S{year}'] + data[f'V{year}'] * 2 + data[f'X{year}'] +
        data[f'D{year}'] + data[f'N{year}']
    ) / 10

# Lớp 10 dùng hậu tố 1; lớp 11 dùng hậu tố 2; lớp 12 dùng hậu tố 6 theo tài liệu.
df['TBM1'] = calculate_tbm(df, 1)
df['TBM2'] = calculate_tbm(df, 2)
df['TBM3'] = calculate_tbm(df, 6)
display(df[['STT', 'TBM1', 'TBM2', 'TBM3']].head(10))


# ## Câu 8. Tạo các biến xếp loại `XL1`, `XL2`, `XL3`

# In[7]:


xl_bins = [-np.inf, 5.0, 6.5, 8.0, 9.0, np.inf]
xl_labels = ['Y', 'TB', 'K', 'G', 'XS']
for i in range(1, 4):
    df[f'XL{i}'] = pd.cut(df[f'TBM{i}'], bins=xl_bins, labels=xl_labels, right=False)

display(df[['STT', 'TBM1', 'XL1', 'TBM2', 'XL2', 'TBM3', 'XL3']].head(10))


# ## Câu 9. Quy đổi điểm trung bình từ thang 10 sang thang 4 bằng Min–Max

# In[8]:


# Min-Max từ miền [0, 10] sang [0, 4]: x_new = x * 4 / 10.
for i in range(1, 4):
    df[f'US_TBM{i}'] = df[f'TBM{i}'] * 4 / 10

display(df[['STT', 'TBM1', 'US_TBM1', 'TBM2', 'US_TBM2', 'TBM3', 'US_TBM3']].head(10))


# ## Câu 10. Tạo biến kết quả xét tuyển `KQXT`

# In[9]:


def admission_result(row):
    if row['KT'] in ['A', 'A1']:
        admission_score = (row['DH1'] * 2 + row['DH2'] + row['DH3']) / 4
    elif row['KT'] == 'B':
        admission_score = (row['DH1'] + row['DH2'] * 2 + row['DH3']) / 4
    else:
        admission_score = (row['DH1'] + row['DH2'] + row['DH3']) / 3
    return int(admission_score >= 5.0)

df['KQXT'] = df.apply(admission_result, axis=1).astype(int)
display(df[['STT', 'KT', 'DH1', 'DH2', 'DH3', 'KQXT']].head(15))
print('Phân bố kết quả xét tuyển:')
display(df['KQXT'].value_counts().sort_index().rename(index={0: 'Rot', 1: 'Dau'}).rename('So_luong').to_frame())


# ## Câu 11. Lưu dữ liệu đã xử lý

# In[10]:


processed_path = BASE_DIR / 'processed_dulieuxettuyendaihoc.csv'
df.to_csv(processed_path, index=False)
print('Đã lưu:', processed_path)
print('Kích thước dữ liệu sau xử lý:', df.shape)
print('Tổng dữ liệu thiếu còn lại:', int(df.isna().sum().sum()))

