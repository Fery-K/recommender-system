import torch
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from model import TPASASRec
from utils import computeRePos
from datetime import datetime, timezone
import re
import json


# IMPORT DATA
df = pd.read_csv('./data/interactions.csv')
db = pd.read_csv('./data/data_sample52.csv')
df_movies = pd.read_csv('./data/item_info.csv')

ts_min = df.groupby('user_id')['timestamp'].min().to_dict()
with open('./data/user_map.json', 'r') as umap, open('./data/item_map.json', 'r') as imap:
    user_dict = json.load(umap)
    item_dict = json.load(imap)
user_map = {int(key): int(value) for key, value in user_dict.items()}
item_map = {int(key): int(value) for key, value in item_dict.items()}
user_map_inv = {int(value): int(key) for key, value in user_dict.items()}
item_map_inv = {int(value): int(key) for key, value in item_dict.items()}


# INIT MODEL
usernum = 6040
itemnum = 3416
timenum = 89247793
batch_size = 128
maxlen = 50
hidden_units = 50
dropout_rate = 0.2
time_span = 2048
num_blocks = 2
num_heads = 1
num_filters = 50
window_size = 16

model = TPASASRec(usernum, itemnum, timenum, batch_size=batch_size, maxlen=maxlen,
                  hidden_units=hidden_units, dropout_rate=dropout_rate, time_span=time_span,
                  num_blocks=num_blocks, num_heads=num_heads, num_filters=num_filters, window_size=window_size, device='cpu')
model.load_state_dict(torch.load('./model_movielens1m/tpasasrec_v9_epoch20.pth'))
model.eval()

# PAGE CONFIG
def set_page_configuration():
    st.set_page_config(
        page_title=f'Recommender Demo',
        page_icon='🎥',
        layout='wide',
        initial_sidebar_state='expanded')


# BODY 1
def model_process():
    with st.sidebar:
        input_user = st.slider('Pilih ID Pengguna:', 1, usernum, 22)
        filter = st.segmented_control('', ['Tanpa Sampel Negatif', 'Dengan Sampel Negatif'], default='Dengan Sampel Negatif')

    with st.form('Input Form'):
        s49 = db[db['user_id'] == user_map[input_user]].iloc[48]
        s50 = db[db['user_id'] == user_map[input_user]].iloc[49]
        s51 = db[db['user_id'] == user_map[input_user]].iloc[50]

        input_seq = st.selectbox(
            'Pilih Interaksi (Judul Film)',
            df_movies['title'].values,
            df_movies[df_movies['movie_id'] == item_map_inv[(s50.loc['item_id'].item())]].index.item()
        )

        col1, col2 = st.columns(2)
        with col1:
            input_date = st.date_input(
                'Tanggal Interaksi (YYYY/MM/DD)',
                value=datetime.strptime(s50.loc['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).date(),
                min_value=datetime.strptime(s49.loc['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).date(),
            )
        with col2:
            input_time = st.text_input(
                'Waktu Interaksi (HH:MM:SS)',
                value=datetime.strptime(s50.loc['datetime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).time()
            )
        col3, col4 = st.columns([7, 1])
        with col3:
            input_genre = st.pills(
                'Genre',
                df_movies['genre'].str.split('|').explode().str.strip().str.lower().unique(),
                selection_mode='multi'
            )
        with col4:
            st.write('')
            st.write('')
            genre_inclusive = st.toggle('Inklusif')

        submitted = st.form_submit_button('Submit', use_container_width=True)

    tab1, tab2 = st.tabs(['Rekomendasi', 'Interaksi Historis'])
    with tab1:
        if submitted and input_seq:
            input_seq_id = item_map[df_movies[df_movies['title'] == input_seq]['movie_id'].item()]
            seqs = torch.tensor(db[db['user_id'] == user_map[input_user]].iloc[:49]['item_id'].tolist() + [input_seq_id], dtype=torch.long)

            input_ts = int(datetime.strptime(str(input_date) + ' ' + input_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp())
            ts_scaled = round((input_ts - ts_min[input_user]) / (s49.loc['timestamp'] - ts_min[input_user]) * (s49.loc['timescaled'] - 1)) + 1
            time_seq = torch.tensor(db[db['user_id'] == user_map[input_user]].iloc[:49]['timescaled'].tolist() + [ts_scaled])

            if input_genre:
                if not genre_inclusive:
                    pattern = '|'.join(rf'\b{re.escape(genre)}\b' for genre in input_genre)
                    filtered_genre = df_movies[df_movies['genre'].str.contains(pattern, case=False, na=False)]
                else:
                    filtered_genre = df_movies.copy()
                    for genre in input_genre:
                        pattern = rf'\b{re.escape(genre)}\b'
                        filtered_genre = filtered_genre[filtered_genre['genre'].str.contains(pattern, case=False, na=False)]

                filtered_ids = torch.tensor([filtered_genre['movie_id'].map(item_map).values])
            else:
                if filter == 'Dengan Sampel Negatif':
                    items = set(range(1, itemnum + 1)) - set(
                        db[db['user_id'] == user_map[input_user]].iloc[:50]['item_id']) - {s51.loc['item_id'].item()}
                    np.random.seed(22)
                    neg_sample = np.random.choice(list(items), size=100, replace=False).tolist()
                    neg_sample.append(s51.loc['item_id'].item())
                    filtered_ids = torch.tensor([neg_sample])
                else:
                    filtered_ids = None

            logits, pred = model.predict(
                user_map[input_user],
                seqs.unsqueeze(0),
                torch.tensor(computeRePos(torch.tensor(time_seq), 2048)).unsqueeze(0),
                k=10,
                item_ids=filtered_ids
            )

            pred_id = [k for k, v in item_map.items() if v in pred]
            df_recs = df_movies[df_movies['movie_id'].isin(pred_id)].reindex([i-1 for i in pred.squeeze(0).numpy()])

            recs_top = st.columns(5)
            for i in range(min(len(pred_id), 5)):
                with recs_top[i]:
                    st.markdown('''
                        <div style="text-align: center;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" fill="currentColor" class="bi bi-app" viewBox="0 0 16 16">
                            <path d="M11 2a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V5a3 3 0 0 1 3-3zM5 1a4 4 0 0 0-4 4v6a4 4 0 0 0 4 4h6a4 4 0 0 0 4-4V5a4 4 0 0 0-4-4z"/>
                            </svg>
                         </div>
                        ''', unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <h5>Rekomendasi {i+1}</h5>
                            <p>{df_recs['title'].iloc[i]}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            if len(pred_id) > 5:
                recs_bot = st.columns(5)
                for i in range(5, len(pred_id)):
                    with recs_bot[i-5]:
                        st.markdown('''
                            <div style="text-align: center;">
                                <svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" fill="currentColor" class="bi bi-app" viewBox="0 0 16 16">
                                <path d="M11 2a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V5a3 3 0 0 1 3-3zM5 1a4 4 0 0 0-4 4v6a4 4 0 0 0 4 4h6a4 4 0 0 0 4-4V5a4 4 0 0 0-4-4z"/>
                                </svg>
                             </div>
                            ''', unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <h5>Rekomendasi {i + 1}</h5>
                                <p>{df_recs['title'].iloc[i]}</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
    with tab2:
        user_history = db[db['user_id'] == user_map[input_user]][:51][::-1]
        user_history['movie_id'] = user_history['item_id'].map(item_map_inv)
        user_history = user_history.merge(df_movies, on='movie_id', how='left')

        for i, row in user_history.iterrows():
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 5px; border: 1px solid #eee; border-radius: 8px;">
                <strong>{51 - i}. {row['title']}</strong><br>
                <em>Genre:</em> {row['genre']}<br>
                <small><em>Timestamp:</em> {row['datetime']}</small>
            </div>
            """, unsafe_allow_html=True)

# NAVBAR
def navbar_menu():
    with st.sidebar:
        selected_navbar_menu = option_menu(
            menu_title='Menu',
            options=['Demo Model'],
            icons=['1-square'],
            menu_icon='cast',
            default_index=0,
            orientation='vertical',
            styles={'nav-link': {'--hover-color': '#FFB0B0',
                                 '--active-background-color': '#E64242'},
                    'nav-link-selected': {'background-color': '#E64242'},
                    }
        )

        st.markdown('<hr>', unsafe_allow_html=True)

        return selected_navbar_menu


# MAIN
def main():
    selected_navbar_menu = navbar_menu()

    if selected_navbar_menu == 'Demo Model':
        st.title('Input -> Output')
        model_process()


# RUN PROGRAM
if __name__ == '__main__':
    set_page_configuration()
    main()