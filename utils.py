import numpy as np
from tqdm import tqdm
from collections import defaultdict


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()

    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])

    user_map = {user: idx for idx, user in enumerate(user_set)}
    item_map = {item: idx + 1 for idx, item in enumerate(item_set)}

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        # User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))
        User_res[user_map[user]] = [
            [item_map[item[0]], time_map[item[1]]]
            for item in items
        ]

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        # User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        User_res[user] = [
            [item[0], int(round((item[1] - time_min) / time_scale) + 1)]
            for item in items
        ]
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max), user_map, item_map


def data_partition(data):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    user_count = data['user_id'].value_counts().to_dict()
    item_count = data['item_id'].value_counts().to_dict()
    time_set = set()

    filtered_data = data[(data['user_id'].map(user_count) >= 5) & (data['item_id'].map(item_count) >= 5)]

    for _, row in filtered_data.iterrows():
        u = int(row['user_id'])
        i = int(row['item_id'])
        ts = row['timestamp']

        time_set.add(ts)
        User[u].append([i, ts])

    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum, user_map, item_map = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum, timenum], user_map, item_map


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int64)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen=50, time_span=256):
    data_train = dict()
    for user in tqdm(range(usernum), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int64)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


