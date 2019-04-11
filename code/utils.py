import multiprocessing as mp
import pandas as pd
import numpy as np
from scipy.stats import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score
feat_stat = [
    'diffmean',
    'diffmax',
    # 'norm',
    'dis_rate',   # waiting 优化
    'p1dis',
    'p2dis',
    'p1dis_rate',
    'p2dis_rate',
    # 'diff',
    # 'p1dis_rate2',
    # 'p2dis_rate2',
    'bin',
    'p1norm',
    'p2norm',
]

two_count_peak_cols = [
    'var_4',
    'var_6',
    'var_9',
    'var_12',
    'var_15',
    'var_23',
    'var_25',
    'var_27',
    'var_28',
    'var_34',
    'var_42',
    'var_43',
    'var_50',
    'var_53',
    'var_57',
    'var_59',
    'var_64',
    'var_66',
    'var_71',
    'var_79',
    'var_91',
    'var_93',
    'var_95',
    'var_98',
    'var_103',
    'var_105',
    'var_108',
    'var_111',
    'var_114',
    'var_125',
    'var_126',
    'var_130',
    'var_131',
    'var_132',
    'var_133',
    'var_144',
    'var_148',
    'var_156',
    'var_161',
    'var_162',
    'var_166',
    'var_169',
    'var_181',
    'var_189',
    'var_192',
    'var_195'
]

three_count_peak_cols = [
    'var_6',
    'var_25',
    'var_28',
    'var_34',
    'var_42',
    'var_43',
    'var_50',
    'var_53',
    'var_57',
    'var_59',
    'var_66',
    'var_71',
    'var_93',
    'var_95',
    'var_98',
    'var_105',
    'var_108',
    'var_111',
    'var_114',
    'var_125',
    'var_126',
    'var_130',
    'var_131',
    'var_133',
    'var_144',
    'var_148',
    'var_156',
    'var_161',
    'var_166',
    'var_169',
    'var_189',
    'var_197'
]

def threshold_search(y_true, y_prob):
    best_threshold = 0
    best_score = 0
    raw_score = accuracy_score(y_true,(y_prob>0.5).astype(int))
    for th in np.linspace(0.3, 0.7, 40, endpoint=False):
        score = accuracy_score(y_true,(y_prob>th).astype(int))
        if score > best_score:
            best_threshold = th
            best_score = score
    print(best_threshold)


def cal_freq_feat(train,test,all_data,cols):
    freq_cols = []
    for col in cols:
        map_count = all_data[col].value_counts(sort=False)
        map_count = map_count.sort_index()

        if True:#  smooth
            train[col + 'freq'] = train[col].map(map_count).clip(1,4).astype(np.int8)
            test[col + 'freq'] = test[col].map(map_count).clip(1,4).astype(np.int8)

        freq_cols.append(col + 'freq')
        unique_count = set(map_count.values)
        if True: # get feature
            mean_map = {c: np.mean(map_count[map_count == c].index) for c in unique_count}
            train[col + 'mean'] = train[col + 'freq'].map(mean_map)
            test[col + 'mean'] = test[col + 'freq'].map(mean_map)

            std_map = {c: np.std(map_count[map_count == c].index) for c in unique_count}
            train[col + 'std'] = train[col + 'freq'].map(std_map)
            test[col + 'std'] = test[col + 'freq'].map(std_map)

            max_map = {c: np.max(map_count[map_count == c].index.values) for c in unique_count}
            train[col + 'max'] = train[col + 'freq'].map(max_map)
            test[col + 'max'] = test[col + 'freq'].map(max_map)

            min_map = {c: np.min(map_count[map_count == c].index.values) for c in unique_count}
            train[col + 'min'] = train[col + 'freq'].map(min_map)
            test[col + 'min'] = test[col + 'freq'].map(min_map)

            median_map = {c: np.median(map_count[map_count == c].index.values) for c in unique_count}
            train[col + 'median'] = train[col + 'freq'].map(median_map)
            test[col + 'median'] = test[col + 'freq'].map(median_map)


            train[col + 'norm'] = (train[col]-train[col+'mean']) / train[col + 'std']
            test[col + 'norm'] = (test[col]-test[col+'mean']) / test[col + 'std']

            train[col + 'dis_rate'] = (train[col] - train[col + 'median'])/(train[col+'mean'] - train[col + 'median'])
            test[col + 'dis_rate'] = (test[col] - test[col + 'median'])/(test[col+'mean'] - test[col + 'median'])


        if True: # peak feature
            def cal_value(x):
                if len(x) == 2:
                    return (x[0] + x[1]) / 2
                return 0.25 * x[0] + 0.5 * x[1] + 0.25 * x[2]
            c2p1 = {}
            c2p2 = {}
            c2s1 = {}
            c2s2 = {}
            c2m = {}

            for c in unique_count:
                values = pd.Series(map_count[map_count == c].index)
                max_v = values[len(values) - 1]
                min_v = values[0]
                m = (min_v+max_v)/2
                value_distr = values.value_counts(bins=50, sort=False).rolling(window=3, min_periods=1, center=True)
                value_distr = value_distr.apply(cal_value).rolling(window=3, min_periods=1,center=True).apply(cal_value)
                value_distr.index = [v.mid for v in value_distr.index]
                c2p1[c] = value_distr[value_distr.index < m].idxmax()
                c2p2[c] = value_distr[value_distr.index > m].idxmax()
                c2s1[c] = values[values <= m].std()
                c2s2[c] = values[values > m].std()
                c2m[c] = m


            best_values = map_count[map_count == 1].index
            v2bin = {v: b for v, b in zip(best_values, pd.cut(best_values, bins=3, labels=False, ))}
            train[col + 'bin'] = train[col].apply(lambda x:v2bin[x] if x in v2bin else x)
            test[col + 'bin'] = test[col].apply(lambda x:v2bin[x] if x in v2bin else x)

            if col in two_count_peak_cols:
                best_values = map_count[map_count == 2].index
                v2bin = {v: b for v, b in zip(best_values, pd.cut(best_values, bins=3, labels=False, ))}
                train[col + 'bin2'] = train[col].apply(lambda x: v2bin[x] if x in v2bin else 4).astype(np.int8)
                test[col + 'bin2'] = test[col].apply(lambda x: v2bin[x] if x in v2bin else 4).astype(np.int8)

            if col in three_count_peak_cols:
                best_values = map_count[map_count == 3].index
                v2bin = {v: b for v, b in zip(best_values, pd.cut(best_values, bins=3, labels=False, ))}
                train[col + 'bin3'] = train[col].apply(lambda x: v2bin[x] if x in v2bin else 4).astype(np.int8)
                test[col + 'bin3'] = test[col].apply(lambda x: v2bin[x] if x in v2bin else 4).astype(np.int8)

            train[col + 'p1'] = train[col + 'freq'].map(c2p1)
            train[col + 'p2'] = train[col + 'freq'].map(c2p2)
            test[col + 'p1'] = test[col + 'freq'].map(c2p1)
            test[col + 'p2'] = test[col + 'freq'].map(c2p2)
            train[col + 'm'] = train[col + 'freq'].map(c2m)
            test[col + 'm'] = test[col + 'freq'].map(c2m)
            train[col + 's2'] = train[col + 'freq'].map(c2s2)
            train[col + 's1'] = train[col + 'freq'].map(c2s1)
            test[col + 's2'] = test[col + 'freq'].map(c2s2)
            test[col + 's1'] = test[col + 'freq'].map(c2s1)
            train[col + 'c1'] = (train[col + 'freq'] == 1).astype(np.int8)
            test[col + 'c1'] = (test[col + 'freq'] == 1).astype(np.int8)
            train[col + 'isP1'] = (train[col] < train[col + 'm']).astype(np.int8)
            test[col + 'isP1'] = (test[col] < test[col + 'm']).astype(np.int8)


            train[col + 'p1dis'] = (train[col] - train[col+'p1']) * train[col + 'isP1'] * train[col + 'c1'] +\
                                   (train[col] - train[col+'mean']) * (1-train[col + 'c1'])
            test[col + 'p1dis'] = (test[col] - test[col + 'p1']) * test[col + 'isP1'] * test[col + 'c1'] + \
                                  (test[col] - test[col + 'mean']) * (1 - test[col + 'c1'])
            train[col + 'p2dis'] = (train[col] - train[col+'p2']) * (1-train[col + 'isP1']) * train[col + 'c1'] +\
                                   (train[col]-train[col+'mean']) * (1-train[col + 'c1'])
            test[col + 'p2dis'] = (test[col] - test[col + 'p2']) * (1-test[col + 'isP1']) * test[col + 'c1'] +\
                                   (test[col]-test[col+'mean']) * (1-test[col + 'c1'])

            train[col + 'p1norm'] = ((train[col] - train[col + 'p1'])/train[col + 's1']) * train[col + 'isP1'] * train[col + 'c1'] + \
                                   ((train[col] - train[col + 'mean'])/train[col + 'std']) * (1 - train[col + 'c1'])
            test[col + 'p1norm'] = ((test[col] - test[col + 'p1'])/test[col + 's1']) * test[col + 'isP1'] * test[col + 'c1'] + \
                                  ((test[col] - test[col + 'mean'])/test[col+'std']) * (1 - test[col + 'c1'])
            train[col + 'p2norm'] = ((train[col] - train[col + 'p2'])/train[col + 's2']) * (1 - train[col + 'isP1']) * train[col + 'c1'] + \
                                   ((train[col] - train[col + 'mean'])/train[col + 'std']) * (1 - train[col + 'c1'])
            test[col + 'p2norm'] = ((test[col] - test[col + 'p2'])/test[col + 's2']) * (1 - test[col + 'isP1']) * test[col + 'c1'] + \
                                  ((test[col] - test[col + 'mean'])/test[col+'std']) * (1 - test[col + 'c1'])

            dis1 = ((train[col] - train[col + 'median']) / (train[col + 'p1'] - train[col + 'median'])) ** (train[col + 'isP1']* train[col + 'c1'])
            dis2 = ((train[col] - train[col + 'median']) / (train[col + 'mean'] - train[col + 'median'])) ** (1-train[col + 'c1'])
            train[col + 'p1dis_rate2'] = dis1 * dis2

            dis1 = ((train[col] - train[col + 'median']) / (train[col + 'p2'] - train[col + 'median'])) ** ((1-train[col + 'isP1']) * train[col + 'c1'])
            dis2 = ((train[col] - train[col + 'median']) / (train[col + 'mean'] - train[col + 'median'])) ** (1-train[col + 'c1'])
            train[col + 'p2dis_rate2'] = dis1 * dis2

            dis1 = ((test[col] - test[col + 'median']) / (test[col + 'p1'] - test[col + 'median'])) ** (test[col + 'isP1'] * test[col + 'c1'])
            dis2 = ((test[col] - test[col + 'median']) / (test[col + 'mean'] - test[col + 'median'])) ** (1 - test[col + 'c1'])
            test[col + 'p1dis_rate2'] = dis1 * dis2

            dis1 = ((test[col] - test[col + 'median']) / (test[col + 'p2'] - test[col + 'median'])) ** ((1 - test[col + 'isP1']) * test[col + 'c1'])
            dis2 = ((test[col] - test[col + 'median']) / (test[col + 'mean'] - test[col + 'median'])) ** (1 - test[col + 'c1'])
            test[col + 'p2dis_rate2'] = dis1 * dis2

            train[col + 'p1dis_rate'] = ((train[col] - train[col + 'm']) / (train[col + 'p1'] - train[col + 'm']))
            train[col + 'p2dis_rate'] = ((train[col] - train[col + 'm']) / (train[col + 'p2'] - train[col + 'm']))
            test[col + 'p1dis_rate'] = ((test[col] - test[col + 'm']) / (test[col + 'p1'] - test[col + 'm']))
            test[col + 'p2dis_rate'] = ((test[col] - test[col + 'm']) / (test[col + 'p2'] - test[col + 'm']))


        if True: # diff feature
            freq2diff = {c: pd.Series(map_count[map_count == c].index,index=map_count[map_count == c].index).diff().fillna(0) for c in unique_count}
            v2diff = {}
            for c,v in freq2diff.items():
                if c == 2:
                    v2diff.update(v.to_dict())
            freq2diff = {c: d.values for c,d in freq2diff.items()}

            diff_mean = {c: v[1:].mean() if v.shape[0] > 1 else np.nan for c, v in freq2diff.items()}
            diff_max = {c: v[1:].max() if v.shape[0] > 1 else np.nan for c, v in freq2diff.items()}

            train[col + 'diffmean'] = train[col + 'freq'].map(diff_mean)
            test[col + 'diffmean'] = test[col + 'freq'].map(diff_mean)

            train[col + 'diffmax'] = train[col + 'freq'].map(diff_max)
            test[col + 'diffmax'] = test[col + 'freq'].map(diff_max)


        map_count = all_data[col].value_counts(sort=False)
        train[col + 'freq'] = train[col].map(map_count).astype(np.int16)
        test[col + 'freq'] = test[col].map(map_count).astype(np.int16)

        for feat in feat_stat:
            freq_cols.append(col+feat)
    for col in set(two_count_peak_cols) & set(cols):
        freq_cols.append(col + 'bin2')
    for col in set(three_count_peak_cols) & set(cols):
        freq_cols.append(col + 'bin3')
    return train[freq_cols],test[freq_cols]

def get_features(train,test,all_data,pre_feat):

    workers = mp.cpu_count()//2
    pool = mp.Pool(workers)
    results = []
    ave_col = (len(pre_feat)+workers-1)//workers
    for i in range(workers):
        deal_cols = pre_feat[i*ave_col:(i+1)*ave_col]
        res = pool.apply_async(cal_freq_feat,
                    args=(train[deal_cols],test[deal_cols],all_data[deal_cols],deal_cols))
        results.append(res)
    pool.close()
    pool.join()

    tr,te = [],[]
    for res in tqdm(results):
        sub_tr,sub_te = res.get()
        tr.append(sub_tr)
        te.append(sub_te)
    train = pd.concat([train]+tr,axis=1)
    test = pd.concat([test]+te,axis=1)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(train.shape,test.shape)
    return train,test

def augment(tr,y,cols,t1=5,t2=5):
    print(t1,t2)
    xs,xn = [],[]

    feat_cols = ['','freq']+feat_stat
    all_feat = set(tr.columns)
    for i in range(t1):
        x1 = tr[y>0].reset_index(drop=True)
        ids = np.arange(x1.shape[0])
        for c in cols:
            np.random.shuffle(ids)
            for feat in feat_cols:
                if c+feat in all_feat:
                    x1[c+feat] = x1[c+feat].values[ids]
            if c in two_count_peak_cols:
                x1[c + 'bin2'] = x1[c + 'bin2'].values[ids]
            if c in three_count_peak_cols:
                x1[c + 'bin3'] = x1[c + 'bin3'].values[ids]
        xs.append(x1)

    for i in range(t2):
        x1 = tr[y==0].reset_index(drop=True)
        ids = np.arange(x1.shape[0])
        for c in cols:
            np.random.shuffle(ids)
            for feat in feat_cols:
                if c + feat in all_feat:
                    x1[c+feat] = x1[c+feat].values[ids]
            if c in two_count_peak_cols:
                x1[c + 'bin2'] = x1[c + 'bin2'].values[ids]
            if c in three_count_peak_cols:
                x1[c + 'bin3'] = x1[c + 'bin3'].values[ids]
        xn.append(x1)

    ys = np.ones(t1 * len(xs[0]))
    yn = np.zeros(t2 * len(xn[0]))
    y = np.concatenate([y, ys, yn])
    tr = pd.concat([tr]+xs+xn).reset_index(drop=True)

    print(tr.shape)
    return tr,y
