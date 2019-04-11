import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from tqdm import tqdm
from scipy.stats import *
import multiprocessing as mp
from catboost import Pool, CatBoostClassifier,CatBoostRegressor
import warnings
import gc
from utils import *
warnings.filterwarnings('ignore')
pd.set_option('display.max_row',100)


def main(cfg):

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    pre_feat = [c for c in train.columns if c not in ['ID_code', 'target']] #basic features
    target = train['target'].values

    params = {
        'num_rounds': 6000000,
        'verbose_eval': 5000,
        'early_stop': 4000,
    }
    print(params)

    if True: # filter no use sample
        freq_cols = []
        for col in pre_feat:
            test[col + 'freq'] = test[col].map(test[col].value_counts(sort=False))
            freq_cols.append(col + 'freq')
        test['num_unique'] = (test[freq_cols]>=2).sum(axis=1)
        real_idx = test['num_unique'] < 200
        real_test = test.loc[real_idx,pre_feat]
        # real_test = test.copy()

        assert len(real_test) == 100000
        all_data = train[pre_feat].append(real_test).reset_index(drop=True)
        test.drop(freq_cols,axis=1,inplace=True)

    train,test = get_features(train,test,all_data,pre_feat)
    train['allfreq'] = train[freq_cols].sum(axis=1)
    train['num_unique'] = (train[freq_cols]>=2).sum(axis=1)

    test['allfreq'] = test[freq_cols].sum(axis=1)
    test['num_unique'] = (test[freq_cols]>=2).sum(axis=1)

    new_stat = ['freq'] + feat_stat
    features = pre_feat + ['allfreq', 'num_unique'] + \
               [col + 'bin2' for col in two_count_peak_cols] + \
               [col + 'bin3' for col in three_count_peak_cols]
    for s in new_stat:
        features += [col+s for col in pre_feat]

    folds = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=False,
            random_state=random_state).split(train.values, target)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    # feat_score = pd.read_csv('feat_importance.csv')['name'].values[:2000].tolist()
    # features = list(set(features) & set(feat_score))
    for fold_, (trn_idx, val_idx) in enumerate(folds):

        if fold_ not in cfg['folds']:
            continue

        val_x, val_y = train.iloc[val_idx], target[val_idx]
        tr_x, tr_y = train.iloc[trn_idx], target[trn_idx]
        # tr_x,val_x,te_x = cal_freq_TE(tr_x,val_x,test,all_data,pre_feat)

        tr_x,val_x,te_x = tr_x[features],val_x[features],test[features]

        tr_x, tr_y = augment(tr_x,tr_y,pre_feat,cfg['t1'],cfg['t2'])
        tr_x['allfreq'] = tr_x[freq_cols].sum(axis=1)
        tr_x['num_unique'] = (tr_x[freq_cols] >= 2).sum(axis=1)

        print("Fold idx:{}".format(fold_ + 1))


        d_train = Pool(tr_x, label=tr_y)
        d_valid = Pool(val_x, label=val_y)

        model = CatBoostClassifier(iterations=params['num_rounds'],
                                   learning_rate=0.003,
                                   od_type='Iter',
                                   od_wait=params['early_stop'],
                                   loss_function="Logloss",
                                   eval_metric='AUC',
                                   #         depth=3,
                                   bagging_temperature=0.7,
                                   random_seed=2019,
                                   task_type='GPU'
                                   )
        model.fit(d_train, eval_set=d_valid,
                  use_best_model=True,
                  verbose=params['verbose_eval']
                  )

        oof[val_idx] = model.predict_proba(val_x)[:, 1]
        pred = model.predict_proba(te_x)[:, 1]
        predictions += pred / cfg['n_splits']
        threshold_search(target[val_idx],oof[val_idx])
        np.save('../submit/' + cfg['name'] + str(fold_), pred)
        np.save('../oof/' + cfg['name'] + ''.join([str(fold) for fold in cfg['folds']]), oof)
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    # np.save('../input/oof',oof)
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub["target"] = predictions
    sub.to_csv(cfg['name']+"submission.csv", index=False)



if __name__ == '__main__':

    cfg = {}
    cfg['n_splits'] = 10
    cfg['t1'] = 5
    cfg['t2'] = 5
    cfg['folds'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # cfg['name'] = 'cat_g1feat51rs'
    # random_state = 51
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'cat_g1feat52rs'
    # random_state = 52
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'cat_g1feat53rs'
    # random_state = 53
    # np.random.seed(random_state)
    # main(cfg)

    # cfg['name'] = 'cat_g2feat54rs'
    # random_state = 54
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'cat_g2feat55rs'
    # random_state = 55
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'cat_g2feat56rs'
    # random_state = 56
    # np.random.seed(random_state)
    # main(cfg)

    cfg['name'] = 'cat_g3feat57rs'
    random_state = 57
    np.random.seed(random_state)
    main(cfg)

    cfg['name'] = 'cat_g3feat58rs'
    random_state = 58
    np.random.seed(random_state)
    main(cfg)

    cfg['name'] = 'cat_g3feat59rs'
    random_state = 59
    np.random.seed(random_state)
    main(cfg)
