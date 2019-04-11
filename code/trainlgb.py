import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from tqdm import tqdm
from scipy.stats import *
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

    param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.4,
        'boost_from_average': 'false',
        'boost': 'gbdt',
        'feature_fraction': 0.041,
        'learning_rate': 0.0083,
        'max_depth': -1,
        'metric': 'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 17,
        'num_threads': -1,
        'tree_learner': 'serial',
        'objective': 'binary',
        # 'device_type':'gpu',
        # 'is_unbalance':True,
        'verbosity': -1,
        'max_bin':2815,
    }
    print(param)


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
    features = pre_feat + ['allfreq','num_unique'] + \
               [col+'bin2' for col in two_count_peak_cols] +\
               [col + 'bin3' for col in three_count_peak_cols]
    for s in new_stat:
        features += [col+s for col in pre_feat]

    folds = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True,
            random_state=random_state).split(train.values, target)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    print(cfg)

    for fold_, (trn_idx, val_idx) in enumerate(folds):
        if fold_ not in cfg['folds']:
            continue
        val_x, val_y = train.iloc[val_idx], target[val_idx]
        tr_x, tr_y = train.iloc[trn_idx], target[trn_idx]
        # tr_x,val_x,te_x = cal_freq_TE(tr_x,val_x,test,pre_feat)
        tr_x,val_x,te_x = tr_x[features],val_x[features],test[features]

        tr_x, tr_y = augment(tr_x,tr_y,pre_feat,cfg['t1'],cfg['t2'])
        tr_x['allfreq'] = tr_x[freq_cols].sum(axis=1)
        tr_x['num_unique'] = (tr_x[freq_cols] >= 2).sum(axis=1)

        print("Fold idx:{}".format(fold_ + 1),tr_x.shape)
        gc.collect()
        tr_x = lgb.Dataset(tr_x, label=tr_y)
        val_data = lgb.Dataset(val_x, label=val_y)


        clf = lgb.train(param, tr_x, 1000000, valid_sets = [tr_x, val_data],
                        verbose_eval=5000, early_stopping_rounds = 4000)
        oof[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration)
        pred = clf.predict(te_x, num_iteration=clf.best_iteration)
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

    cfg['folds'] = [0,1,2,3,4,5,6,7,8,9]

    # cfg['name'] = 'nn_g2feat41rs'
    # random_state = 41
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'lgb_g2feat42rs'
    # random_state = 42
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'lgb_g2feat43rs'
    # random_state = 43
    # np.random.seed(random_state)
    # main(cfg)

    # cfg['name'] = 'nn_g3feat44rs'
    # random_state = 44
    # np.random.seed(random_state)
    # main(cfg)
    #
    cfg['name'] = 'lgb_g3feat45rs'
    random_state = 45
    np.random.seed(random_state)
    main(cfg)

    cfg['name'] = 'lgb_g3feat46rs'
    random_state = 46
    np.random.seed(random_state)
    main(cfg)

    # cfg['name'] = 'lgb_g1feat47rs'
    # random_state = 47
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'lgb_g1feat48rs'
    # random_state = 48
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'lgb_g1feat49rs'
    # random_state = 49
    # np.random.seed(random_state)
    # main(cfg)







