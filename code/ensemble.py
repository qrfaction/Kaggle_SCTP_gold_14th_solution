import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def folds_average(cfg):


    test = pd.read_csv('../input/test.csv',usecols=['ID_code'])
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub["target"] = np.average([np.load('./submit/'+cfg['name']+str(i)+'.npy') for i in range(6)],axis=0)
    sub.to_csv(cfg['name'] + "submission.csv", index=False)

    target = pd.read_csv('../input/train.csv',usecols=['target'])['target'].values
    oof = np.sum([np.load(file) for file in glob('../oof/'+cfg['name']+'*.npy')], axis=0)

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

def stacking(files):

    tr_oof = [np.sum([np.load(file) for file in glob('./oof/'+name+'*.npy')], axis=0)[:,np.newaxis] for name in files]
    tr_oof = np.concatenate(tr_oof,axis=1)
    test_pred = [np.average([np.load('./submit/'+name+str(i)+'.npy') for i in range(10)],axis=0)[:,np.newaxis] for name in files]
    test_pred = np.concatenate(test_pred,axis=1)
    target = pd.read_csv('../input/train.csv', usecols=['target'])['target'].values


    print(tr_oof.shape,test_pred.shape)
    print(pd.DataFrame(tr_oof,columns=files).corr())
    # param = {
    #     'bagging_freq': 5,
    #     'bagging_fraction': 0.8,
    #     'boost_from_average': 'false',
    #     'boost': 'gbdt',
    #     'feature_fraction': 1,
    #     'learning_rate': 0.0083,
    #     'max_depth': 3,
    #     'metric': 'auc',
    #     'min_data_in_leaf': 80,
    #     # 'min_sum_hessian_in_leaf': 10.0,
    #     'num_leaves': 4,
    #     'num_threads': -1,
    #     'tree_learner': 'serial',
    #     'objective': 'binary',
    #     'lambda_l2':0.5,
    #     # 'lambda_l1':0.5,
    #     # 'device_type':'gpu',
    #     # 'is_unbalance':True,
    #     'verbosity': -1,
    # }
    # folds = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=False,
    #                         random_state=99999).split(tr_oof, target)
    #
    # predictions = np.zeros(len(test_pred))
    # oof = np.zeros(len(tr_oof))
    # for fold_, (trn_idx, val_idx) in enumerate(folds):
    #     print('curr fold ',fold_)
    #     val_x, val_y = tr_oof[val_idx], target[val_idx]
    #     tr_x, tr_y = tr_oof[trn_idx], target[trn_idx]
    #
    #     tr_x = lgb.Dataset(tr_x, label=tr_y)
    #     val_data = lgb.Dataset(val_x, label=val_y)
    #
    #
    #     clf = lgb.train(param, tr_x, 1000000, valid_sets = [tr_x, val_data],
    #                     verbose_eval=200, early_stopping_rounds = 400)
    #     oof[val_idx] = clf.predict(val_x,num_iteration=clf.best_iteration)
    #     pred = clf.predict(test_pred, num_iteration=clf.best_iteration)
    #
    #     # clf = LogisticRegression(C=0.5,tol=1e-6,max_iter=10000,n_jobs=-1)
    #     # clf.fit(tr_x,tr_y)
    #     # oof[val_idx] = clf.predict_proba(val_x)[:,1]
    #     # pred = clf.predict_proba(test_pred)[:,1]
    #     print("CV score: {:<8.5f}".format(roc_auc_score(target[val_idx], oof[val_idx])))
    #     predictions += pred / cfg['n_splits']
    tr_oof = pd.DataFrame(tr_oof).rank()
    # print(tr_oof)
    oof = np.average(tr_oof,axis=1)
    predictions = np.average(test_pred,axis=1)
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    test = pd.read_csv('../input/test.csv', usecols=['ID_code'])
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub['target'] = predictions
    sub.to_csv(cfg['name'] + "submission.csv", index=False)


def weighted_average(files):

    cat_oof = np.average([np.sum([np.load(file) for file in glob('./oof/'+name+'*.npy')], axis=0)
                          for name in files if 'cat_' in name],axis=0)
    cat_submit = np.average([np.average([np.load('./submit/'+name+str(i)+'.npy') for i in range(10)],axis=0)
                             for name in files if 'cat_' in name],axis=0)


    lgb_oof = np.average([np.sum([np.load(file) for file in glob('./oof/'+name+'*.npy')], axis=0)
                          for name in files if 'lgb_' in name],axis=0)
    lgb_submit = np.average([np.average([np.load('./submit/'+name+str(i)+'.npy') for i in range(10)],axis=0)
                             for name in files if 'lgb_' in name],axis=0)

    nn_oof = np.average([np.sum([np.load(file) for file in glob('./oof/'+name+'*.npy')], axis=0)
                         for name in files if 'nn_' in name],axis=0)
    nn_submit = np.average([np.average([np.load('./submit/'+name+str(i)+'.npy') for i in range(10)],axis=0)
                            for name in files if 'nn_' in name],axis=0)



    target = pd.read_csv('../input/train.csv', usecols=['target'])['target'].values
    oof = np.average([lgb_oof,nn_oof],axis=0)
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    predictions = np.average([lgb_submit,nn_submit], axis=0)
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    test = pd.read_csv('../input/test.csv', usecols=['ID_code'])
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub['target'] = predictions
    sub.to_csv(cfg['name'] + "submission.csv", index=False)

if __name__ == '__main__':

    cfg ={}
    cfg['name'] = 'ann_g2feat66rs'
    cfg['n_splits'] = 5
    files = [
        'lgb_seed72',
        'lgb_seed42',
        'lgb_seed666',
        'cat_seed66',
        'nn_lr5',
        'nn_g2feat5_8step',
        # 'lgb_bin2815',
        # 'lgb_g2featbin1023',
        'lgb_g2featbin2047',
        # 'lgb_g2featbin2815',
        'lgb_g2feat2047bin99999sf',
        'lgb_g3feat2815bin',
        'cat_42seedg2feat',

        'cat_g1feat51rs',
        'cat_g1feat52rs',
        'cat_g1feat53rs',
        'cat_g2feat54rs',
        'cat_g2feat55rs',
        'cat_g2feat56rs',
        'cat_g3feat57rs',
        'cat_g3feat58rs',

        'ann_g3feat61rs',
        'ann_g3feat62rs',
        'ann_g3feat63rs',
        'ann_g2feat64rs',
        'ann_g2feat65rs',
        'ann_g2feat66rs',
        'ann_g1feat67rs',
        'ann_g1feat68rs',
        'ann_g1feat69rs',

        'nn_g2feat41rs',
        'lgb_g2feat42rs',
        'lgb_g2feat43rs',
        'nn_g3feat44rs',
        'lgb_g3feat45rs',
        'lgb_g3feat46rs',
        'lgb_g1feat47rs',
        'lgb_g1feat48rs',
        'lgb_g1feat49rs',

    ]
    folds_average(cfg)
    # stacking(files)
    # weighted_average(files)






















