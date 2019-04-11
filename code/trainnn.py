import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
tf_cfg = tf.ConfigProto(allow_soft_placement=True)
tf_cfg.gpu_options.allow_growth=True
session = tf.Session(config=tf_cfg)
KTF.set_session(session)
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import StratifiedKFold
from utils import *
from keras.layers import *
from keras.optimizers import Nadam
from keras import Model
from keras import backend as K
warnings.filterwarnings('ignore')
pd.set_option('display.max_row',100)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    sample_size = x['x1'].shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)

    mixed_x = {k:lam*v + (1-lam)*v[index_array] for k,v in x.items()}
    mixed_y = lam * y + (1 - lam) * y[index_array]
    return mixed_x, mixed_y



def batch_generator(X, y, cfg):
    y = np.array(y)
    sample_size = X['x1'].shape[0]
    index_array = np.arange(sample_size)
    steps = (sample_size + cfg['bs'] - 1) // cfg['bs']
    while True:
        np.random.shuffle(index_array)
        for i in range(steps):
            batch_ids = index_array[i*cfg['bs']:(i+1)*cfg['bs']]
            X_batch = {k:v[batch_ids] for k,v in X.items()}
            y_batch = y[batch_ids]
            if cfg['mixup']:
                X_batch, y_batch = mixup_data(X_batch, y_batch, alpha=1.0)
            yield X_batch, y_batch

def get_model(cfg):
    x1_in = Input(shape=cfg['x1_shape'],name='x1')
    x2_in = Input(shape=cfg['x2_shape'],name='x2')

    x1 = BatchNormalization()(x1_in)
    x2 = BatchNormalization()(x2_in)

    x1 = LocallyConnected1D(32, activation='relu',kernel_size=1)(x1)
    x1 = Dense(96, activation='relu')(x1)
    x1 = Dense(8, activation='relu')(x1)
    x1 = Flatten()(x1)

    x2 = Reshape((-1,1))(x2)
    x2 = LocallyConnected1D(16, activation='relu',kernel_size=1)(x2)
    x2 = Dense(4, activation='relu')(x2)
    x2 = Flatten()(x2)
    x = concatenate([x1,x2])
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model([x1_in,x2_in],output)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=cfg['lr']))
    return model

def data_reshape(x):
    dataset = []
    var_cols = ['var_'+str(i) for i in range(200)]
    for feat in feat_stat:
        cols = [c + feat for c in var_cols]
        dataset.append(np.expand_dims(x[cols].values,axis=2))
    dataset = np.concatenate(dataset,axis=-1)
    print(dataset.shape)
    binfeat = [col+'bin2' for col in two_count_peak_cols] + [col+'bin3' for col in three_count_peak_cols]
    dataset = {
        'x1':dataset,
        'x2':x[binfeat].values
    }
    return dataset

def main(cfg):

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    pre_feat = [c for c in train.columns if c not in ['ID_code', 'target']] #basic features
    target = train['target'].values

    print(cfg)

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

    folds = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True,
            random_state=random_state).split(train.values, target)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds):

        if fold_ not in cfg['folds']:
            continue

        val_x, val_y = train.iloc[val_idx], target[val_idx]
        tr_x, tr_y = train.iloc[trn_idx], target[trn_idx]
        tr_x,val_x,te_x = tr_x[features],val_x[features],test[features]

        tr_x, tr_y = augment(tr_x,tr_y,pre_feat,cfg['t1'],cfg['t2'])
        tr_x['allfreq'] = tr_x[freq_cols].sum(axis=1)
        tr_x['num_unique'] = (tr_x[freq_cols] >= 2).sum(axis=1)

        cfg['num_data'] = len(tr_x)
        print("Fold idx:{}".format(fold_ + 1))

        tr_x = data_reshape(tr_x)
        val_x = data_reshape(val_x)
        te_x = data_reshape(te_x)

        if True: # init
            best_epoch = 0
            best_score = 0
            cfg['x1_shape'] = tr_x['x1'].shape[1:]
            cfg['x2_shape'] = (tr_x['x2'].shape[1],)

        model = get_model(cfg)

        for epoch in range(10000):
            cfg['mixup'] = False
            data_loader = batch_generator(tr_x,tr_y,cfg)
            model.fit_generator(
                data_loader,
                steps_per_epoch=(cfg['num_data']+cfg['bs']-1)//cfg['bs'],
                workers=16,
                verbose=2
            )
            pred = model.predict(val_x)[:,0]
            score = roc_auc_score(val_y,pred)

            if epoch in [5]:
                K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)/3)
            if epoch in [8]:
                K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)/2)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                oof[val_idx] = pred
                model.save_weights('../model/'+cfg['name']+str(fold_)+'.h5')
            print('curr score', score, '\t', 'best_score', best_score)
            if epoch - best_epoch > cfg['patience']:
                break

        model.load_weights('../model/'+cfg['name']+str(fold_)+'.h5')
        pred = model.predict(te_x)[:,0]
        predictions += pred / cfg['n_splits']

        threshold_search(target[val_idx],oof[val_idx])
        np.save('../submit/' + cfg['name'] + str(fold_), pred)
        np.save('../oof/' + cfg['name'] + ''.join([str(fold) for fold in cfg['folds']]), oof)
    np.save('../oof/' + cfg['name'] + ''.join([str(fold) for fold in cfg['folds']]), oof)

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    sub = pd.DataFrame({"ID_code": test.ID_code.values})
    sub["target"] = predictions
    sub.to_csv(cfg['name']+"submission.csv", index=False)



if __name__ == '__main__':

    cfg = {}
    cfg['n_splits'] = 10
    cfg['patience'] = 5
    cfg['t1'] = 5
    cfg['t2'] = 5
    cfg['lr'] = 0.0005
    cfg['bs'] = 256
    cfg['folds'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # cfg['name'] = 'ann_g2feat64rs'
    # random_state = 64
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'ann_g2feat65rs'
    # random_state = 65
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'ann_g2feat66rs'
    # random_state = 66
    # np.random.seed(random_state)
    # main(cfg)

    cfg['name'] = 'ann_g3feat61rs'
    random_state = 61
    np.random.seed(random_state)
    main(cfg)

    cfg['name'] = 'ann_g3feat62rs'
    random_state = 62
    np.random.seed(random_state)
    main(cfg)

    cfg['name'] = 'ann_g3feat63rs'
    random_state = 63
    np.random.seed(random_state)
    main(cfg)

    # cfg['name'] = 'ann_g1feat67rs'
    # random_state = 67
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'ann_g1feat68rs'
    # random_state = 68
    # np.random.seed(random_state)
    # main(cfg)
    #
    # cfg['name'] = 'ann_g1feat69rs'
    # random_state = 69
    # np.random.seed(random_state)
    # main(cfg)

