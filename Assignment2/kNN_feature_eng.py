
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split  
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# specifying some paths
path = r"C:\Users\janse\OneDrive\Bureaublad\Master\Data Mining Techniques\Repo\Assignment2"
testpath = os.path.join(path, 'data', 'test_set_VU_DM.csv')
trainpath = os.path.join(path, 'data', 'training_set_VU_DM.csv')
samplepath = os.path.join(path, 'data', 'train_sample.csv')
resultpath = os.path.join(path, 'data', 'result.csv')


# In[3]:


# load train_main
# made a function for quick reusability during development
def reload_train():
    train_main = pd.read_csv(trainpath)
    train_main.date_time = pd.to_datetime(train_main.date_time)
    return train_main


# In[4]:


train_main = reload_train()


# # Feature Extraction

# In[5]:


def average_prop_score(train_df):
    cols = ['prop_id', 'price_usd', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2']
    df = train_df[cols]
    df['prop_location_score2'] = df['prop_location_score2'].fillna(df.prop_location_score2.mean())
    return df.groupby('prop_id').agg('mean').reset_index()

def rate(row, mean_rate):
    if row['srch_id'] < 20:
        return mean_rate
    else:
        return row['click_bool']
    
def avg_click_rate(train_main):
    prop_click_rate = train_main[['prop_id','click_bool', 'srch_id']].groupby('prop_id').agg({'click_bool' : 'mean', 'srch_id': 'count'}).reset_index()
    mean_rate = prop_click_rate.click_bool.mean()
    prop_click_rate['click_rate'] = prop_click_rate.apply(lambda x: rate(x, mean_rate), axis=1)
    return prop_click_rate[['prop_id', 'click_rate']]

def split_sets(df, test_size = 0.2, shuffle=False, downsample=False):
    '''
        Returns: X_train, X_test, Y_train, Y_test
    '''
    df_train, df_test = train_test_split(df.sort_values('srch_id'), test_size=test_size, shuffle=shuffle) 
    
    if downsample:
        df_train = downsample(df_train)
    
    X_train = df_train.drop(['click_bool','booking_bool'] , axis=1)
    X_train = X_train.fillna(X_train.mean())
    X_test = df_test.drop(['click_bool','booking_bool'] , axis=1)    
    X_test = X_test.fillna(X_test.mean())
    
    Y_train = df_train.click_bool + df_train.booking_bool * 4
    Y_test = df_test.click_bool + df_train.booking_bool * 4
    
    target_train = df_train.click_bool + 4 * df_train.booking_bool
    target_test = df_test.click_bool + 4 * df_test.booking_bool    
    
    return X_train, X_test, Y_train, Y_test, target_train, target_test

def cumulate_comp_scores(train_df):    
    # Cumulative competitor scores

    cols = [col for col in train_df.columns if col.endswith('rate_percent_diff')]
    cols2 = [col for col in train_df.columns if col.endswith('_rate')]
    cols3 = [col for col in train_df.columns if col.endswith('_inv')]

    df = train_df[cols.extend(cols2).extend(cols3).extend(['prop_id'])]
    df['comp_rate_percent_diff_cumm'] = 0

    for col1, col2, col3 in zip(cols, cols2, cols3):
        df['comp_rate_percent_diff_cumm'] += df[col1].fillna(0) * df[col2].fillna(0) * (1 - df[col3].fillna(0))
        
    df = train2.drop(cols, axis=1)
    df = train2.drop(cols2, axis=1)
    df = train2.drop(cols3, axis=1)
    return df

def avg_prop_features(train_main, test_main = None, predict_phase=False):
    cols = [
       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
       'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'price_usd', 'promotion_flag',
       'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count',
       'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool']
    df = train_main[cols].groupby('prop_id').agg(['mean', 'std']) # median?
    
    if predict_phase:
        df = df.merge(test_main[cols + ['srch_id']], on='prop_id')
    else:
        df = df.merge(train_main[cols + ['srch_id']], on='prop_id')   
        
    return df

def create_labels(df):
    df['label'] = df.click_bool + 4 * df.booking_bool
    return df

def drop_unnecessary_cols(df):
    cols_to_drop = ['prop_id', 'srch_id', 'booking_bool', 'position', 'gross_price_usd', 'date_time']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df


# In[6]:


def feature_eng(train_main):
    df = train_main[['prop_id', 'srch_id','date_time', 'click_bool', 'booking_bool']]        
    df = df.merge(avg_prop_features(train_main), on=['prop_id', 'srch_id'])    
    df = df.fillna(df.mean())
    return df.sort_values(['prop_id', 'srch_id'], ascending=[True, False])

def test_eng(test_main, train_main):
    df = avg_prop_features(train_main, test_main=test_main, predict_phase=True)  
    df = df.fillna(df.mean())
    return df.sort_values(['srch_id', 'prop_id'], ascending=[True, False])


# # Evaluation

# In[ ]:



 


# In[7]:



def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def eval_ndcg(df, srch_id_col, predict_col, value_col, n=9999999999999999):
    df = df.sort_values([srch_id_col, predict_col], ascending=False)
    k = 5
    ndcgs = []
    for i, srchid in enumerate(df[srch_id_col].unique()):
        if i == n:
            break
        if i % 10000 == 0 and i != 0:
            print(i)
            print(np.mean(ndcgs))
        r = df[df[srch_id_col] == srchid][value_col]
        ndcgs.append(ndcg_at_k(r,k))

    print(np.mean(ndcgs))


# In[8]:


# Create train / test sets
df = feature_eng(train_main)
df = create_labels(df)

df_train = df[df.date_time < '2013-05-30']
df_test = df[df.date_time >= '2013-05-30']

df = drop_unnecessary_cols(df)


# # Nearest Neighbour

# In[17]:


df_train.columns


# In[ ]:


from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  

train_y = df_train.label
test_y = df_test.label

drop_cols = ['label','prop_id','srch_id','date_time','click_bool','booking_bool']
train_x = df_train.drop(drop_cols, axis=1)
test_x = df_train.drop(drop_cols, axis=1)

classifier = KNeighborsClassifier(n_neighbors=20)  
classifier.fit(train_x, train_y)  

pred = classifier.predict(test_x)


# In[ ]:


# evaluation
results = pd.DataFrame({'srch_id': X_test.srch_id, 'predicted': pred, 'actual' : test_y})
eval_score = eval_ndcg(results, 'srch_id', 'predicted', 'actual')

