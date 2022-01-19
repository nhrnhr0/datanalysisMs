import os
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split

# a lightfm training model on the data from the data folder to predict probebility of a user liking/buying a product
# and find correlation between users and products, products and products and users and users
cwd = os.getcwd()

BASE_DATA_PATH = cwd + '/main/data/'


def conver_user_interactions(dir_path):
    # read all the files in the directory
    # file structure: <user_id>.xls
    # file headers: product_id, duration, prc, prc2
    # return a coords dataframe with the following columns:
    # user_id, product_id, prc2
    files = os.listdir(dir_path)
    ret = pd.DataFrame()
    for file in files:
        xls = pd.ExcelFile(r'{}/{}'.format(dir_path, file))
        iteraction_data = xls.parse(0)
        user_id = file.split('.')[0]
        iteraction_data['user_id'] = user_id
        iteraction_data.drop(['duration', 'prc'], axis=1, inplace=True)
        iteraction_data.rename(columns={'prc2': 'prc', 'id':'product_id'}, inplace=True)
        iteraction_data = iteraction_data.reindex(columns=['user_id', 'product_id', 'prc'])
        ret = ret.append(iteraction_data)
    return ret

def get_interactions_dataset(data):
    all_users = pd.read_csv(BASE_DATA_PATH + 'global/users.csv')
    all_products = pd.read_csv(BASE_DATA_PATH + 'global/products.csv')
    
    dataset = Dataset()
    dataset.fit(users=all_users['id'], items=all_products['id'],)
    return dataset

def main():
    data = conver_user_interactions(BASE_DATA_PATH + 'clients_interactions')
    data.to_csv(BASE_DATA_PATH + 'clients_interactions.csv', index=False)
    print(data.dtypes)
    # convert the user_id to int
    data['user_id'] = data['user_id'].astype(int)
    print(data.dtypes)
    dataset = get_interactions_dataset(data)
    print(dataset)
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))
    (interactions, weights) = dataset.build_interactions(data.itertuples(index=False))
    print(repr(interactions))
    model = LightFM(loss='bpr')
    train,test = random_train_test_split(interactions, test_percentage=0.2)
    for i in range(100):
        model.fit_partial(train, item_features=None,epochs=300, num_threads=2,verbose=True)
        #print(model)
        test_precision = precision_at_k(model, test, k=5).mean()
        print(test_precision)
    pass
if __name__ == '__main__':
    main()