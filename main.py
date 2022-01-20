import os
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k,auc_score
from lightfm.cross_validation import random_train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy

# a lightfm training model on the data from the data folder to predict probebility of a user liking/buying a product
# and find correlation between users and products, products and products and users and users
cwd = os.getcwd()
import matplotlib

BASE_DATA_PATH = cwd + '/main/data/'
# seed for pseudonumber generations
SEEDNO = 42
matplotlib.use("svg")

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
def get_all_users():
    all_users = pd.read_csv(BASE_DATA_PATH + 'global/users.csv')
    return all_users

def get_all_products():
    all_products = pd.read_csv(BASE_DATA_PATH + 'global/products.csv')
    return all_products

def get_interactions_dataset(data):
    
    
    dataset = Dataset()
    dataset.fit(users=get_all_users()['id'], items=get_all_products()['id'],)
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
    model = LightFM(loss='warp')
    iterarray = range(1000)
    train,test = random_train_test_split(interactions, test_percentage=0.2,random_state=np.random.RandomState(SEEDNO))
    count = 0
    best = 0
    train_scores = []
    test_scores = []
    for e in iterarray:
        if count>5:
            break
        model.fit_partial(train, epochs=1)
        auc_train= precision_at_k(model, train, None, k=10).mean()
        auc_test= precision_at_k(model, test, None, k=10).mean()
        print(f'Epoch: {e}, Train AUC={auc_train:.3f}, Test AUC={auc_test:.3f}')
        train_scores.append(auc_train)
        test_scores.append(auc_test)
        if auc_test > best:
            best_model = deepcopy(model)
            best = auc_test
        else:
            count += 1

    model= deepcopy(best_model)
    '''
    for i in iterarray:
        model.fit_partial(train, item_features=None,epochs=10, num_threads=1,verbose=False)
        #print(model)
        test_precision = precision_at_k(model, test, k=5).mean()
        test_patk.append(test_precision)
        print(test_precision)
    pass
    '''
    save_plot_to_png(iterarray,(test_scores,train_scores), ['test','train'],'test_train_auc_scores.png')
    
    # predict the best items for user id
    user_index = 1
    all_users_ids = [u for u in get_all_users()['id']]
    for user_index in range(len(all_users_ids)):
        export_best_items_for_user(model, user_index,all_users_ids[user_index])

    
def export_best_items_for_user(model, user_index, user_id):
    best_items = model.predict(user_index, np.arange(len(get_all_products()['id'])), num_threads=1)
    best_items_df = pd.DataFrame(best_items, columns=['probability'])
    best_items_df['product_id'] = get_all_products()['id']#np.arange(num_items)
    best_items_df['product_title'] = get_all_products()['title']

    best_items_df.sort_values(by='probability', ascending=False, inplace=True)
    best_items_df.to_csv(BASE_DATA_PATH + f'best_items_for_user_{user_id}.csv', index=False)
    pass

def export_best_items_for_all_users(model):
    # predict the best items for all users
    all_users = get_all_users()
    offset = len(get_all_products()['id']) - len(all_users)
    best_items = model.predict(np.arange(len(all_users['id'])), np.arange(len(get_all_products()['id'])), num_threads=1)
    print(best_items)
    best_items_df = pd.DataFrame(best_items, columns=['probability'])
    best_items_df['product_id'] = get_all_products()['id']
    best_items_df['product_title'] = get_all_products()['title']

    best_items_df.sort_values(by='probability', ascending=False, inplace=True)
    print(best_items_df)
    best_items_df.to_csv(BASE_DATA_PATH + 'best_items_for_all_users.csv', index=False)



def save_plot_to_png(iterarray,packs, labels, filename):
    sns.set_style('white')

    def plot_patk(iterarray, patk, k=5):
        plt.plot(iterarray, patk);
        plt.xlabel('Epochs', fontsize=24);
        plt.ylabel('p@{}'.format(k), fontsize=24);
        plt.xticks(fontsize=14);
        plt.yticks(fontsize=14);

    # Plot test on right
    ax = plt.subplot(1, 2, 2)
    fig = ax.get_figure()
    sns.despine(fig)
    i = 0
    for pack in packs:
        plot_patk(range(len(pack)), pack, k=5)
        i+=1
    plt.legend(labels=labels)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    
if __name__ == '__main__':
    main()