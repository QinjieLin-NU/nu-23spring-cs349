import json 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_ratings(folder, category):
    training_file, testing_file = f"{folder}/{category}_Ratings_training.csv",  f"{folder}/{category}_Ratings_test.csv"
    training_ratings, testing_ratings = pd.read_csv(training_file), pd.read_csv(testing_file)
    return training_ratings, testing_ratings

def load_reviews(folder, category):
    training_file, testing_file = f"{folder}/{category}_Reviews_training.json",  f"{folder}/{category}_Reviews_test.json"
    training_reviews, testing_reviews = pd.read_json(training_file,lines=True), pd.read_json(testing_file,lines=True)
    return training_reviews, testing_reviews

def devide(trn_ratings, tst_ratings, trn_reviews, tst_reviews):
    print("deviding...")
    # megre 
    trn_df = trn_ratings.merge(trn_reviews, on=['asin', 'reviewerID', 'unixReviewTime', "overall"])
    tst_df = tst_ratings.merge(tst_reviews, on=['asin', 'reviewerID', 'unixReviewTime'])
    # contact
    total_df = pd.concat([trn_df, tst_df]).reset_index(drop=True)
    # get awesomenss 
    t_list = [4.3, 4.4, 4.5, 4.6, 4.7, 3.8, 3.9, 4.0, 4.1, 4.2, 3.5, 3.6, 3.7, 4.8, 4.9, 5.0 ]
    for t in t_list:
        awesome_threshold = round(t,3)
        total_awesomeness = total_df.groupby('asin')\
            .apply(lambda x: 0.0 if x['verified'].sum() == 0 else (x['overall'] * x['verified']).sum() / x['verified'].sum())\
            .rename('awesomeness')\
            .reset_index()
        total_awesomeness['awesomeness'] = total_awesomeness['awesomeness'].apply(lambda x: 1 if x > awesome_threshold else 0)
        ratio_ones = total_awesomeness['awesomeness'].value_counts(normalize=True)[1]
        print(f"awesome_threshold: {awesome_threshold}, ratio of awesome product: {ratio_ones}")
        if 0.45 < ratio_ones < 0.55:
            print("successfully found balance")
            break
    # drop overall and merge to total df 
    merged_df = total_df.drop('overall', axis=1).merge(total_awesomeness, on=['asin'])
    # devide
    train_data = merged_df.sample(frac=0.5, random_state=42)
    test_data_1 = merged_df.drop(train_data.index).sample(frac=1/3, random_state=42)
    test_data_2 = merged_df.drop(train_data.index).drop(test_data_1.index).sample(frac=1/3, random_state=42)
    test_data_3 = merged_df.drop(train_data.index).drop(test_data_1.index).drop(test_data_2.index)
    return train_data, test_data_1, test_data_2, test_data_3
    
def plot_single(ax_n, input_df, label):
    ax_n.hist(list(input_df['awesomeness']), bins=2, edgecolor='black')
    ax_n.set_xlabel('Awesomeness')
    ax_n.set_ylabel('Number of Products')
    ax_n.set_title(f'Distribution of Product Awesomeness in {label}')

def plot_awssomeness(train0, test1, test2, test3, data_category):
    print("plotting...")
    fig, ((ax0, ax1),(ax2, ax3)) = plt.subplots(2,2, figsize=(15, 10))
    plot_single(ax0, train0, f"train of {data_category}")
    plot_single(ax1, test1, f"test1 of {data_category}")
    plot_single(ax2, test2, f"test2 of {data_category}")
    plot_single(ax3, test3, f"test3 of {data_category}")
    # save the plot
    plt.savefig(f'./plots/{data_category}.png')
    return

def save_data(train0, test1, test2, test3, data_category):
    print("saving...")
    train0.to_json(f"./devided_dataset/{data_category}_train.json") 
    test1.to_json(f"./devided_dataset/{data_category}_test1.json") 
    test2.to_json(f"./devided_dataset/{data_category}_test2.json") 
    test3.to_json(f"./devided_dataset/{data_category}_test3.json") 
    return 
    
if __name__=="__main__":
    category = ["CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Grocery_and_Gourmet_Food", "Sports_and_Outdoors", "Toys_and_Games", "Automotive"]
    for c in category:
        print(f"processing {c}...")
        trn_ratings, tst_ratings = load_ratings("./ML_datasets/ML_datasets", c)
        trn_reviews, tst_reviews = load_reviews("./ML_datasets/ML_datasets", c)
        train_data, test_data_1, test_data_2, test_data_3 = devide(trn_ratings, tst_ratings, trn_reviews, tst_reviews)
        plot_awssomeness(train_data, test_data_1, test_data_2, test_data_3, c)
        save_data(train_data, test_data_1, test_data_2, test_data_3, c)
    
    