import pandas as pd
import os 

# category = ["CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Grocery_and_Gourmet_Food", "Sports_and_Outdoors", "Toys_and_Games", "Automotive"]
category = ["CDs_and_Vinyl", "Grocery_and_Gourmet_Food",  "Toys_and_Games"]
date_type = ['train', 'test1', 'test2', 'test3']

# for c in category:
#     for t in date_type:
#         filename = f"./devided_dataset/{c}_{t}.json"
#         original_df = pd.read_json(filename)
#         os.makedirs(f"./devided_dataset_v2/{c}", exist_ok=True)
#         os.makedirs(f"./devided_dataset_v2/{c}/{t}", exist_ok=True)
#         if 'test' not in filename:
#             print("="*10, "training", "="*10)
#             featured_filename = f"./devided_dataset_v2/{c}/{t}/review_training.json"
#             label_filename = f"./devided_dataset_v2/{c}/{t}/product_training.json"
#             features = original_df.drop("awesomeness", axis=1)
#             awesomeness_df = original_df.loc[:, ['asin', 'awesomeness']]
#             labels = awesomeness_df.groupby('asin').agg({'awesomeness': 'mean'}).reset_index()
#             features.to_json(featured_filename)
#             labels.to_json(label_filename)
#         else:
#             print("="*10, "testing", "="*10)
#             featured_filename = f"./devided_dataset_v2/{c}/{t}/review_test.json"
#             label_filename = f"./devided_dataset_v2/{c}/{t}/product_labels.json"
#             new_label_filename = f"./devided_dataset_v2/{c}/{t}/product_test.json"
#             features = original_df.drop("awesomeness", axis=1)
#             awesomeness_df = original_df.loc[:, ['asin', 'awesomeness']]
#             labels = awesomeness_df.groupby('asin').agg({'awesomeness': 'mean'}).reset_index()
#             new_labels = labels.copy()
#             new_labels = new_labels.drop("awesomeness", axis=1)
#             features.to_json(featured_filename)
#             # labels.to_json(label_filename)
#             new_labels.to_json(new_label_filename)
            

for c in category:
    for t in date_type:
        filename = f"./devided_dataset/{c}_{t}.json"
        original_df = pd.read_json(filename)
        os.makedirs(f"./devided_dataset_v2_with_labels/{c}", exist_ok=True)
        os.makedirs(f"./devided_dataset_v2_with_labels/{c}/{t}", exist_ok=True)
        if 'test' not in filename:
            print("="*10, "training", "="*10)
            featured_filename = f"./devided_dataset_v2_with_labels/{c}/{t}/review_training.json"
            label_filename = f"./devided_dataset_v2_with_labels/{c}/{t}/product_training.json"
            features = original_df.drop("awesomeness", axis=1)
            awesomeness_df = original_df.loc[:, ['asin', 'awesomeness']]
            labels = awesomeness_df.groupby('asin').agg({'awesomeness': 'mean'}).reset_index()
            features.to_json(featured_filename)
            labels.to_json(label_filename)
        else:
            print("="*10, "testing", "="*10)
            featured_filename = f"./devided_dataset_v2_with_labels/{c}/{t}/review_test.json"
            label_filename = f"./devided_dataset_v2_with_labels/{c}/{t}/product_labels.json"
            new_label_filename = f"./devided_dataset_v2_with_labels/{c}/{t}/product_test.json"
            features = original_df.drop("awesomeness", axis=1)
            awesomeness_df = original_df.loc[:, ['asin', 'awesomeness']]
            labels = awesomeness_df.groupby('asin').agg({'awesomeness': 'mean'}).reset_index()
            new_labels = labels.copy()
            new_labels = new_labels.drop("awesomeness", axis=1)
            features.to_json(featured_filename)
            labels.to_json(label_filename)
            new_labels.to_json(new_label_filename)