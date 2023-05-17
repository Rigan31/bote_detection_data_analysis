import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
import pickle

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import LabelEncoder




def KeyPress_MakeCluster(num_of_tables, num_of_runs):

    # user_id,duration,interArrivalTime,Type

    human_dataset = pd.read_csv('csv_files/raw_csv/KeyPress_human.csv')
    bot_dataset = pd.read_csv('csv_files/raw_csv/KeyPress_bot.csv')

    cols_to_normalize = ['duration', 'interArrivalTime']

    # calculate the mean and standard deviation of the columns to normalize
    bot_means = bot_dataset[cols_to_normalize].mean()
    bot_stds = bot_dataset[cols_to_normalize].std()

    human_means = human_dataset[cols_to_normalize].mean()
    human_stds = human_dataset[cols_to_normalize].std()

    # normalize the data
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    human_dataset[norm_cols] = (human_dataset[cols_to_normalize] - human_means) / human_stds
    human_dataset[norm_cols] = human_dataset[norm_cols].fillna(0)

    bot_dataset[norm_cols] = (bot_dataset[cols_to_normalize] - bot_means) / bot_stds
    bot_dataset[norm_cols] = bot_dataset[norm_cols].fillna(0)

    # human_dataset.to_csv("normalized/human_norm_keyPress.csv", index=False)
    # bot_dataset.to_csv("normalized/bot_norm_keyPress.csv", index=False)

    # print("after writing csv")

    # Split human dataset
    X_train_human, X_test_human = train_test_split(human_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_human)

    # Split bot dataset
    X_train_bot, X_test_bot = train_test_split(bot_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_bot)


    # apply K-Means clustering to the normalized data
    # kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])
    
    
    # save means, stds, and kmeans model to a file using pickle
    with open('models/kmeans_human_keyPress_model.pkl', 'wb') as f:
        pickle.dump((human_means, human_stds, kmeans_human), f)


    with open('models/kmeans_bot_keyPress_model.pkl', 'wb') as f:
        pickle.dump((bot_means, bot_stds, kmeans_bot), f)
    
    X_train_human['human_group'] = kmeans_human.labels_

    X_train_bot['bot_group'] = kmeans_bot.labels_


    # loop over the groups and write each one to a separate CSV file
    for group_num, group_df in X_train_human.groupby('human_group'):
        filename = f'groups/KeyPress/human/KPHuman_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)

    for group_num, group_df in X_train_bot.groupby('bot_group'):
        filename = f'groups/KeyPress/bot/KPBot_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)


    print("Key press make cluster done")



def KeyPressClusterDistance(client_data):

    # load the saved file and retrieve means, stds, and kmeans model
    with open('models/kmeans_human_keyPress_model.pkl', 'rb') as f:
        train_means_human, train_stds_human, kmeans_human = pickle.load(f)

    with open('models/kmeans_bot_keyPress_model.pkl', 'rb') as f:
        train_means_bot, train_stds_bot, kmeans_bot = pickle.load(f)


    cols_to_normalize = ['duration', 'interArrivalTime']
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    
    client_data[norm_cols] = (client_data[cols_to_normalize] - train_means_human) / train_stds_human
    client_data[norm_cols] = client_data[norm_cols].fillna(0)

    # Calculate the distance to each group
    distances_human = kmeans_human.transform(client_data[norm_cols])
    distances_bot = kmeans_bot.transform(client_data[norm_cols])


    # Print the distances    
    print("Key Press human distance: ", np.amin(distances_human))
    print("Key Press bot distance: ", np.amin(distances_bot))

    result = "Human" if np.amin(distances_human) < np.amin(distances_bot) else "Bot"

    print("Key Press Verdict: ", result)

    return (distances_human, distances_bot)


def ScrollingClusterDistance(client_data):

    # load the saved file and retrieve means, stds, and kmeans model
    with open('models/kmeans_human_scrolling_model.pkl', 'rb') as f:
        train_means_human, train_stds_human, kmeans_human = pickle.load(f)

    with open('models/kmeans_bot_scrolling_model.pkl', 'rb') as f:
        train_means_bot, train_stds_bot, kmeans_bot = pickle.load(f)


    cols_to_normalize = ['duration', 'distance', 'avgSpeed']
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    
    client_data[norm_cols] = (client_data[cols_to_normalize] - train_means_human) / train_stds_human
    client_data[norm_cols] = client_data[norm_cols].fillna(0)

    # Calculate the distance to each group
    distances_human = kmeans_human.transform(client_data[norm_cols])
    distances_bot = kmeans_bot.transform(client_data[norm_cols])


    # Print the distances    
    print("Scrolling human distance: ", np.amin(distances_human))
    print("Scrolling bot distance: ", np.amin(distances_bot))

    result = "Human" if np.amin(distances_human) < np.amin(distances_bot) else "Bot"

    print("Scrolling Verdict: ", result)

    return (distances_human, distances_bot)




def Scrolling_MakeCluster(num_of_tables, num_of_runs):

    # "id","user_id","time","duration","startY","endY","speedList"

    human_dataset = pd.read_csv('csv_files/raw_csv/Scrolling_human.csv')
    bot_dataset = pd.read_csv('csv_files/raw_csv/Scrolling_bot.csv')

    cols_to_normalize = ['duration', 'distance', 'avgSpeed']

    # calculate the mean and standard deviation of the columns to normalize
    bot_means = bot_dataset[cols_to_normalize].mean()
    bot_stds = bot_dataset[cols_to_normalize].std()

    human_means = human_dataset[cols_to_normalize].mean()
    human_stds = human_dataset[cols_to_normalize].std()

    # normalize the data
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    human_dataset[norm_cols] = (human_dataset[cols_to_normalize] - human_means) / human_stds
    human_dataset[norm_cols] = human_dataset[norm_cols].fillna(0)

    bot_dataset[norm_cols] = (bot_dataset[cols_to_normalize] - bot_means) / bot_stds
    bot_dataset[norm_cols] = bot_dataset[norm_cols].fillna(0)

    # human_dataset.to_csv("normalized/human_norm_keyPress.csv", index=False)
    # bot_dataset.to_csv("normalized/bot_norm_keyPress.csv", index=False)


    # Split human dataset
    X_train_human, X_test_human = train_test_split(human_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_human)

    # Split bot dataset
    X_train_bot, X_test_bot = train_test_split(bot_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_bot)


    # apply K-Means clustering to the normalized data
    # kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])
    
    
    # save means, stds, and kmeans model to a file using pickle
    with open('models/kmeans_human_scrolling_model.pkl', 'wb') as f:
        pickle.dump((human_means, human_stds, kmeans_human), f)


    with open('models/kmeans_bot_scrolling_model.pkl', 'wb') as f:
        pickle.dump((bot_means, bot_stds, kmeans_bot), f)
    
    X_train_human['human_group'] = kmeans_human.labels_

    X_train_bot['bot_group'] = kmeans_bot.labels_


    # loop over the groups and write each one to a separate CSV file
    for group_num, group_df in X_train_human.groupby('human_group'):
        filename = f'groups/Scrolling/human/SCHuman_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)

    for group_num, group_df in X_train_bot.groupby('bot_group'):
        filename = f'groups/Scrolling/bot/SCBot_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)


    print("Scrolling make cluster done")



def MouseClick_MakeCluster(num_of_tables, num_of_runs):

    # user_id,duration,Type

    human_dataset = pd.read_csv('csv_files/raw_csv/MouseClick_human.csv')
    bot_dataset = pd.read_csv('csv_files/raw_csv/MouseClick_bot.csv')

    cols_to_normalize = ['duration']

    # calculate the mean and standard deviation of the columns to normalize
    bot_means = bot_dataset[cols_to_normalize].mean()
    bot_stds = bot_dataset[cols_to_normalize].std()

    human_means = human_dataset[cols_to_normalize].mean()
    human_stds = human_dataset[cols_to_normalize].std()

    # normalize the data
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    human_dataset[norm_cols] = (human_dataset[cols_to_normalize] - human_means) / human_stds
    human_dataset[norm_cols] = human_dataset[norm_cols].fillna(0)

    bot_dataset[norm_cols] = (bot_dataset[cols_to_normalize] - bot_means) / bot_stds
    bot_dataset[norm_cols] = bot_dataset[norm_cols].fillna(0)

    # human_dataset.to_csv("normalized/human_norm_keyPress.csv", index=False)
    # bot_dataset.to_csv("normalized/bot_norm_keyPress.csv", index=False)


    # Split human dataset
    X_train_human, X_test_human = train_test_split(human_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_human)

    # Split bot dataset
    X_train_bot, X_test_bot = train_test_split(bot_dataset[norm_cols], test_size=0.1, random_state=42)

    # Train k-means on training set
    kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(X_train_bot)


    # apply K-Means clustering to the normalized data
    # kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])
    
    
    # save means, stds, and kmeans model to a file using pickle
    with open('models/kmeans_human_mouseClick_model.pkl', 'wb') as f:
        pickle.dump((human_means, human_stds, kmeans_human), f)


    with open('models/kmeans_bot_mouseClick_model.pkl', 'wb') as f:
        pickle.dump((bot_means, bot_stds, kmeans_bot), f)
    
    X_train_human['human_group'] = kmeans_human.labels_

    X_train_bot['bot_group'] = kmeans_bot.labels_


    # loop over the groups and write each one to a separate CSV file
    for group_num, group_df in X_train_human.groupby('human_group'):
        filename = f'groups/MouseClick/human/MCHuman_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)

    for group_num, group_df in X_train_bot.groupby('bot_group'):
        filename = f'groups/MouseClick/bot/MCBot_group_{group_num}.csv'
        group_df[cols_to_normalize].to_csv(filename, index=False)


    print("Mouse Click make cluster done")


def MouseClickClusterDistance(client_data):

    # load the saved file and retrieve means, stds, and kmeans model
    with open('models/kmeans_human_mouseClick_model.pkl', 'rb') as f:
        train_means_human, train_stds_human, kmeans_human = pickle.load(f)

    with open('models/kmeans_bot_mouseClick_model.pkl', 'rb') as f:
        train_means_bot, train_stds_bot, kmeans_bot = pickle.load(f)


    cols_to_normalize = ['duration']
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    
    client_data[norm_cols] = (client_data[cols_to_normalize] - train_means_human) / train_stds_human
    client_data[norm_cols] = client_data[norm_cols].fillna(0)

    # Calculate the distance to each group
    distances_human = kmeans_human.transform(client_data[norm_cols])
    distances_bot = kmeans_bot.transform(client_data[norm_cols])


    # Print the distances    
    print("Mouse Click human distance: ", np.amin(distances_human))
    print("Mouse Click bot distance: ", np.amin(distances_bot))

    result = "Human" if np.amin(distances_human) < np.amin(distances_bot) else "Bot"

    print("Mouse Click Verdict: ", result)

    return (distances_human, distances_bot)



def MouseMove_MakeCluster(num_of_tables, num_of_runs, random_state_human, random_state_bot):

    # Load data into a Pandas DataFrame
    # dataset = pd.read_csv('modified_mouseMoveTable.csv')

    human_dataset = pd.read_csv('csv_files/raw_csv/MouseMove_human.csv')
    bot_dataset = pd.read_csv('csv_files/raw_csv/MouseMove_bot.csv')


    # create a scaler object
    # scaler = MinMaxScaler()
    
    # cols_to_normalize = ['distance', 'duration', 'gradient', 'variance', 'velocity', 'displacement', 'efficiency', 'finalGradient', 'amplify']


    # replace 'inf' values with -1 in 'finalGradient' and 'amplify' columns
    human_dataset.loc[(human_dataset['finalGradient'] == np.inf) | (human_dataset['finalGradient'] == -np.inf), 'finalGradient'] = -1.0
    human_dataset.loc[(human_dataset['amplify'] == np.inf) | (human_dataset['amplify'] == -np.inf), 'amplify'] = 1.0

    human_dataset.loc[(human_dataset['finalGradient'] == -1.0), 'variance'] = 0



    bot_dataset.loc[(bot_dataset['finalGradient'] == np.inf) | (bot_dataset['finalGradient'] == -np.inf), 'finalGradient'] = -1.0
    bot_dataset.loc[(bot_dataset['amplify'] == np.inf) | (bot_dataset['amplify'] == -np.inf), 'amplify'] = 1.0

    bot_dataset.loc[(bot_dataset['finalGradient'] == -1.0), 'variance'] = 0

    bot_dataset = bot_dataset.loc[bot_dataset['finalGradient'] != -1.0]
    human_dataset = human_dataset.loc[human_dataset['finalGradient'] != -1.0]


    bot_dataset = bot_dataset.loc[bot_dataset['variance'] != -1.0]
    human_dataset = human_dataset.loc[human_dataset['variance'] != -1.0]

    bot_dataset = bot_dataset.loc[bot_dataset['distance'] != 0]
    human_dataset = human_dataset.loc[human_dataset['distance'] != 0]


    # bot_dataset['amplify'] = abs(bot_dataset['amplify'])
    # human_dataset['amplify'] = abs(human_dataset['amplify'])


    # columns to normalize
    # cols_to_normalize = ['distance', 'duration', 'gradient', 'variance', 'velocity', 'displacement', 'efficiency', 'finalGradient', 'amplify']

    cols_to_normalize = ['distance', 'displacement', 'variance', 'velocity', 'efficiency', 'amplify']

    # norm_cols = [col + '_normalized' for col in cols_to_normalize]

    combined_dataset = pd.concat([human_dataset, bot_dataset], ignore_index=True)
    combined_dataset, X_test = train_test_split(combined_dataset, test_size=0.1, random_state=random_state_human)


    kmeans_combined = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(combined_dataset[cols_to_normalize])


    combined_dataset['cluster_label'] = kmeans_combined.labels_

    # print("combined_dataset head: ", combined_dataset.head)


    models = []
    for cluster_label in range(num_of_tables):
        # Select instances belonging to the current cluster
        cluster_data = combined_dataset[combined_dataset['cluster_label'] == cluster_label]
        
        # Split data into features (X) and target (y)
        X = cluster_data[cols_to_normalize]
        y = cluster_data['Type']  # Replace 'target_column' with the actual target column name


        # Perform label encoding on the target variable
        label_encoder = LabelEncoder()
        # y = label_encoder.fit_transform(y)

        # Define the predefined mapping dictionary for encoding
        mapping = {"Human": 0, "Bot": 1}

        # Create a new list with the mapped values
        y_encoded = [mapping[label] for label in y]

        # Fit the label encoder on the encoded values
        label_encoder.fit(y_encoded)

        # Transform the encoded values
        y = label_encoder.transform(y_encoded)

        # Train linear regression
        linear_reg = LinearRegression()
        linear_reg.fit(X, y)
        
        # Train random forest
        random_forest = RandomForestRegressor()
        random_forest.fit(X, y)
        
        # Train SVM
        svm = SVR()
        svm.fit(X, y)
        
        # Store the trained models
        models.append({'linear_reg': linear_reg, 'random_forest': random_forest, 'svm': svm})

    return models, kmeans_combined, X_test




def Second_Resolver(models, kmeans_combined, num_of_tables, X_test):
    predictions = []
    cols_to_normalize = ['distance', 'displacement', 'variance', 'velocity', 'efficiency', 'amplify']

    test_cluster_labels = kmeans_combined.predict(X_test[cols_to_normalize])

    predictions = []  # List to store the predicted types

    linear_vote, rf_vote, svm_vote = 0, 0, 0
    batch_size = len(X_test)

    for i in range(len(X_test)):
        cluster_models = models[test_cluster_labels[i]]

        linear_reg = cluster_models['linear_reg']
        random_forest = cluster_models['random_forest']
        svm = cluster_models['svm']



        row_to_predict = pd.DataFrame(X_test.iloc[i]).T  # Extract the first row

        # print("row to predict: ", row_to_predict)

        # Reshape the row to match the expected input shape
        row_to_predict_reshaped = row_to_predict[cols_to_normalize].values.reshape(1, -1)

        # Predict the output using linear regression
        # linear_reg_pred = linear_reg.predict(row_to_predict_reshaped)

        # linear_reg_preds = linear_reg.predict(pd.DataFrame(X_test[i])[cols_to_normalize])
                
                
        linear_reg_preds = linear_reg.predict(row_to_predict[cols_to_normalize])
        random_forest_preds = random_forest.predict(row_to_predict[cols_to_normalize])
        svm_preds = svm.predict(row_to_predict[cols_to_normalize])

        # linear_reg_preds = 'Human' if linear_reg_preds < 0.5 else 'Bot'
        # random_forest_preds = 'Human' if random_forest_preds < 0.5 else 'Bot'
        # svm_preds = 'Human' if svm_preds < 0.5 else 'Bot'

        if linear_reg_preds > 0.5:
            linear_vote += 1
        if random_forest_preds > 0.5:
            rf_vote += 1
        if svm_preds > 0.5:
            svm_vote += 1

    
    linear_result, rf_result, svm_result = 'Human', 'Human', 'Human'
    if linear_vote > int(batch_size/2):
        linear_result = 'Bot'
    if rf_vote > int(batch_size/2):
        rf_result = 'Bot'
    if svm_vote > int(batch_size/2): 
        svm_result = 'Bot'

    
    
        # Store the predictions for the current cluster
    cluster_predictions = {'linear': linear_result,
                        'rf': rf_result,
                        'svm': svm_result}
    
    return cluster_predictions
    # predictions.append(cluster_predictions)




    for cluster_label in range(num_of_tables):
        # Retrieve the models for the current cluster
        cluster_models = models[cluster_label]
        
        # Get the corresponding models for linear regression, random forest, and SVM
        linear_reg = cluster_models['linear_reg']
        random_forest = cluster_models['random_forest']
        svm = cluster_models['svm']
        
        # Make predictions using the models
        linear_reg_preds = linear_reg.predict(X_test[cols_to_normalize])
        random_forest_preds = random_forest.predict(X_test[cols_to_normalize])
        svm_preds = svm.predict(X_test[cols_to_normalize])

        
        # Store the predictions for the current cluster
        cluster_predictions = {'linear_reg': linear_reg_preds,
                            'random_forest': random_forest_preds,
                            'svm': svm_preds}
        predictions.append(cluster_predictions)

    # Combine the predictions from all clusters:


    combined_predictions = pd.concat([pd.DataFrame(predictions[i]) for i in range(num_of_tables)], axis=1)

    print("combined_predctions head: ")
    print(combined_predictions.head)

    print("X_test")
    print(X_test)

    return X_test, combined_predictions

    # Now, the combined_predictions DataFrame contains the predictions made by the trained linear regression, random forest, and SVM models for each cluster on the test set. You can analyze and evaluate the performance of the models using appropriate metrics or further process the predictions as needed.



    # Split human dataset
    # human_dataset, X_test_human = train_test_split(human_dataset[norm_cols], test_size=0.1, random_state=42)
    # human_dataset, X_test_human = train_test_split(human_dataset, test_size=0.1, random_state=random_state_human)


    # Split bot dataset
    # bot_dataset, X_test_bot = train_test_split(bot_dataset[norm_cols], test_size=0.1, random_state=42)
    # bot_dataset, X_test_bot = train_test_split(bot_dataset, test_size=0.1, random_state=random_state_bot)


    # calculate the mean and standard deviation of the columns to normalize
    # bot_means = bot_dataset[cols_to_normalize].mean()
    # bot_stds = bot_dataset[cols_to_normalize].std()

    # human_means = human_dataset[cols_to_normalize].mean()
    # human_stds = human_dataset[cols_to_normalize].std()


    # # print("before std norm")
    # # normalize the data
    # human_dataset[norm_cols] = (human_dataset[cols_to_normalize] - human_means) / human_stds
    # human_dataset[norm_cols] = human_dataset[norm_cols].fillna(0)


    # bot_dataset[norm_cols] = (bot_dataset[cols_to_normalize] - bot_means) / bot_stds
    # bot_dataset[norm_cols] = bot_dataset[norm_cols].fillna(0)


    # =======================================================


    # Normalize the columns
    # normalized_data = bot_dataset[cols_to_normalize].copy()
    # # normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()

    # # Apply PCA to reduce the dimensions to 2
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(normalized_data)

    # # Create a scatter plot
    # plt.scatter(pca_result[:, 0], pca_result[:, 1])
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('Clusters Visualization')
    
    # # Set the limits for the x and y axes
    # plt.xlim([-5000, 5000])  # Set the desired limits for the x-axis
    # # plt.ylim([-5, 5])  # Set the desired limits for the y-axis

    
    # plt.show()

    # Train k-means on training set
    # kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # Train k-means on training set
    # kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])




    # ======================================================

    # bot_labels = kmeans_bot.labels_
    # human_labels = kmeans_human.labels_

    # # Normalize the columns for bot_dataset
    # normalized_bot_data = bot_dataset[cols_to_normalize].copy()
    # # normalized_bot_data = (normalized_bot_data - normalized_bot_data.mean()) / normalized_bot_data.std()

    # # Normalize the columns for human_dataset
    # normalized_human_data = human_dataset[cols_to_normalize].copy()
    # # normalized_human_data = (normalized_human_data - normalized_human_data.mean()) / normalized_human_data.std()

    # # Apply PCA to reduce the dimensions to 2 for bot_dataset
    # pca_bot = PCA(n_components=2)
    # pca_result_bot = pca_bot.fit_transform(normalized_bot_data)

    # # Apply PCA to reduce the dimensions to 2 for human_dataset
    # pca_human = PCA(n_components=2)
    # pca_result_human = pca_human.fit_transform(normalized_human_data)

    # # Create a figure with two subplots
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot for bot_dataset
    # axes[0].scatter(pca_result_bot[:, 0], pca_result_bot[:, 1], c=bot_labels, s=2)
    # axes[0].set_xlabel('Principal Component 1')
    # axes[0].set_ylabel('Principal Component 2')
    # axes[0].set_title('Bot Dataset Clusters')
    # axes[0].set_xlim([-3000, 5000])

    # # Plot for human_dataset
    # axes[1].scatter(pca_result_human[:, 0], pca_result_human[:, 1], c=human_labels, s=2)
    # axes[1].set_xlabel('Principal Component 1')
    # axes[1].set_ylabel('Principal Component 2')
    # axes[1].set_title('Human Dataset Clusters')
    # axes[1].set_xlim([-2000, 5000])

    # # Adjust spacing between subplots
    # plt.tight_layout()

    # # Display the figure
    # plt.show()



    # # ==========================================================

    # # Combine the bot and human datasets
    # combined_data = pd.concat([normalized_bot_data, normalized_human_data])

    # # Apply PCA to reduce the dimensions to 2 for combined data
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(combined_data)

    # # Create a figure
    # fig, ax = plt.subplots(figsize=(8, 6))

    # # Plot bot dataset in blue
    # ax.scatter(pca_result[:len(normalized_bot_data), 0], pca_result[:len(normalized_bot_data), 1], c='blue', label='Bot', s=1)

    # # Plot human dataset in green
    # ax.scatter(pca_result[len(normalized_bot_data):, 0], pca_result[len(normalized_bot_data):, 1], c='green', label='Human', s=1)

    # # Set labels and title
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # ax.set_title('Bot and Human Datasets')
    # ax.set_xlim([-2000, 3000])

    # # Set legend
    # ax.legend()

    # # Display the plot
    # plt.show()




    # ===========================================================


    # human_dataset.to_csv("csv_files/normalized/human_nor_mouseMove.csv", index=False)
    # bot_dataset.to_csv("csv_files/normalized/bot_norm_mouseMove.csv", index=False)


    
    
    # save means, stds, and kmeans model to a file using pickle
    # with open('models/kmeans_human_mouseMove_model.pkl', 'wb') as f:
    #     pickle.dump((human_means, human_stds, kmeans_human), f)


    # with open('models/kmeans_bot_mouseMove_model.pkl', 'wb') as f:
    #     pickle.dump((bot_means, bot_stds, kmeans_bot), f)
    
    # human_dataset['human_group'] = kmeans_human.labels_

    # bot_dataset['bot_group'] = kmeans_bot.labels_



    # # loop over the groups and write each one to a separate CSV file
    # for group_num, group_df in human_dataset.groupby('human_group'):
    #     filename = f'groups/MouseMove/human/MMHuman_group_{group_num}.csv'
    #     group_df[norm_cols].to_csv(filename, index=False)

    # for group_num, group_df in bot_dataset.groupby('bot_group'):
    #     filename = f'groups/MouseMove/bot/MMBot_group_{group_num}.csv'
    #     group_df[norm_cols].to_csv(filename, index=False)


    # print("after grouping data")

    # return (X_test_human, X_test_bot)



def MouseMoveClusterDistance(client_data):

    with open('models/kmeans_human_mouseMove_model.pkl', 'rb') as f:
        train_means_human, train_stds_human, kmeans_human = pickle.load(f)

    with open('models/kmeans_bot_mouseMove_model.pkl', 'rb') as f:
        train_means_bot, train_stds_bot, kmeans_bot = pickle.load(f)


    # print("Client data: ")
    # print(client_data)


    cols_to_normalize = ['distance', 'displacement', 'variance', 'velocity', 'efficiency', 'amplify']

    # cols_to_normalize = ['distance', 'duration', 'gradient', 'variance', 'velocity', 'displacement', 'efficiency', 'finalGradient', 'amplify']
    norm_cols = [col + '_normalized' for col in cols_to_normalize]
    

    # client_means = client_data[cols_to_normalize].mean()
    # client_stds = client_data[cols_to_normalize].std()

    # human_means = human_dataset[cols_to_normalize].mean()
    # human_stds = human_dataset[cols_to_normalize].std()


    client_data_human = client_data.copy()
    client_data_human[norm_cols] = (client_data[cols_to_normalize] - train_means_human) / train_stds_human
    client_data_human[norm_cols] = client_data_human[norm_cols].fillna(0)

    client_data_bot = client_data.copy()
    client_data_bot[norm_cols] = (client_data[cols_to_normalize] - train_means_bot) / train_stds_bot
    client_data_bot[norm_cols] = client_data_bot[norm_cols].fillna(0)

    distances_human = kmeans_human.transform(client_data_human[norm_cols])
    distances_bot = kmeans_bot.transform(client_data_bot[norm_cols])


    # print("human distance: ", np.mean(distances_human))
    # print("bot distance: ", np.mean(distances_bot))


    # result = "Human" if np.amin(distances_human) < np.amin(distances_bot) else "Bot"

    result = "Human" if np.mean(distances_human) < np.mean(distances_bot) else "Bot"

    return (distances_human, distances_bot)



def Resolver(client_data):

    mm_human_weight, mm_bot_weight = 0.7, 0.7
    mc_human_weight, mc_bot_weight = 0.1, 0.1
    sc_human_weight, sc_bot_weight = 0.1, 0.1
    kp_human_weight, kp_bot_weight = 0.1, 0.1

    mm_human, mm_bot = MouseMoveClusterDistance(client_data)

    # mc_human, mc_bot = MouseClickClusterDistance(client_data)

    # sc_human, sc_bot = ScrollingClusterDistance(client_data)

    # kp_human, kp_bot = KeyPressClusterDistance(client_data)

    # human_score = mm_human * mm_human_weight + mc_human *mc_human_weight + sc_human * sc_human_weight + kp_human * kp_human_weight

    # bot_score = mm_bot * mm_bot_weight + mc_bot * mc_bot_weight + sc_bot * sc_bot_weight + kp_bot * kp_bot_weight

    # print("human score: ", human_score)
    # print("bot score: ", bot_score)

    # result = "Human" if human_score < bot_score else "Bot"

    mm_human = np.mean(mm_human)
    mm_bot = np.mean(mm_bot)

    result = "Human" if mm_human < mm_bot else "Bot"

    # print("Detection: ", result)

    return result


if __name__ == '__main__':


    # num_clusters = 50, num_runs = 20, batch_size = 1, TPR = 0.67, TNR = 0.71
    # num_clusters = 50, num_runs = 20, batch_size = 3, TPR = 0.88, TNR = 0.704
    # num_clusters = 50, num_runs = 20, batch_size = 5, TPR = 0.9, TNR = 0.746
    # num_clusters = 50, num_runs = 20, batch_size = 8, TPR = 1.0, TNR = 0.75
    # num_clusters = 50, num_runs = 20, batch_size = 12, TPR = 1.0, TNR = 0.78
    # num_clusters = 50, num_runs = 20, batch_size = 18, TPR = 1.0, TNR = 0.94


    # num_clusters = 60, num_runs = 20, batch_size = 5, TPR = 1.0, TNR = 0.143

    # Open a file in write mode
    # file = open('output/output2.txt', 'w')

    

    maxi, best_k = 0, 0
    best_TPR, best_TNR = 0, 0
    # for k in range(20, 120, 5):
    #     total_TPR, total_TNR, total_FPR, total_FNR = 0.0, 0.0, 0.0, 0.0
    #     for i in range(30, 45):
    #         for j in range(30, 45):
    # X_test_human, X_test_bot = MouseMove_MakeCluster(k, 20, i, j)


    # file_linear = open('output/output6_linear.txt', 'w')
    # file_rf = open('output/output6_rf.txt', 'w')
    # file_svm = open('output/output6_svm.txt', 'w')

    batch_list = [1, 3, 5, 8, 10, 15, 20, 25, 30]


    for bb in batch_list:
        batch_size = bb
        print("for batch_size: ", batch_size)
        
        for k in range(35, 40, 5):

            total_linear_TPR, total_linear_TNR, total_linear_FPR, total_linear_FNR = 0.0, 0.0, 0.0, 0.0
            total_rf_TPR, total_rf_TNR, total_rf_FPR, total_rf_FNR = 0.0, 0.0, 0.0, 0.0
            total_svm_TPR, total_svm_TNR, total_svm_FPR, total_svm_FNR = 0.0, 0.0, 0.0, 0.0

            num_of_tables = k

            for i in range(30, 34):
                for j in range(30, 34):

                    linear_TP, linear_TN, linear_FP, linear_FN = 0, 0, 0, 0
                    rf_TP, rf_TN, rf_FP, rf_FN = 0, 0, 0, 0
                    svm_TP, svm_TN, svm_FP, svm_FN = 0, 0, 0, 0

                    models, kmeans_combined, X_test = MouseMove_MakeCluster(num_of_tables, 20, i, j)

                    X_test_human = X_test[X_test['Type'] == 'Human']
                    X_test_bot = X_test[X_test['Type'] == 'Bot']

                    # assume evaluation function is called 'evaluate'
                    num_human = len(X_test_human) // batch_size
                    num_bot = len(X_test_bot) // batch_size

                    # num_entity = len(X_test)//batch_size


                    
                    # print("X.human: ", X_test_human.shape)
                    # print("X.bot: ", X_test_bot.shape)

                    # print("num_human: ", num_human)
                    # print("num_bot: ", num_bot)

                    # print("Human Data: ")

                    for i in range(num_human):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        batch = X_test_human[start_idx:end_idx]
                        result = Second_Resolver(models, kmeans_combined, num_of_tables, batch)
                        
                        if(result['linear'] == 'Human'):
                            linear_TN += 1
                        else:
                            linear_FP += 1


                        if(result['rf'] == 'Human'):
                            rf_TN += 1
                        else:
                            rf_FP += 1


                        if(result['svm'] == 'Human'):
                            svm_TN += 1
                        else:
                            svm_FP += 1

                        # print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")
                        
                    # # handle the last batch separately in case it has less than 10 rows
                    if len(X_test_human) % batch_size > 0:
                        start_idx = num_human * batch_size
                        end_idx = len(X_test_human)
                        batch = X_test_human[start_idx:end_idx]
                        result = Second_Resolver(models, kmeans_combined, num_of_tables, batch)
                        # print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")

                        if(result['linear'] == 'Human'):
                            linear_TN += 1
                        else:
                            linear_FP += 1

                        if(result['rf'] == 'Human'):
                            rf_TN += 1
                        else:
                            rf_FP += 1

                        if(result['svm'] == 'Human'):
                            svm_TN += 1
                        else:
                            svm_FP += 1



                    # print("\nBot Data: \n")

                    for i in range(num_bot):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        batch = X_test_bot[start_idx:end_idx]
                        result = Second_Resolver(models, kmeans_combined, num_of_tables, batch)
                        # print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")

                        if result['linear'] == 'Bot':
                            linear_TP += 1
                        else:
                            linear_FN += 1


                        if result['rf'] == 'Bot':
                            rf_TP += 1
                        else:
                            rf_FN += 1

                        if result['svm'] == 'Bot':
                            svm_TP += 1
                        else:
                            svm_FN += 1
                        
                    # handle the last batch separately in case it has less than 10 rows
                    if len(X_test_bot) % batch_size > 0:
                        start_idx = num_bot * batch_size
                        end_idx = len(X_test_bot)
                        batch = X_test_bot[start_idx:end_idx]
                        result = Second_Resolver(models, kmeans_combined, num_of_tables, batch)
                        # print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")

                        if result['linear'] == 'Bot':
                            linear_TP += 1
                        else:
                            linear_FN += 1


                        if result['rf'] == 'Bot':
                            rf_TP += 1
                        else:
                            rf_FN += 1

                        if result['svm'] == 'Bot':
                            svm_TP += 1
                        else:
                            svm_FN += 1


                    LINEAR_TPR = linear_TP / (linear_TP + linear_FN)

                    LINEAR_TNR = linear_TN / (linear_TN + linear_FP)

                    LINEAR_FPR = linear_FP / (linear_FP + linear_TN)

                    LINEAR_FNR = linear_FN / (linear_TP + linear_FN)

                    total_linear_TPR += LINEAR_TPR
                    total_linear_TNR += LINEAR_TNR
                    total_linear_FPR += LINEAR_FPR
                    total_linear_FNR += LINEAR_FNR


                    RF_TPR = rf_TP / (rf_TP + rf_FN)

                    RF_TNR = rf_TN / (rf_TN + rf_FP)

                    RF_FPR = rf_FP / (rf_FP + rf_TN)

                    RF_FNR = rf_FN / (rf_TP + rf_FN)

                    total_rf_TPR += RF_TPR
                    total_rf_TNR += RF_TNR
                    total_rf_FPR += RF_FPR
                    total_rf_FNR += RF_FNR


                    SVM_TPR = svm_TP / (svm_TP + svm_FN)

                    SVM_TNR = svm_TN / (svm_TN + svm_FP)

                    SVM_FPR = svm_FP / (svm_FP + svm_TN)

                    SVM_FNR = svm_FN / (svm_TP + svm_FN)

                    total_svm_TPR += SVM_TPR
                    total_svm_TNR += SVM_TNR
                    total_svm_FPR += SVM_FPR
                    total_svm_FNR += SVM_FNR



            # total_TPR += TPR
            # total_TNR += TNR
            # total_FPR += FPR
            # total_FNR += FNR

            total_linear_TPR /= 16
            total_linear_TNR /= 16
            total_linear_FPR /= 16
            total_linear_FNR /= 16

            total_rf_TPR /= 16
            total_rf_TNR /= 16
            total_rf_FPR /= 16
            total_rf_FNR /= 16

            total_svm_TPR /= 16
            total_svm_TNR /= 16
            total_svm_FPR /= 16
            total_svm_FNR /= 16

            # Write content to the file
            # file_linear.write(f"for k = {k}\n")
            # file_linear.write(f"TPR: {total_linear_TPR}\n")
            # file_linear.write(f"TNR: {total_linear_TNR}\n")
            # file_linear.write(f"FPR: {total_linear_FPR}\n")
            # file_linear.write(f"FNR: {total_linear_FNR}\n\n")


            # file_rf.write(f"for k = {k}\n")
            # file_rf.write(f"TPR: {total_rf_TPR}\n")
            # file_rf.write(f"TNR: {total_rf_TNR}\n")
            # file_rf.write(f"FPR: {total_rf_FPR}\n")
            # file_rf.write(f"FNR: {total_rf_FNR}\n\n")


            # file_svm.write(f"for k = {k}\n")
            # file_svm.write(f"TPR: {total_svm_TPR}\n")
            # file_svm.write(f"TNR: {total_svm_TNR}\n")
            # file_svm.write(f"FPR: {total_svm_FPR}\n")
            # file_svm.write(f"FNR: {total_svm_FNR}\n\n")


            print("for k = ", k)
            print("Linear algo:")
            print(f"TPR: {total_linear_TPR}")
            print(f"TNR: {total_linear_TNR}")
            print(f"FPR: {total_linear_FPR}")
            print(f"FNR: {total_linear_FNR}")

            print("Ranfom Forest algo:")
            print(f"TPR: {total_rf_TPR}")
            print(f"TNR: {total_rf_TNR}")
            print(f"FPR: {total_rf_FPR}")
            print(f"FNR: {total_rf_FNR}")

            print("SVM algo:")
            print(f"TPR: {total_svm_TPR}")
            print(f"TNR: {total_svm_TNR}")
            print(f"FPR: {total_svm_FPR}")
            print(f"FNR: {total_svm_FNR}")
            print("\n")

               

    # total_TPR /= 225
    # total_TNR /= 225
    # total_FPR /= 225
    # total_FNR /= 225

    # # Write content to the file
    # file.write(f"for k = {k}\n")
    # file.write(f"TPR: {total_TPR}\n")
    # file.write(f"TNR: {total_TNR}\n")
    # file.write(f"FPR: {total_FPR}\n")
    # file.write(f"FNR: {total_FNR}\n\n")

    #     # Close the file


        
    # print(f"for k = {k}")
    # print(f"TPR: {total_TPR}")
    # print(f"TNR: {total_TNR}")
    # print(f"FPR: {total_FPR}")
    # print(f"FNR: {total_FNR}")

    # if maxi < (total_TPR + total_TNR):
    #     maxi = total_TPR + total_TNR
    #     best_k = k
    #     best_TPR = total_TPR
    #     best_TNR = total_TNR

    # file.close()


    


    



