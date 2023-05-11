import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
import pickle


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



def MouseMove_MakeCluster(num_of_tables, num_of_runs):

    # Load data into a Pandas DataFrame
    # dataset = pd.read_csv('modified_mouseMoveTable.csv')

    human_dataset = pd.read_csv('csv_files/raw_csv/MouseMove_human.csv')
    bot_dataset = pd.read_csv('csv_files/raw_csv/MouseMove_bot.csv')


    # dataset['Type'] = dataset.apply(lambda row: 'Bot' if (row['user_id'] > 59) else 'Human', axis=1)

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


    # columns to normalize
    # cols_to_normalize = ['distance', 'duration', 'gradient', 'variance', 'velocity', 'displacement', 'efficiency', 'finalGradient', 'amplify']

    cols_to_normalize = ['distance', 'displacement', 'variance', 'velocity', 'efficiency']

    norm_cols = [col + '_normalized' for col in cols_to_normalize]


    # bot_dataset = dataset.loc[dataset['user_id'] > 59]
    # human_dataset = dataset.loc[dataset['user_id'] <= 59]



    # Split human dataset
    # human_dataset, X_test_human = train_test_split(human_dataset[norm_cols], test_size=0.1, random_state=42)
    human_dataset, X_test_human = train_test_split(human_dataset, test_size=0.1, random_state=42)


    # Split bot dataset
    # bot_dataset, X_test_bot = train_test_split(bot_dataset[norm_cols], test_size=0.1, random_state=42)
    bot_dataset, X_test_bot = train_test_split(bot_dataset, test_size=0.1, random_state=42)


    # calculate the mean and standard deviation of the columns to normalize
    bot_means = bot_dataset[cols_to_normalize].mean()
    bot_stds = bot_dataset[cols_to_normalize].std()

    human_means = human_dataset[cols_to_normalize].mean()
    human_stds = human_dataset[cols_to_normalize].std()


    print("before std norm")
    # normalize the data
    human_dataset[norm_cols] = (human_dataset[cols_to_normalize] - human_means) / human_stds
    human_dataset[norm_cols] = human_dataset[norm_cols].fillna(0)


    bot_dataset[norm_cols] = (bot_dataset[cols_to_normalize] - bot_means) / bot_stds
    bot_dataset[norm_cols] = bot_dataset[norm_cols].fillna(0)


    # human_dataset.to_csv("csv_files/normalized/human_nor_mouseMove.csv", index=False)
    # bot_dataset.to_csv("csv_files/normalized/bot_norm_mouseMove.csv", index=False)


    # Train k-means on training set
    kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # Train k-means on training set
    kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])



    # apply K-Means clustering to the normalized data
    # kmeans_human = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(human_dataset[norm_cols])

    # kmeans_bot = KMeans(n_clusters=num_of_tables, n_init=num_of_runs, random_state=0).fit(bot_dataset[norm_cols])
    

    # models/kmeans_bot_mouseClick_model.pkl

    
    # save means, stds, and kmeans model to a file using pickle
    with open('models/kmeans_human_mouseMove_model.pkl', 'wb') as f:
        pickle.dump((human_means, human_stds, kmeans_human), f)


    with open('models/kmeans_bot_mouseMove_model.pkl', 'wb') as f:
        pickle.dump((bot_means, bot_stds, kmeans_bot), f)


    # save the fitted model to a file using pickle
    # with open('kmeans_model.pkl', 'wb') as f:
    #     pickle.dump(kmeans, f)
    
    human_dataset['human_group'] = kmeans_human.labels_

    bot_dataset['bot_group'] = kmeans_bot.labels_



    # loop over the groups and write each one to a separate CSV file
    for group_num, group_df in human_dataset.groupby('human_group'):
        filename = f'groups/MouseMove/human/MMHuman_group_{group_num}.csv'
        group_df[norm_cols].to_csv(filename, index=False)

    for group_num, group_df in bot_dataset.groupby('bot_group'):
        filename = f'groups/MouseMove/bot/MMBot_group_{group_num}.csv'
        group_df[norm_cols].to_csv(filename, index=False)


    print("after grouping data")

    return (X_test_human, X_test_bot)



def MouseMoveClusterDistance(client_data):

    # load the saved model from a file
    # with open('kmeans_model.pkl', 'rb') as f:
    #     kmeans = pickle.load(f)

    # load the saved file and retrieve means, stds, and kmeans model
    with open('models/kmeans_human_mouseMove_model.pkl', 'rb') as f:
        train_means_human, train_stds_human, kmeans_human = pickle.load(f)

    with open('models/kmeans_bot_mouseMove_model.pkl', 'rb') as f:
        train_means_bot, train_stds_bot, kmeans_bot = pickle.load(f)


    # print("Client data: ")
    # print(client_data)


    cols_to_normalize = ['distance', 'displacement', 'variance', 'velocity', 'efficiency']

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


    # client_data = client_data.copy()
    # client_data.loc[:, norm_cols] = (client_data.loc[:, cols_to_normalize] - client_means) / client_stds
    # client_data.loc[:, norm_cols] = client_data.loc[:, norm_cols].fillna(0)


    # print("client data: ", client_data)


    # client_data.loc[:, norm_cols] = (client_data.loc[:, cols_to_normalize] - train_means_human) / train_stds_human
    # client_data.loc[:, norm_cols] = client_data.loc[:, norm_cols].fillna(0)

    # client_data[norm_cols] = (client_data[cols_to_normalize] - train_means_human) / train_stds_human
    # client_data[norm_cols] = client_data[norm_cols].fillna(0)


    # Normalize the new data point
    # new_point_norm = (client_data[cols_to_normalize] - train_means) / train_stds

    # Calculate the distance to each group
    # distances_human = kmeans_human.transform(client_data[norm_cols])
    # distances_bot = kmeans_bot.transform(client_data[norm_cols])

    # distances_human = kmeans_human.transform(client_data_human[norm_cols])
    # distances_bot = kmeans_bot.transform(client_data_bot[norm_cols])


    distances_human = kmeans_human.transform(client_data_human[norm_cols])
    distances_bot = kmeans_bot.transform(client_data_bot[norm_cols])


    # predict the clusters of the test set
    # client_data['cluster'] = kmeans.predict(client_data[norm_cols])

    # print the predicted clusters for the test set
    # print(client_data['cluster'])

    # Print the distances    
    # print("human distance: ", np.amin(distances_human))
    # print("bot distance: ", np.amin(distances_bot))

    print("human distance: ", np.mean(distances_human))
    print("bot distance: ", np.mean(distances_bot*0.9))


    # result = "Human" if np.amin(distances_human) < np.amin(distances_bot) else "Bot"

    result = "Human" if np.mean(distances_human) < np.mean(distances_bot) else "Bot"


    # print("Verdict: ", result)

    return (distances_human, distances_bot*0.9)



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






# numpy 2d array
keyPressDuration = np.array([])
print(keyPressDuration.shape)

def findDisplacement(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def findGradient(x1, y1, x2, y2):
    return np.array((y2- y1)/(x2 - x1))


def drawBarChart(X, Y, xLabel, yLabel, title, gapBetween):
    fig, ax = plt.subplots()

    # Plot proportion as bars
    ax.bar(X, Y, width=gapBetween*0.8, align='edge')

    # Set x-axis tick labels to the middle coordinate of each bar
    # ax.set_xticks(X)
    # ax.set_xticklabels(X)

    # Set axis labels and title
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    # ax.set_title(title)
    # save image
    plt.savefig('./../data/' + title + '.png')


def MouseMoveTableData():
    dataset = pd.read_csv('./../MouseMoveTable.csv')
    print(dataset)

    # add new column to the dataset which will store the velocity of the mouse = distance/duration
    dataset['velocity'] = dataset['distance'] / dataset['duration']
    # replace NaN values with 0
    dataset['velocity'] = dataset['velocity'].fillna(0)
    dataset['velocity'] = dataset['velocity']*1000.0 #convert to px/sec
    print(dataset)

    #####################################Displacement vs Proportion############################################

    #find displacement
    dataset['displacement'] = findDisplacement(dataset['startX'], dataset['startY'], dataset['endX'], dataset['endY'])

    maxiDisplacement = min(dataset['displacement'].max(), 7500)
    gapBetween = 100
    displacementRange = []
    proportion = []
    for i in range(0, int(maxiDisplacement / gapBetween) + 1):
        displacementRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['displacement'] >= i * gapBetween) & (dataset['displacement'] < (i + 1) * gapBetween)]['displacement'].count() / dataset['displacement'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    displacementRange = np.array(displacementRange)
    proportion = np.array(proportion)

    bar_middle = [int(x + gapBetween / 2) for x in displacementRange[:-1]]

    drawBarChart(displacementRange, proportion, 'Displacement Range', 'Proportion', 'Displacement vs Proportion', gapBetween)

    #######################################################################################################

    #####################################Distance vs Proportion############################################

    maxiDistance = min(dataset['distance'].max(), 4000)
    gapBetween = 100
    distanceRange = []
    proportion = []
    for i in range(0, int(maxiDistance / gapBetween) + 1):
        distanceRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['distance'] >= i * gapBetween) & (dataset['distance'] < (i + 1) * gapBetween)]['distance'].count() / dataset['distance'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    
    distanceRange = np.array(distanceRange)
    proportion = np.array(proportion)

    bar_middle = [int(x + gapBetween / 2) for x in distanceRange[:-1]]

    drawBarChart(distanceRange, proportion, 'Distance Range', 'Proportion', 'Distance vs Proportion', gapBetween)
    #######################################################################################################


    #####################################Velocity vs Proportion############################################

    maxiVelocity = min(dataset['velocity'].max(), 20000)
    print('maxiVelocity: ', maxiVelocity)
    gapBetween = 500
    velocityRange = []
    proportion = []
    for i in range(0, int(maxiVelocity / gapBetween) + 1):
        velocityRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['velocity'] >= i * gapBetween) & (dataset['velocity'] < (i + 1) * gapBetween)]['velocity'].count() / dataset['velocity'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    velocityRange = np.array(velocityRange)
    proportion = np.array(proportion)

    bar_middle = [int(x + gapBetween / 2) for x in velocityRange[:-1]]

    drawBarChart(velocityRange, proportion, 'Velocity Range', 'Proportion', 'Velocity vs Proportion', gapBetween)

    ####################################################################



    ##################################### Efficiency vs Proportion############################################
    #find efficiency
    dataset['efficiency'] = dataset['displacement'] / dataset['distance']
    # remove inf row
    dataset = dataset[dataset['efficiency'] != np.inf]
    maxEfficiency = 1.0
    print(maxEfficiency)
    gapBetween = 0.05
    efficiencyRange = []
    proportion = []
    for i in range(0, int(maxEfficiency / gapBetween) + 1):
        efficiencyRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['efficiency'] >= i * gapBetween) & (dataset['efficiency'] < (i + 1) * gapBetween)]['efficiency'].count() / dataset['efficiency'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    efficiencyRange = np.array(efficiencyRange)
    proportion = np.array(proportion)
    drawBarChart(efficiencyRange, proportion, 'Efficiency Range', 'Proportion', 'Efficiency vs Proportion', gapBetween)
    #############################################################################################################

    ##################################### Duration vs Proportion############################################
    maxiDuration = min(dataset['duration'].max(), 2000)
    gapBetween = 100
    durationRange = []
    proportion = []
    for i in range(0, int(maxiDuration / gapBetween) + 1):
        durationRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['duration'] >= i * gapBetween) & (dataset['duration'] < (i + 1) * gapBetween)]['duration'].count() / dataset['duration'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    durationRange = np.array(durationRange)
    proportion = np.array(proportion)
    drawBarChart(durationRange, proportion, 'Duration Range', 'Proportion', 'Duration vs Proportion', gapBetween)
    #############################################################################################################


    ########################################### Variance vs Efficiency ########################################
    # plot the variance vs efficiency
    print(dataset['efficiency'])
    print(dataset['variance'])

    #remove the row containing Nan
    dataset = dataset[dataset['variance'].notna()]
    print(dataset['efficiency'].max())

    # clear plt
    plt.clf()
    # reduce the scatter point size
    # plt.rcParams['figure.figsize'] = (10, 10)
    plt.scatter(dataset['efficiency'], dataset['variance'], s=5)
    plt.xlabel('Efficiency')
    plt.ylabel('Variance')
    plt.savefig('./../data/Variance vs Efficiency.png')


    ########################################### Average gradient vs start to end gradient######################
    dataset['finalGradient'] = findGradient(dataset['startX'], dataset['startY'], dataset['endX'], dataset['endY']) 
    dataset['amplify'] = dataset['finalGradient'] / dataset['gradient']
    # remove inf row
    dataset = dataset[dataset['amplify'] != np.inf]
    maxEfficiency = min(dataset['amplify'].max(), 6)
    print(maxEfficiency)
    gapBetween = 0.05
    efficiencyRange = []
    proportion = []
    for i in range(0, int(maxEfficiency / gapBetween) + 1):
        efficiencyRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['amplify'] >= i * gapBetween) & (dataset['amplify'] < (i + 1) * gapBetween)]['amplify'].count() / dataset['amplify'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    efficiencyRange = np.array(efficiencyRange)
    proportion = np.array(proportion)
    drawBarChart(efficiencyRange, proportion, 'Amplify Range', 'Proportion', 'Amplify vs Proportion', gapBetween)
    

    dataset.to_csv('modified_mouseMoveTable.csv', index=False)



def keyPressTableData2():
    dataset = pd.read_csv('KeyPressTable.csv')
    dataset['Type'] = dataset.apply(lambda row: 'Bot' if (row['user_id'] > 59 and row['user_id'] < 112) else 'Human', axis=1)
    dataset = dataset[['duration', 'Type']].sample(frac=1)
    dataset.to_csv('KeyPressDuration.csv', index=False)



def keyPressTableData():
    dataset = pd.read_csv('KeyPressTable.csv')


    #####################################Duration vs Proportion############################################
    maxiDuration = min(dataset['duration'].max(), 400)
    gapBetween = 10
    durationRange = []
    proportion = []
    for i in range(0, int(maxiDuration / gapBetween) + 1):
        durationRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['duration'] >= i * gapBetween) & (dataset['duration'] < (i + 1) * gapBetween)]['duration'].count() / dataset['duration'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    durationRange = np.array(durationRange)
    proportion = np.array(proportion)
    drawBarChart(durationRange, proportion, 'Duration Range', 'Proportion', 'Key press Duration vs Proportion', gapBetween)

    

    #############################################################################################################


    #####################################Inter arrival key stroke time vs Proportion############################################
    # separate each user and find the inter arrival time for each user  

    interArrivalList = []
    mm = True
    for user in dataset['user_id'].unique():
        userDataset = dataset[dataset['user_id'] == user]
        userDataset = userDataset.sort_values(by=['time'])


        userDataset['interArrivalTime'] = userDataset['time'].diff()
        userDataset = userDataset.dropna()



        # remove the row where inter arrival time greater than 5s
        userDataset = userDataset[userDataset['interArrivalTime'] < 5000]

        #remove the row where the inter arrival time is 0
        userDataset = userDataset[userDataset['interArrivalTime'] != 0]
        userDataset['Type'] = userDataset.apply(lambda row: 'Bot' if (row['user_id'] > 59 and row['user_id'] < 112) else 'Human', axis=1)
        newUserDataset = userDataset[['user_id','duration', 'interArrivalTime', 'Type']]
        # print(newUserDataset)
        if mm:
            newUserDataset.to_csv('KeyPressDurationInterArrival.csv',index=False)
            mm = False
        else:
            newUserDataset.to_csv('KeyPressDurationInterArrival.csv', mode='a', header=False, index=False)
        # append inter arrival time to interArrivalList
        interArrivalList.extend(userDataset['interArrivalTime'].tolist())
    
    # print(interArrivalList)
    
    interArrivalList = np.array(interArrivalList)
    print(interArrivalList)
    maxiInterArrivalTime = min(interArrivalList.max(), 600)
    gapBetween = 10
    interArrivalTimeRange = []
    proportion = []

    print('maxiInterArrivalTime: ', maxiInterArrivalTime)
    print("shape", type(interArrivalList))
    for i in range(0, int(maxiInterArrivalTime / gapBetween) + 1):
        interArrivalTimeRange.append(i*gapBetween)
        tmpProportion = np.sum((interArrivalList >= i * gapBetween) & (interArrivalList < (i + 1) * gapBetween))/ interArrivalList.size
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    interArrivalTimeRange = np.array(interArrivalTimeRange)
    proportion = np.array(proportion)
    drawBarChart(interArrivalTimeRange, proportion, 'Inter Arrival Time Range', 'Proportion', 'Inter Arrival Time vs Proportion', gapBetween) 
    #############################################################################################################

    #print(dataset)


def mouseClickTableData():
    dataset = pd.read_csv('MouseUpTable.csv')
    dataset['Type'] = dataset.apply(lambda row: 'Bot' if (row['user_id'] > 59 and row['user_id'] < 112) else 'Human', axis=1)
    newDataset = dataset[['user_id','duration', 'Type']].sample(frac=1)
    newDataset.to_csv('MouseClickPressDuration.csv', index=False)

    #####################################Mouse Click Duration vs Proportion############################################
    maxiDuration = min(dataset['duration'].max(), 400)
    gapBetween = 10
    durationRange = []
    proportion = []
    for i in range(0, int(maxiDuration / gapBetween) + 1):
        durationRange.append(i*gapBetween)
        tmpProportion = dataset[(dataset['duration'] >= i * gapBetween) & (dataset['duration'] < (i + 1) * gapBetween)]['duration'].count() / dataset['duration'].count()
        proportion.append(tmpProportion)
        # print('Proportion of displacement in range ', i * gapBetween, ' to ', (i + 1) * gapBetween, ' is ', tmpProportion)
    durationRange = np.array(durationRange)
    proportion = np.array(proportion)
    drawBarChart(durationRange, proportion, 'Duration Range', 'Proportion', 'Mouse Click Duration vs Proportion', gapBetween)    


if __name__ == '__main__':
    #keyPressTableData()
    # keyPressTableData2()
    # MouseMoveTableData()
    # mouseClickTableData()

    # groupSimilarData(20, 10)

    # Best: num_cluster = 20, num_run = 20, batch_size = 0.8, TPR = 0.96, TNR = 0.75
    # num_cluster = 20, num_run = 20, batch_size = 10, TPR = 1.0, TNR = 0.78125
    # num_cluster = 20, num_run = 20, batch_size = 12, TPR = 1.0, TNR = 0.8888
    # num_cluster = 20, num_run = 20, batch_size = 25, TPR = 1.0, TNR = 0.923



    X_test_human, X_test_bot = MouseMove_MakeCluster(20, 20)

    # assume evaluation function is called 'evaluate'
    batch_size = 10
    num_human = len(X_test_human) // batch_size
    num_bot = len(X_test_bot) // batch_size

    TP, TN, FP, FN = 0, 0, 0, 0



    print("X.human: ", X_test_human.shape)
    print("X.bot: ", X_test_bot.shape)

    print("num_human: ", num_human)
    print("num_bot: ", num_bot)

    print("Human Data: ")

    for i in range(num_human):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = X_test_human[start_idx:end_idx]
        result = Resolver(batch)
        if(result == 'Human'):
            TN += 1
        else:
            FP += 1

        print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")
        
    # handle the last batch separately in case it has less than 10 rows
    if len(X_test_human) % batch_size > 0:
        start_idx = num_human * batch_size
        end_idx = len(X_test_human)
        batch = X_test_human[start_idx:end_idx]
        result = Resolver(batch)
        print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")

        if(result == 'Human'):
            TN += 1
        else:
            FP += 1


    print("\nBot Data: \n")

    for i in range(num_bot):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = X_test_bot[start_idx:end_idx]
        result = Resolver(batch)
        print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")

        if result == 'Bot':
            TP += 1
        else:
            FN += 1
        
    # handle the last batch separately in case it has less than 10 rows
    if len(X_test_bot) % batch_size > 0:
        start_idx = num_bot * batch_size
        end_idx = len(X_test_bot)
        batch = X_test_bot[start_idx:end_idx]
        result = Resolver(batch)
        print(f"Evaluation result for rows {start_idx} to {end_idx}: {result}")


    TPR = TP / (TP + FN)

    TNR = TN / (TN + FP)

    FPR = FP / (FP + TN)

    FNR = FN / (TP + FN)

    print(f"TPR: {TPR}")
    print(f"TNR: {TNR}")
    print(f"FPR: {FPR}")
    print(f"FNR: {FNR}")

    



