# import pickle
# from sklearn.linear_model import LinearRegression

# def train():
#     # train the regression models for each cluster
#     reg_models = {}
#     for cluster in range(num_of_tables):
#         # get the data points in the current cluster
#         cluster_data = human_dataset[human_dataset['cluster'] == cluster]
        
#         # train the regression model on the current cluster
#         X = cluster_data[norm_cols]
#         y = cluster_data['Type']
#         reg_model = LinearRegression().fit(X, y)
        
#         # save the regression model to a file using pickle
#         with open(f'reg_model_cluster{cluster}.pkl', 'wb') as f:
#             pickle.dump(reg_model, f)
            
#         # store the regression model in a dictionary
#         reg_models[cluster] = reg_model
        
#     # to reload the regression model later, you can use the following code:
#     with open('reg_model_cluster0.pkl', 'rb') as f:
#         reg_model_cluster0 = pickle.load(f)


# def predict_cluster(data, kmeans, reg_models):
#     """
#     Predicts the cluster and class (human/bot) for the given data point.
    
#     Args:
#     data: pandas DataFrame containing the data point to be predicted.
#     kmeans: trained KMeans model for clustering.
#     reg_models: dictionary of regression models for each cluster.
    
#     Returns:
#     cluster_num: predicted cluster number for the data point.
#     class_label: predicted class label for the data point.
#     """
#     # Normalize the new data point
#     norm_cols = [col + '_normalized' for col in cols_to_normalize]
#     data[norm_cols] = (data[cols_to_normalize] - human_means) / human_stds
#     data[norm_cols] = data[norm_cols].fillna(0)

#     # Predict the cluster for the new data point
#     cluster_num = kmeans.predict(data[norm_cols])[0]

#     # Load the regression model for the predicted cluster
#     with open(f'reg_model_cluster{cluster_num}.pkl', 'rb') as f:
#         reg_model = pickle.load(f)

#     # Predict the class (human/bot) for the new data point using the regression model
#     predicted_val = reg_model.predict(data[norm_cols])[0]
#     if predicted_val >= threshold:
#         class_label = 'Bot'
#     else:
#         class_label = 'Human'

#     return cluster_num, class_label


# cluster_num, class_label = predict_cluster(new_data_point, kmeans_human, reg_models)


while(True):
    x = int(input("Enter first number: \n"))
    y = int(input("Enter second number: \n"))

    # result = True if x > y/2 else False
    print(int(y/2))