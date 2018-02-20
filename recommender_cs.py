import sys
import pandas as pd
import numpy as np
import random
from surprise import AlgoBase, Dataset, evaluate, print_perf
from surprise.dataset import Reader
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.knns import KNNBasic, KNNWithMeans, KNNBaseline
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.co_clustering import CoClustering
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GlobalMean(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean

class MeanofMeans(AlgoBase):
    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        users = np.array([u for (u, _, _) in self.trainset.all_ratings()])
        items = np.array([i for (_, i, _) in self.trainset.all_ratings()])
        ratings = np.array([r for (_, _, r) in self.trainset.all_ratings()])

        user_means,item_means = {},{}
        for user in np.unique(users):
            user_means[user] = ratings[users==user].mean()
        for item in np.unique(items):
            item_means[item] = ratings[items==item].mean()

        self.global_mean = ratings.mean()
        self.user_means = user_means
        self.item_means = item_means

    def estimate(self, u, i):
        """
        return the mean of means estimate
        """

        if u not in self.user_means:
            return(np.mean([self.global_mean,
                            self.item_means[i]]))

        if i not in self.item_means:
            return(np.mean([self.global_mean,
                            self.user_means[u]]))

        return(np.mean([self.global_mean,
                        self.user_means[u],
                        self.item_means[i]]))

def crossval_scores(data, models):
    seed = 123
    names = []
    rmse = []
    mae = []
    scoring = 'accuracy'
    for name, model in models:

        scores = evaluate(model, data, measures=['RMSE', 'MAE'])
        cv_rmse = scores['rmse']
        cv_mae = scores['mae']

        rmse.append(cv_rmse)
        mae.append(cv_mae)
        names.append(name)

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison of CrossVal Scores (Scaled Data)')
    ax1 = fig.add_subplot(121)
    # ax1.set_title('RSME')
    ax1.set_ylabel('RSME')
    sns.boxplot(data=rmse, orient='v', ax=ax1)
    # plt.boxplot(rmse)
    ax1.set_xticklabels(names)

    ax2 = fig.add_subplot(122)
    # ax2.set_title('MAE')
    ax2.set_ylabel('MAE')
    sns.boxplot(data=mae, orient='v', ax=ax2)
    # plt.boxplot(mae)
    ax2.set_xticklabels(names)

    plt.show()

    return names, rmse, mae

def convert_df_to_data(df, reader):
    data_from_df = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data_from_df

def norm_f_max_min(row, df_max, df_min):
    user_ind = list(row.index).index('userId')
    rating_ind = list(row.index).index('rating')
    user = row[user_ind]
    rating = row[rating_ind]
    max_rating = df_max[ df_max.index == user ].rating.values[0]
    min_rating = df_min[ df_min.index == user ].rating.values[0]
    rang = max_rating - min_rating
    if rang == 0:
        return rating/5
    else:
        return (rating - min_rating) / rang

def scale_ratings(df):
    s_df = df.copy()
    df_max = s_df.groupby('userId').max()
    df_min = s_df.groupby('userId').min()
    s_df['rating'] = s_df.apply(norm_f_max_min, axis=1, df_max=df_max, df_min=df_min)
    return s_df

def scale_ratings2(df):
    X_max = df.groupby('userId')['rating'].max().values
    X_min = df.groupby('userId')['rating'].min().values
    user_ids = np.array(ratings_df.groupby('userId')['rating'].max().index)
    # reshaping:
    X_min = X_min.reshape(X_min.shape[0], 1)
    X_max = X_max.reshape(X_max.shape[0], 1)
    user_ids = user_ids.reshape(user_ids.shape[0], 1)
    X = np.hstack([user_ids, X_max, X_min])
    ratings_range = X[:,1] - X[:, 2]
    # X = np.hstack([X, ratings_range.reshape(ratings_range.shape[0], 1)])
    X_ratings = df.values
    pass

def get_actual_est_ratings(model, data):
    trainset = data.build_full_trainset()
    model.train(trainset)
    testset = trainset.build_testset()
    predictions = model.test(testset)

    actual_ratings = [pred.r_ui for pred in predictions]
    est_ratings = [pred.est for pred in predictions]

    return actual_ratings, est_ratings

def tpfn(actual_rating, pred_rating, thresh):
    n = len(actual_rating)
    y_true = np.where(actual_rating >= thresh, 1, 0)
    y_pred = np.where(pred_rating >= thresh, 1, 0)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(n):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_pred[i] == 1:
                FP += 1
            else:
                TN += 1

    total_actual_true = sum(y_true)
    total_actual_false = n - total_actual_true

    TPR = TP/total_actual_true
    TNR = TN/total_actual_false
    FPR = FP/total_actual_false
    FNR = FN/total_actual_true

    return TP, TN, FP, FN, TPR, TNR, FPR, FNR, y_true, y_pred

def get_fpfns(data, models, thresh):
    names, fps, fns, tps, tns, precisions, recalls, f1s = [], [], [], [], [], [], [], []
    for name, model in models:
        act, est = get_actual_est_ratings(model, data)
        act = np.array(act)
        est = np.array(est)
        TP, TN, FP, FN, TPR, TNR, FPR, FNR, y_true, y_pred = tpfn(act, est, thresh)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1 = (precision*recall)/(precision + recall)

        fps.append(FP)
        fns.append(FN)
        tps.append(TP)
        tns.append(TN)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        names.append(name)

        print('{0} :\nPrecision : {1:0.03f}\nRecall : {2:0.03f}\nF1 Score : {3:0.03f}'.format(name, precision, recall, f1))
        print('.......')

    score_names = ['Precision', 'Recall', 'F1']
    scores = [precisions, recalls, f1s]
    for i in range(3):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison {} Score'.format(score_names[i]))
        ax = fig.add_subplot(111)
        sns.barplot(x=names, y=scores[i], orient='v', ax=ax)
        ax.set_ylabel(score_names[i])
        # ax.set(ylim=(0.75, 1))
        plt.show()


    return names, fps, fns, tps, tns, precisions, recalls, f1s

def plot_profit_curves(data, models, cost_matrix):

    optimal_profits = []
    for name, model in models:
        print('Training {}'.format(name))
        act, est = get_actual_est_ratings(model, data)
        print('Done Training {}'.format(name))
        print('.........')
        act = np.array(act)
        est = np.array(est)
        profits, thresholds = get_profit_curve(cost_matrix, est, act)
        max_profit = profits.max()
        max_ind = list(profits).index(max_profit)
        optimal_thresh = thresholds[max_ind]
        optimal_profits.append([name, max_profit, optimal_thresh])
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=name)

    plt.title("Profit Curves (1 TP = ${0}, 1 FP = ${1})".format(cost_matrix[0][0], cost_matrix[0][1]))
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.show()

    return optimal_profits

def get_profit_curve(cost_matrix, est, act):

    n = float(len(act))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in est else [1]
    # since the length of est is ~100,000, the computing time is huge. This is why I'm setting the total evaluated thresholds to be every 100 est values.
    thresholds = maybe_one + sorted(est[::300], reverse=True)
    profits = []
    for threshold in thresholds:
        # y_true is going to be defined by default as any true rating above 0.5 (scaled) will be 1 (user likes this movie) and below 0.5 (user doesn't like this movie).
        y_true = np.where(act >= 0.5, 1, 0)
        y_pred = np.where(est >= threshold, 1, 0)
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred)
        conf_mat = np.array([[TP, FP], [FN, TN]])
        # TP, TN, FP, FN, TPR, TNR, FPR, FNR, y_true, y_pred = tpfn(act, est, threshold)
        # confusion_matrix = np.array([[TP, FP], [FN, TN]])
        threshold_profit = np.sum(conf_mat * cost_matrix) / n
        profits.append(threshold_profit)
        # print('threshold : {0}, and profit {1}'.format(threshold, threshold_profit))
    return np.array(profits), np.array(thresholds)

def get_most_profitable_model(optimal_profits):

    baseline_profit = 0
    best_profit = 0
    for model_name, profit_return, optimal_thresh in optimal_profits:
        if model_name == 'MoM':
            mom_profit = profit_return
        if model_name == 'GM':
            gm_profit = profit_return
        if profit_return > best_profit:
            best_profit = profit_return
            best_model = model_name
            best_thresh = optimal_thresh

    print('Most profitable model : {0}\nProfit return : {1:0.2f}%\nThreshold : {2:0.2f}\n\nCompare with...\nMean of Means profit return : {3:0.2f}%\nGlobal Means profit return : {4:0.2f}%'.format(best_model, best_profit*100, best_thresh, mom_profit*100, gm_profit*100))

    return best_model, best_profit, best_thresh

def plot_eda(df):
    ratings_counts = df.groupby('userId').count()['movieId'].values
    sns.distplot(ratings_counts,norm_hist=False, kde=False, bins=40)
    plt.title('Distribution of the average amount of movies rated per user')
    plt.xlabel('movies rated')
    plt.ylabel('amount of users')
    plt.show()

def show_examples(df, movies_df, model, userid):

    user_df = df[df['userId'] == userid]
    user_movies_df = pd.merge(user_df, movies_df, how='inner', on='movieId')

    user_mid = user_df['movieId'].values
    user_predictions = [model.predict(uid=1, iid=movieid).est for movieid in user_mid]
    user_recs = [1 if model.predict(uid=1, iid=movieid).est > 0.5 else 0 for movieid in user_mid]
    user_movies_df['prediction'] = user_predictions
    user_movies_df['recommendation'] = user_recs

    TP = len(user_movies_df[(user_movies_df['recommendation'] == 1) & (user_movies_df['rating'] >= 0.5)])
    FP = len(user_movies_df[(user_movies_df['recommendation'] == 1) & (user_movies_df['rating'] < 0.5)])

    # showing model's recommendations based on predicting a rating greater than 0.5 (scaled):
    print(model.__class__.__name__)
    print('User {0}:\nCorrect recommendations : {1}\nWrong recommendations : {2}'.format(userid, TP, FP))
    print('.........')
    print('Correct Recs: :\n{}'.format(user_movies_df[(user_movies_df['recommendation'] == 1) & (user_movies_df['rating'] >= 0.5)][['title', 'genres', 'rating', 'prediction']]))
    print('.........')
    print('Wrong Recs:\n{}'.format(user_movies_df[(user_movies_df['recommendation'] == 1) & (user_movies_df['rating'] < 0.5)][['title', 'genres', 'rating', 'prediction']]))
    print('....................')
    print('....................')

def get_your_recs(personal_df, df, movies_df, model):

    # arbitrarily calling ourselves userId #700
    personal_df['userId'] = [700 for _ in range(len(personal_df))]

    new_df = pd.concat([df, personal_df], ignore_index=True)
    new_df_scaled = scale_ratings(new_df)
    scaled_reader = Reader(rating_scale=(0, 1))
    new_data = convert_df_to_data(new_df_scaled, scaled_reader)
    new_data.split(n_folds=5)

    new_trainset = new_data.build_full_trainset()
    model.train(new_trainset)

    unique_movie_ids = df['movieId'].unique()
    my_predictions = [model.predict(700, mid) for mid in unique_movie_ids]

    my_highest_est = np.array([pred.est for pred in my_predictions]).max()

    my_top_recs_ids = [pred.iid for pred in my_predictions if pred.est >= my_highest_est]

    id_to_title = {mid: m for m, mid in zip(movies_df['title'].values, movies_df['movieId'])}
    # id_to_title = {mid : title for title, mid in movies_id_dict.items()}
    my_top_recs = [id_to_title[mid] for mid in my_top_recs_ids]

    my_predictions = sorted(my_predictions, key=lambda pred : pred.est, reverse=True)
    all_my_estimates = [(id_to_title[pred.iid], pred.est) for pred in my_predictions]

    all_my_estimates_2016 = [est for est in all_my_estimates if '(2016)' in est[0]]

    return my_top_recs, all_my_estimates_2016, all_my_estimates

def get_grid_searched_model(model, param_grid, data):
    gs_model = GridSearch(model, param_grid, measures=['rmse', 'mae'])

    gs_model.fit(data)

    return gs_model

class MyMovieRecommender():
    def __init__(self, model, ratings_df, movies_df, not_scaled=False):
        self.model = model

        if is_scaled:
            self.ratings_df = scale_ratings(ratings_df)
        else:
            self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.my_estimates = []
        self.my_rated_movies = []



    def get_my_recs(self, my_ratings_dict):

        self.my_rated_movies = list(my_ratings_dict.keys())
        personal_dict = {'movieId' : list(my_ratings_dict.keys()), 'rating' : list(my_ratings_dict.values())}

        personal_df = pd.DataFrame(personal_dict)

        id_to_title = {mid: m for m, mid in zip(self.movies_df['title'].values, self.movies_df['movieId'])}

        # arbitrarily calling ourselves userId #10000
        personal_df['userId'] = [10000 for _ in range(len(personal_df))]

        personal_df = scale_ratings(personal_df)

        new_df = pd.concat([self.ratings_df, personal_df], ignore_index=True)
        # new_df_scaled = scale_ratings(new_df)
        scaled_reader = Reader(rating_scale=(0, 1))
        new_data = convert_df_to_data(new_df, scaled_reader)
        new_data.split(n_folds=5)

        new_trainset = new_data.build_full_trainset()
        self.model.train(new_trainset)

        unique_movie_ids = self.ratings_df['movieId'].unique()
        my_predictions = [self.model.predict(10000, mid) for mid in unique_movie_ids if id_to_title[mid] not in self.my_rated_movies]
        my_predictions = sorted(my_predictions, key=lambda pred : pred.est, reverse=True)

        my_estimates = [(id_to_title[pred.iid], pred.est) for pred in my_predictions]

        self.my_estimates = my_estimates

if __name__ == '__main__':

    movies_df = pd.read_csv('data/movies/movies.csv')
    # data = Dataset.load_builtin('ml-100k')
    # data.split(n_folds=5)
    scaled_reader = Reader(rating_scale=(0, 1))
    reader = Reader(rating_scale=(1, 5))
    df = pd.read_csv('data/movies/ratings.csv')
    scaled_df = scale_ratings(df)
    scaled_data = convert_df_to_data(scaled_df, scaled_reader)
    scaled_data.split(n_folds=5)

    data = convert_df_to_data(df, reader)
    data.split(n_folds=5)

    # plot some EDA figures:
    plot_average_rating_hist(df)

    # Cross Valdiation Tests for different Classification Models:
    models = []
    models.append(('GM', GlobalMean()))
    models.append(('MoM', MeanofMeans()))
    models.append(('BLO', BaselineOnly()))
    models.append(('KNNb', KNNBasic()))
    models.append(('KNNwm', KNNWithMeans()))
    models.append(('KNNbl', KNNBaseline()))
    models.append(('SVD', SVD()))
    models.append(('NMF', NMF()))
    models.append(('SO', SlopeOne()))
    models.append(('CoC', CoClustering()))

    # plotting box plot of cross validation scores for array of recommendation models on scaled ratings data:
    model_names, rmses, maes = crossval_scores(scaled_data, models[:-1])

    # Now to find out which recommendation model has the lowest amount of false positives (recommending a movie that a user wounldn't like) and false negatives (failing to recommend a movie that a user would like). We'll choose a model based on the f1 score.
    model_names, fps, fns, tps, tns, precisions, recalls, f1s = get_fpfns(scaled_data, models, thresh=0.5)

    # Highest F1 score was the SVD model. We'll go with this model build a recommender system.

    '''To make a business case we'll have to make some assuptions about the costs and benefits that Movies-Legit service experiences when giving users recommendations they like (True Positive) and giving users recommendations they don't like (False Positives).

    Assumptions:
    - $10/mo is the subscription to use movie service (like a Netflix)
    - 1 out of 20 users will renew their subscription if they are recommended a movie they like (True Positive)
    ---> 1 TP = $10 * (1/20) = $0.50
    - 1 out of 20 users will end their subscription if they are recommended a movie they don't like (False Positive)
    ---> 1 FP = -$10 * (1/20) =  -$0.50

    With these assumptions we can gather a quick estimate of the profit gained or lost when moving from the current recommender system (Mean of Means) to another recommender system
    Cost Matrix format --> -----------
                           | TP | FP |
                           -----------
                           | FN | TN |
                           -----------
    '''

    cost_matrix = np.array([[0.5, -0.5], [0, 0]])
    optimal_profits = plot_profit_curves(scaled_data, models, cost_matrix)
    best_model, best_profit, best_thresh = get_most_profitable_model(optimal_profits)

    # NMF is the most profitable model to use.
    trainset = scaled_data.build_full_trainset()

    nmf = NMF()
    nmf.train(trainset)
    # show_examples(scaled_df, movies_df, nmf, 1)
    show_examples(scaled_df, movies_df, nmf, 10)

    gm = GlobalMean()
    gm.train(trainset)
    show_examples(scaled_df, movies_df, gm, 10)

    mom = MeanofMeans()
    mom.train(trainset)
    show_examples(scaled_df, movies_df, mom, 10)

    ###### Now it's time to test out recommendations for yourself #########
    movies_you_love = ['Mad Max: Fury Road (2015)', 'Star Trek (2009)', 'Beerfest (2006)', 'Guardians of the Galaxy (2014)', 'Batman Begins (2005)', '21 Jump Street (2012)', '22 Jump Street (2014)', 'Trainwreck (2015)', 'The Brothers Grimsby (2016)', 'Spy (2015)', 'Kingsman: The Secret Service (2015)', 'Ted 2 (2015)', 'Wolf of Wall Street, The (2013)', 'Secret Life of Walter Mitty, The (2013)', 'John Wick (2014)', 'Skyfall (2012)', 'Just Friends (2005)', 'Wedding Crashers (2005)', '50 First Dates (2004)', 'Miss Congeniality (2000)', 'How to Lose a Guy in 10 Days (2003)', 'My Big Fat Greek Wedding (2002)', '40-Year-Old Virgin, The (2005)']

    movies_you_hate = ['Up in the Air (2009)', 'Rocky Horror Picture Show, The (1975)', 'Hairspray (2007)', 'Down and Out in Beverly Hills (1986)', 'Finding Dory (2016)', 'Notebook, The (2004)', 'Jackass Presents: Bad Grandpa (2013)', 'Rise of the Planet of the Apes (2011)', 'Suicide Squad (2016)']

    # movies_you_love = ['Just Friends (2005)', 'Wedding Crashers (2005)', 'Trainwreck (2015)', '50 First Dates (2004)', 'Miss Congeniality (2000)', 'How to Lose a Guy in 10 Days (2003)', 'My Big Fat Greek Wedding (2002)', '40-Year-Old Virgin, The (2005)']

    movies_id_dict = {m: mid for m, mid in zip(movies_df['title'].values, movies_df['movieId'])}

    movies_you_love_id = [movies_id_dict[m] for m in movies_you_love]
    movies_you_hate_id = [movies_id_dict[m] for m in movies_you_hate]

    ratings_dict = {m : 5.0 for m in movies_you_love_id}
    ratings_dict.update({m : 0.0 for m in movies_you_hate_id})

    personal_dict = {'movieId' : list(ratings_dict.keys()), 'rating' : list(ratings_dict.values())}

    personal_df = pd.DataFrame(personal_dict)

    my_top_recs_nmf, all_my_est_2016_nmf, all_my_estimates_nmf = get_your_recs(personal_df, df, movies_df, nmf)

    title_to_genre = {title : genre for title, genre in zip(movies_df['title'], movies_df['genres'])}

    my_romcom_est = [m for m in all_my_estimates_nmf if 'romance' in title_to_genre[m[0]].lower() and 'comedy' in title_to_genre[m[0]].lower() and 'drama' not in title_to_genre[m[0]].lower() and '(201' in m[0]]

    my_comedy_est = [m for m in all_my_estimates_nmf if 'comedy' in title_to_genre[m[0]].lower() and 'drama' not in title_to_genre[m[0]].lower() and '(201' in m[0]]

    my_thriller_est = [m for m in all_my_estimates_nmf if 'thriller' in title_to_genre[m[0]].lower() and 'action' in title_to_genre[m[0]].lower() and 'horror' not in title_to_genre[m[0]].lower() and 'comedy' not in title_to_genre[m[0]].lower() and '(201' in m[0]]

    my_top_recs_svd, all_my_est_2016_svd = get_your_top_recs(personal_df, df, movies_df, SVD())

    my_top_recs_kNNbl, all_my_est_2016_kNNbl = get_your_top_recs(personal_df, df, movies_df, KNNBaseline())
