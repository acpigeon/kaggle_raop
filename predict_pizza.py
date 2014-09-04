__author__ = 'acpigeon'
import json
import random
import datetime
import math
import csv
import joblib
import nltk

import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from nltk.stem.snowball import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_data(filename, max_neg_class=float("inf")):
    """
    request_id: text, key
    requester_number_of_comments_at_request: int
    requester_number_of_comments_in_raop_at_request: int
    requester_number_of_posts_at_request: int
    requester_number_of_posts_on_raop_at_request: int
    requester_number_of_subreddits_at_request: int
    requester_upvotes_minus_downvotes_at_request": int
    requester_upvotes_plus_downvotes_at_request: int
    requester_account_age_in_days_at_request: float
    requester_days_since_first_post_on_raop_at_request: float
    unix_timestamp_of_request: float
    requester_subreddits_at_request: list of strings
    request_text_edit_aware: string
    request_title: string
    """
    input_file = open(filename, 'r')
    file_contents = input_file.read()
    raw_data = json.loads(file_contents)
    random.shuffle(raw_data)  # Shuffle the data now before we transform it

    if 'requester_received_pizza' in raw_data[0].keys():  # this is the train set, downsample neg class
        neg_class_count = 0
        downsampled_data = []
        for example in raw_data:
            if example['requester_received_pizza'] is True or neg_class_count < max_neg_class:
                downsampled_data.append(example)
                if example['requester_received_pizza'] is False:
                    neg_class_count += 1
        return downsampled_data
    else:
        return raw_data


def build_num_features_matrix(data_set):
    """
    Returns an n x 9 matrix of all numeric features.
    """
    n = len(data_set)
    mat = np.zeros((n, 9))
    for i in xrange(n):
        mat[i][0] = data_set[i]['requester_number_of_comments_at_request']
        mat[i][1] = data_set[i]['requester_number_of_comments_in_raop_at_request']
        mat[i][2] = data_set[i]['requester_number_of_posts_at_request']
        mat[i][3] = data_set[i]['requester_number_of_posts_on_raop_at_request']
        mat[i][4] = data_set[i]['requester_number_of_subreddits_at_request']
        mat[i][5] = data_set[i]['requester_upvotes_minus_downvotes_at_request']
        mat[i][6] = data_set[i]['requester_upvotes_plus_downvotes_at_request']
        mat[i][7] = data_set[i]['requester_account_age_in_days_at_request']
        mat[i][8] = data_set[i]['requester_days_since_first_post_on_raop_at_request']
    return scale(mat)


def build_date_features(data_set):
    """
    For the date of posting (from unix_timestamp_of_request), convert to day of week and hours after midnight feature.
    """
    n = len(data_set)
    mat = np.zeros((n, 8))
    date_to_columns = {'Mon': 0,  'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    for i in xrange(n):
        epoch_seconds = data_set[i]['unix_timestamp_of_request']
        day = datetime.datetime.fromtimestamp(epoch_seconds).strftime('%a')
        hours_after_midnight = datetime.datetime.fromtimestamp(epoch_seconds).strftime('%H')
        mat[i][date_to_columns[day]] = 1
        mat[i][7] = int(hours_after_midnight)
        return mat


def build_text_list_features(data_set):
    """
    Convert list of text into categorical features, only used for subreddits here.
    """
    n = len(data_set)
    vectorizer = CountVectorizer()
    lists_of_subreddits = []
    for i in xrange(n):
        lists_of_subreddits.append(' '.join(data_set[i]['requester_subreddits_at_request']))
    mat = vectorizer.fit_transform(lists_of_subreddits)
    return mat.todense()


def get_meta(data_set, field_name):
    """
    Returns an n x 1 array of the doc ids or labels.
    Pass field_name = 'request_id' or 'requester_received_pizza'.
    """
    n = len(data_set)
    if field_name == 'request_id':
        t = object
    else:
        t = float

    mat = np.zeros((n, 1), dtype=t)
    for idx in xrange(n):
        mat[idx] = data_set[idx][field_name]
    return mat


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, PorterStemmer())
    return stems


def generate_tfidf_matrix(train, test, field_name, _min_df=0.01, _max_df=0.7):
    """
    Takes list of lists of text and returns the tfidf matrix.
    Used for request_text_edit_aware, ....
    """
    train_text, test_text = [], []
    for t in train:
        train_text.append(t[field_name])
    for t in test:
        test_text.append(t[field_name])

    v = TfidfVectorizer(stop_words='english', min_df=_min_df, max_df=_max_df, tokenizer=tokenize)
    v.fit(train_text + test_text)
    return v.transform(train_text).todense(), v.transform(test_text).todense()


if __name__ == "__main__":
    # Load train data
    train_data = load_data('train.json')
    train_ids = get_meta(train_data, 'request_id')
    train_numeric_features = build_num_features_matrix(train_data)
    train_date_features = build_date_features(train_data)
    train_subreddit_features = build_text_list_features(train_data)
    train_labels = get_meta(train_data, 'requester_received_pizza')

    # Load test data
    test_data = load_data('test.json')
    test_ids = get_meta(test_data, 'request_id')
    test_numeric_features = build_num_features_matrix(test_data)
    test_date_features = build_date_features(test_data)
    test_subreddit_features = build_text_list_features(test_data)

    # Train all tf features before messing with the data
    tf_train_request, tf_test_request = generate_tfidf_matrix(train_data, test_data, 'request_text_edit_aware')
    tf_train_title, tf_test_title = generate_tfidf_matrix(train_data, test_data, 'request_title')

    # Combine all the features
    train_feature_matrix = np.concatenate((train_numeric_features, train_date_features,
                                           tf_train_request, tf_train_title), axis=1)
    test_feature_matrix = np.concatenate((test_numeric_features, test_date_features,
                                          tf_test_request, tf_test_title), axis=1)

    X_train_all, X_test, y_train, y_test = train_test_split(train_feature_matrix, train_labels.ravel())

    # In the test split, there is a pos/neg imbalance of ~ 730 to 2200
    # Split the negative class into three roughly equal groups so we can train three different models and take the avg
    # Methodology comes from EasyEnsemble approach from http://cse.seu.edu.cn/people/xyliu/publication/tsmcb09.pdf

    X_train_neg_1, y_train_neg_1 = [], []
    X_train_neg_2, y_train_neg_2 = [], []
    X_train_neg_3, y_train_neg_3 = [], []
    X_train_pos, y_train_pos = [], []

    for s in zip(X_train_all, y_train):
        if s[1] == 1.0:
            X_train_pos.append(s[0])
            y_train_pos.append(s[1])
        else:
            sorting_hat = random.choice([1, 2, 3])
            if sorting_hat == 1:
                X_train_neg_1.append(s[0])
                y_train_neg_1.append(s[1])
            elif sorting_hat == 2:
                X_train_neg_2.append(s[0])
                y_train_neg_2.append(s[1])
            else:
                X_train_neg_3.append(s[0])
                y_train_neg_3.append(s[1])

    # Then recombine each of the negative class subsets with the positive class
    # This gives us three separate training groups of approximately equal split!

    X_train_1 = np.array(X_train_neg_1 + X_train_pos)
    y_train_1 = np.array(y_train_neg_1 + y_train_pos)

    X_train_2 = np.array(X_train_neg_2 + X_train_pos)
    y_train_2 = np.array(y_train_neg_2 + y_train_pos)

    X_train_3 = np.array(X_train_neg_3 + X_train_pos)
    y_train_3 = np.array(y_train_neg_3 + y_train_pos)

    # Train the models

    #1
    lr = LogisticRegression(tol=0.01)
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
    clf1 = GridSearchCV(lr, param_grid=params, scoring='roc_auc', verbose=True, cv=5, n_jobs=-1)
    clf1.fit(X_train_1, y_train_1)
    clf_1_x_val_predictions = clf1.predict(X_test)
    class_rep_1 = classification_report(y_test, clf_1_x_val_predictions)
    print clf1.best_params_
    print class_rep_1


    #2
    svc = SVC()
    svc_param_dist = {"C": uniform(),
                         "gamma": uniform(),
                         "kernel": ['linear', 'rbf'],
                         "class_weight": [{1: 1}, {1: 2}, {1: 5}, {1: 10}],
                         "probability": [True]
                         }
    #params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #           'kernel': ['linear'], 'class_weight': [{1: 1}, {1: 5}, {1: 2}, {1: 3}, {1: 10}]}]

    #clf2 = GridSearchCV(svc, param_grid=params, scoring='roc_auc', verbose=True, cv=5, n_jobs=-1)
    clf2 = RandomizedSearchCV(svc, param_distributions=svc_param_dist, n_iter=100)

    clf2.fit(X_train_2, y_train_2)
    clf_2_x_val_predictions = clf2.predict(X_test)
    class_rep_2 = classification_report(y_test, clf_2_x_val_predictions)
    print clf2.best_params_
    print class_rep_2

    #3
    gbc = GradientBoostingClassifier()
    forest_param_dist = {"max_depth": [3,4,5,6,7],
                               "max_features": sp_randint(1, 11),
                               "min_samples_split": sp_randint(1, 11),
                               "min_samples_leaf": sp_randint(1, 11),
                               "subsample": uniform(),
                               "learning_rate": uniform(),
                               "n_estimators": sp_randint(1, 351)}

    clf3 = RandomizedSearchCV(gbc, param_distributions=forest_param_dist, n_iter=100)
    #    clf3 = GridSearchCV(gbc, [{'learning_rate': [.01, .03, .1, .3], 'n_estimators': [50, 100, 150],
    #                             "max_depth": [3, 4, 5]}], cv=5, n_jobs=-1, scoring='roc_auc', verbose=True)
    clf3.fit(X_train_3, y_train_3)
    clf_3_x_val_predictions = clf3.predict(X_test)
    class_rep_3 = classification_report(y_test, clf_3_x_val_predictions)
    print clf3.best_params_
    print class_rep_3

    #joblib.dump(clf, 'model.bin', 5)

    # Average predictions from the three classifiers
    clf_1_x_test_predictions = clf1.best_estimator_.predict_proba(test_feature_matrix)[:, 1]
    clf_2_x_test_predictions = clf2.best_estimator_.predict_proba(test_feature_matrix)[:, 1]
    clf_3_x_test_predictions = clf3.best_estimator_.predict_proba(test_feature_matrix)[:, 1]

    output_predictions = []
    for p in zip(clf_1_x_test_predictions, clf_2_x_test_predictions, clf_3_x_test_predictions):
        if p[0] + p[1] + p[2] > 1.25:
            output_predictions.append(1)
        else:
            output_predictions.append(0)


    output = zip([x[0] for x in test_ids], output_predictions)
    output.insert(0, ["request_id", "requester_received_pizza"])

    output_file = csv.writer(open('predictions.csv', 'w'), delimiter=",", quotechar='"')
    for row in output:
        output_file.writerow(row)
