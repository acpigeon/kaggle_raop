__author__ = 'acpigeon'
import json
import numpy as np
import random
import datetime
from numpy.random import shuffle
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
import csv
import math


def load_data(filename):
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
    unix_timestamp_of_request_utc: float
    requester_username: string
    giver_username_if_known: string
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
            if example['requester_received_pizza'] is True or neg_class_count < 995:
                downsampled_data.append(example)
                if example['requester_received_pizza'] is False:
                    neg_class_count += 1
        return downsampled_data
    else:
        return raw_data


def build_num_features_matrix(data_set):
    """
    Returns an n x 11 matrix of all numeric features.
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
    For the date of posting (from unix_timestamp_of_request), convert to day of week feature.
    """
    n = len(data_set)
    mat = np.zeros((n, 7))
    date_to_columns = {'Mon': 0,  'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    for i in xrange(n):
        epoch_seconds = data_set[i]['unix_timestamp_of_request']
        day = datetime.datetime.fromtimestamp(epoch_seconds).strftime('%a')
        mat[i][date_to_columns[day]] = 1
    return mat




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


def split_matrix(mat):
    """
    Splits the input data and along the split index param.
    """
    split = len(mat) / 3
    xval_split = mat[0:split]
    train_split = mat[split:]
    return train_split, xval_split


def generate_tfidf_matrix(train_data, test_data):
    """
    Takes list of lists of text and returns the tfidf matrix.
    Used for request_text_edit_aware, ....
    """
    train_text, test_text = [], []
    for t in train_data:
        train_text.append(t['request_text_edit_aware'])
    for t in test_data:
        test_text.append(t['request_text_edit_aware'])

    v = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.5)
    v.fit(train_text + test_text)
    return v.transform(train_text), v.transform(test_text)


if __name__ == "__main__":
    # Load train data
    train_data = load_data('train.json')  # check
    train_ids = get_meta(train_data, 'request_id')  # check
    train_numeric_features = build_num_features_matrix(train_data)  # check
    train_date_features = build_date_features(train_data)
    train_labels = get_meta(train_data, 'requester_received_pizza')  # check

    # Load test data
    test_data = load_data('test.json')  # check
    test_ids = get_meta(test_data, 'request_id')  # check
    test_numeric_features = build_num_features_matrix(test_data)  # check
    test_date_features = build_date_features(test_data)

    # Train tf before messing with the data
    tf_train, tf_test = generate_tfidf_matrix(train_data, test_data)

    # Combine all the features
    train_feature_matrix = np.concatenate((train_numeric_features, train_date_features, tf_train.todense()), axis=1)
    test_feature_matrix = np.concatenate((test_numeric_features, test_date_features, tf_test.todense()), axis=1)

    # Split training data in train and xval sets
    id_t, id_v = split_matrix(train_ids)
    X_t, X_v = split_matrix(train_feature_matrix)
    y_t, y_v = split_matrix(train_labels)

    # Train the model
    gbc = GradientBoostingClassifier()
    alpha = np.array([math.pow(10, x) for x in np.arange(-5, 5)])

    clf = GridSearchCV(gbc, [{'n_estimators': [80, 100, 120], "max_depth": [3, 4, 5]}], cv=10, n_jobs=-1)
    clf.fit(X_t, y_t.ravel())

    print clf.score(X_t, y_t.ravel())

    predictions = clf.predict(test_feature_matrix)

    print sum(predictions)

    output = zip([x[0] for x in test_ids], [int(x) for x in predictions])
    output.insert(0, ["request_id", "requester_received_pizza"])

    output_file = csv.writer(open('predictions.csv', 'w'), delimiter=",", quotechar='"')
    for row in output:
        output_file.writerow(row)