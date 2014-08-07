__author__ = 'acpigeon'
import json
import numpy as np
from numpy.random import shuffle
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return raw_data


def build_num_features_matrix(data_set):
    """
    Returns an n x 11 matrix of all numeric features.
    """
    n = len(data_set)
    mat = np.zeros((n, 11))
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
        mat[i][9] = data_set[i]['unix_timestamp_of_request']
        mat[i][10] = data_set[i]['unix_timestamp_of_request_utc']
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
    full_vocab = v.fit(train_text + test_text)
    print len(v.get_feature_names())
    return v.transform(train_text), v.transform(test_text)


if __name__ == "__main__":
    # Load train data
    train_data = load_data('train.json')
    train_ids = get_meta(train_data, 'request_id')
    train_numeric_features = build_num_features_matrix(train_data)
    train_labels = get_meta(train_data, 'requester_received_pizza')

    # Load test data
    test_data = load_data('test.json')
    test_ids = get_meta(test_data, 'request_id')
    test_numeric_features = build_num_features_matrix(test_data)

    # Shuffle data before splitting
    index_array = np.arange(len(train_labels))
    #shuffle(index_array)
    shuffled_ids = train_ids[index_array[:]]
    shuffled_features = train_numeric_features[index_array[:]]
    shuffled_labels = train_labels[index_array[:]].ravel()  # changes column vector to row vector
    print shuffled_labels.shape

    # Split data in train and xval sets
    id_t, id_v = split_matrix(shuffled_ids)
    X_t, X_v = split_matrix(shuffled_features)
    y_t, y_v = split_matrix(shuffled_labels)

    tf_train, tf_test = generate_tfidf_matrix(train_data, test_data)

    print train_numeric_features.shape, tf_train.shape
    print test_numeric_features.shape, tf_test.shape
    feature_matrix = np.concatenate((train_numeric_features, tf_train), axis=1)
    test_feature_matrix = np.concatenate((test_numeric_features, tf_test), axis=1)

    """
    print "Training model..."
    for i, C in enumerate(10.0 ** np.arange(1, 4)):
        lr_l1 = linear_model.LogisticRegression(C=C, penalty='l1', tol=0.01)
        lr_l2 = linear_model.LogisticRegression(C=C, penalty='l2', tol=0.01)
        lr_l1.fit(X_t, np.array(y_t))
        lr_l2.fit(X_t, np.array(y_t))

        print "Scoring model with C=" + str(C) + "..."
        print "L1 Score: " + str(lr_l1.score(X_v, y_v))
        print "L2 Score: " + str(lr_l2.score(X_v, y_v))
        print ""
    """



    final_lr = GradientBoostingClassifier()
    alpha = np.array([math.pow(10, x) for x in np.arange(-5, 5)])

    clf = GridSearchCV(final_lr, [{'n_estimators': [80, 100, 120], "max_depth": [3, 4, 5]}], cv=10, n_jobs=-1)
    clf.fit(feature_matrix.toarray(), shuffled_labels)
    #final_lr.fit(X_t, y_t)
    #predictions = clf.predict(X_t)

    print clf.score(feature_matrix.toarray(), shuffled_labels)

    predictions = clf.predict(test_feature_matrix.toarray())
    #pred2 = clf.predict(X_t)

    #print X_t[0]
    #print test_numeric_features[0]

    print sum(predictions)
    #print sum(pred2)

    #print sum(y_t) * 1.0 / len(y_t)


    output = zip([x[0] for x in test_ids], predictions)
    output.insert(0, ["request_id", "requester_received_pizza"])

    output_file = csv.writer(open('predictions.csv', 'w'), delimiter=",", quotechar='"')
    for row in output:
        output_file.writerow(row)
