__author__ = 'acpigeon'

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import numpy as np
from numpy.random import shuffle
import csv


def get_training_features():
    """
    Train data set includes a bunch of features that aren't known at the time of posting.
    Get the list of the feature subset that we can train on.
    """
    input_file = open('test.json', 'r')
    file_contents = input_file.read()
    json_data = json.loads(file_contents)
    return json_data[0].keys()


def load_data(filename):
    """
    Features: NEED TO UPDATE THIS WITH NON TIME TRAVEL FEATURES
        request_id: doc_id
        requester_number_of_posts_on_raop_at_request: int
        # requester_subreddits_at_request:
        # requester_number_of_posts_on_raop_at_retrieval: int
        # requester_number_of_comments_at_request:
        # request_title:
        # requester_days_since_first_post_on_raop_at_retrieval:
        # giver_username_if_known:
        # requester_days_since_first_post_on_raop_at_request:
        # post_was_edited:
        # re^C_raop_at_retrieval:
        # requester_number_of_comments_at_request:
        # request_title:
        # requester_days_since_first_post_on_raop_at_retrieval:
        # giver_username_if_known:
        # requester_days_since_first_post_on_raop_at_request:
        # post_was_edited:
        # requester_account_age_in_days_at_request:
        # requester_upvotes_minus_downvotes_at_retrieval:
        # requester_number_of_posts_at_retrieval:
        # requester_user_flair:
        # requester_upvotes_minus_downvotes_at_request:
        # requester_username:
        # unix_timestamp_of_request:
        # requester_upvotes_plus_downvotes_at_request:
        # unix_timestamp_of_request_utc:
        # number_of_upvotes_of_request_at_retrieval:
        # number_of_downvotes_of_request_at_retrieval:
        # requester_number_of_comments_in_raop_at_retrieval:
        # request_number_of_comments_at_retrieval:
        # requester_number_of_posts_at_request:
        # requester_number_of_comments_at_retrieval:
        # request_text: string
        # requester_account_age_in_days_at_retrieval:
        # requester_upvotes_plus_downvotes_at_retrieval:
        request_text_edit_aware: string
        # requester_number_of_comments_in_raop_at_request:
        # requester_number_of_subreddits_at_request:

        requester_received_pizza: boolean label
    """
    input_file = open(filename, 'r')
    file_contents = input_file.read()
    raw_data = json.loads(file_contents)
    return raw_data


def generate_tfidf_matrix(list_of_lists, set_type=None):
    """
    Takes list of lists of text and returns the tfidf matrix.
    Used for request_text_edit_aware, ....
    """
    v = TfidfVectorizer(stop_words='english', min_df=0.05, max_df=0.5)
    if set_type == 'train':
        term_scores = v.fit_transform(list_of_lists).toarray()
    else:
        term_scores = v.transform(list_of_lists).toarray()
    #print v.get_feature_names()
    return term_scores


def gen_num_array(list_of_values, n):
    """
    Takes a list of numeric features and returns the numpy array of the values
    Used for requester_number_of_posts_on_raop_at_request, ...
    """
    return np.array(list_of_values).reshape(n, 1)


def gen_features(data_set, gen_labels=False):
    """
    Input a data set and return the numpy feature matrix.
    If the training set is passed in, split into train and xval sets (30%)
    """

    feature_list = get_training_features() # should refactor to run through this list

    request_id = []
    requester_number_of_posts_on_raop_at_request = []
    requester_number_of_posts_at_request = []
    request_text_edit_aware = []
    labels = []
    n = len(data_set)

    if gen_labels:  # if this is the train set, we also want to generate labels
        i = 0
        labels = np.zeros((n, 1))

    for doc in data_set:
        request_id.append(doc['request_id'])
        requester_number_of_posts_on_raop_at_request.append(doc['requester_number_of_posts_on_raop_at_request'])
        requester_number_of_posts_at_request.append(doc['requester_number_of_posts_at_request'])
        request_text_edit_aware.append(doc['request_text_edit_aware'])

        if gen_labels:
            labels[i] = doc['requester_received_pizza']
            i += 1

    # first feature initially populates output variable
    feature_matrix = gen_num_array(requester_number_of_posts_on_raop_at_request, n)

    # subsequent features add on to (copies of) feature_matrix
    #feature_matrix = np.append(feature_matrix, gen_num_array(requester_number_of_posts_at_request, n), axis=1)
    if gen_labels:
        feature_matrix = np.append(feature_matrix, generate_tfidf_matrix(request_text_edit_aware, 'train'), axis=1)
    else:
        feature_matrix = np.append(feature_matrix, generate_tfidf_matrix(request_text_edit_aware, 'test'), axis=1)

    # Finally, if this is the train set, append labels, shuffle, split into train and xVal
    if gen_labels:
        # Add the labels to the feature matrix so we can shuffle and split without losing label order
        feature_matrix = np.append(feature_matrix, gen_num_array(labels, n), axis=1)

        # Shuffle the training examples to make the split random
        index_array = np.arange(len(feature_matrix))
        #shuffle(index_array)
        #shuffled = feature_matrix[index_array[:]]
        #shuffled_request_ids = request_id[index_array[:]]
        shuffled = feature_matrix
        shuffled_request_ids = request_id

        # Split off the labels
        shuffled_labels = shuffled[:, -1]  # should  treat this like the doc_ids and never join with the features

        # Decide where to split the data
        split = len(feature_matrix) / 3

        # Split out train and xval and labels. Don't include the last column (labels)
        x_cross_val = shuffled[0:split, :-1]
        y_cross_val = shuffled_labels[0:split]
        id_cross_val = shuffled_request_ids[0:split]

        x_train = shuffled[split:, :-1]
        y_train = shuffled_labels[split:]
        id_train = shuffled_request_ids[split:]

        #print x_train.shape
        #print x_cross_val.shape
        #print y_train.shape
        #print y_cross_val.shape

        return id_train, x_train, y_train, id_cross_val, x_cross_val, y_cross_val
    else:
        return request_id, feature_matrix


if __name__ == "__main__":
    print "Loading data..."
    train_set = load_data('train.json')
    id_t, X_t, y_t, id_v, X_v, y_v = gen_features(train_set, gen_labels=True)

    #test_set = load_data('test.json')
    #test_request_ids, X_test = gen_features(test_set, gen_labels=False)

    print y_t
    print type(y_t)

    print "Training model..."
    for i, C in enumerate(10.0 ** np.arange(1, 4)):
        lr_l1 = linear_model.LogisticRegression(C=C, penalty='l1', tol=0.01)
        lr_l2 = linear_model.LogisticRegression(C=C, penalty='l2', tol=0.01)
        lr_l1.fit(X_t, y_t)
        lr_l2.fit(X_t, y_t)

        print "Scoring model with C=" + str(C) + "..."
        print "L1 Score: " + str(lr_l1.score(X_v, y_v))
        print "L2 Score: " + str(lr_l2.score(X_v, y_v))
        print ""

    print type(id_t), id_t[0:3]
    print X_t.dtype, X_t[0:3]
    print y_t.dtype, y_t[0:3]
    print ""
    print type(id_v), id_v[0:3]
    print X_v.dtype, X_v[0:3]
    print y_v.dtype, y_v[0:3]

    """

    final_lr = linear_model.LogisticRegression(C=1000.0, penalty='l2', tol=0.1)
    final_lr.fit(X_t, y_t)
    predictions = final_lr.predict(X_test)

    output = zip(test_request_ids, predictions)
    output.insert(0, ["request_id", "requester_received_pizza"])

    output_file = csv.writer(open('predictions.csv', 'w'), delimiter=",", quotechar='"')
    for row in output:
        output_file.writerow(row)
    """

