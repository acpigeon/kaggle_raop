import nltk
from nltk.stem.snowball import PorterStemmer
from sklearn.cross_validation import train_test_split

__author__ = 'acpigeon'
import json
import random
import datetime
import math
import csv
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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


def split_matrix(mat):
    """
    Splits the input data and along the split index param.
    Better: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html#sklearn.cross_validation.train_test_split
    """
    split = len(mat) / 3
    xval_split = mat[0:split]
    train_split = mat[split:]
    return train_split, xval_split


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
    clf1 = LogisticRegression(C=1, penalty='l1', tol=0.01)
    clf1.fit(X_train_1, y_train_1)
    clf_1_predictions = clf1.predict(X_test)
    class_rep_1 = classification_report(y_test, clf_1_predictions)
    print class_rep_1

    #2
    clf2 = LogisticRegression(C=1, penalty='l2', tol=0.01)
    clf2.fit(X_train_2, y_train_2)
    clf_2_predictions = clf2.predict(X_test)
    class_rep_2 = classification_report(y_test, clf_2_predictions)
    print class_rep_2

    #3
    gbc = GradientBoostingClassifier()
    alpha = np.array([math.pow(10, x) for x in np.arange(-2, 2)])

    clf3 = GridSearchCV(gbc, [{'learning_rate': [.01, .03, .1, .3], 'n_estimators': [50, 100, 150],
                              "max_depth": [3, 4, 5]}], cv=5, n_jobs=-1, scoring='roc_auc', verbose=True)

    clf3.fit(X_train_3, y_train_3)
    clf_3_predictions = clf3.predict(X_test)
    class_rep_3 = classification_report(y_test, clf_3_predictions)
    print class_rep_3

    import joblib
    joblib.dump(clf, 'model.bin', 5)

    # Average predictions from the three classifiers
    #predictions = clf.predict(test_feature_matrix)

    #output = zip([x[0] for x in test_ids], [int(x) for x in predictions])
    #output.insert(0, ["request_id", "requester_received_pizza"])

    #output_file = csv.writer(open('predictions.csv', 'w'), delimiter=",", quotechar='"')
    #for row in output:
    #    output_file.writerow(row)
