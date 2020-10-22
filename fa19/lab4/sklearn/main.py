import argparse
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--method', type=str, required=False, default='nb')
    arg_parser.add_argument('--data_path', type=str, required=False, default='/Users/timowang/Entwickler/data/text-analytics/lab4/Consumer_Complaints.csv')
    args = arg_parser.parse_args()
    method = args.method
    data_path = args.data_path
    df = load_consumer_data(data_path)
    category_id_df = df[[
        'Product',
        'category_id'
    ]].drop_duplicates().sort_values('category_id')
    category2id = dict(category_id_df.values)
    id2category = dict(category_id_df[['category_id', 'Product']].values)
    
    X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state=0)
    
    if method == 'nb':
        fit_and_test(MultinomialNB(), X_train, y_train, X_test, y_test)
    elif method == 'logreg':
        fit_and_test(LogisticRegression(random_state=0), X_train, y_train, X_test, y_test)
    elif method == 'svc':
        fit_and_test(LinearSVC(), X_train, y_train, X_test, y_test)
    else:
        raise NotImplementedError


def fit_and_test(model, X_train, y_train, X_test, y_test):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = model.fit(X_train_tfidf, y_train)
    print(clf.predict(tfidf_transformer.transform(count_vect.transform([
        "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]))))
    print(clf.predict(tfidf_transformer.transform(count_vect.transform([
        "I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"]))))
    
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
    print('Accuracy is: %f' % accuracy_score(y_pred, y_test))


# def fit_and_test_logistic_reg(X_train, y_train):
#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(X_train)
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     clf = MultinomialNB().fit(X_train_tfidf, y_train)
#
#
# def fit_and_test_nb(X_train, y_train):
#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(X_train)
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#     clf = MultinomialNB().fit(X_train_tfidf, y_train)
#     print(clf.predict(count_vect.transform([
#                                                "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."])))
#     print(clf.predict(count_vect.transform([
#                                                "I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"])))


def load_consumer_data(data_path):
    df = pd.read_csv(data_path)
    df = df[:100000]
    col = ['Product', 'Consumer complaint narrative']
    df = df[col]
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    df.columns = ['Product', 'Consumer_complaint_narrative']
    df['category_id'] = df['Product'].factorize()[0]
    return df


if __name__ == '__main__':
    main()