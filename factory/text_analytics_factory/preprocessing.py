
import spacy
import string
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return (texts)


def find_optimal_clusters(dataframe,column_name, num_cluster_search, step_size=2):
    tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'    )
    tfidf.fit(dataframe[column_name])
    data = tfidf.transform(dataframe[column_name]) 
    iters = range(2, num_cluster_search+1, step_size)
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    top_keywords = {}
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        top_keywords[i] = ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])
    return top_keywords
            

def kmeans_wtfidf(dataframe, column_name, number_of_clusters=100, cluster_id="ClusterNumber", name_of_cluster = 'ClusterName'):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    dataframe = dataframe.reset_index(drop=True)
    docs = dataframe[column_name]
    vectorizer = TfidfVectorizer(max_features = 2000, ngram_range=(1,1), max_df=int(len(docs)*0.1))
    vectors = vectorizer.fit_transform(tuple(docs))
    x_train = np.array(vectors.toarray())
    k_mean = KMeans(n_clusters=number_of_clusters)
    k_mean.fit(x_train)
    labels = k_mean.labels_
    print("The K Mean Score is :", k_mean.score(x_train))
    top_keys = get_top_keywords(vectors, labels, vectorizer.get_feature_names(), 10)
    dataframe[cluster_id]=labels
    dataframe[name_of_cluster] = dataframe[cluster_id].map(top_keys)
    return dataframe