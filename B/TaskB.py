import pandas as pd
import numpy as np
from nltk.stem.porter import *
import nltk
from sacremoses import MosesDetokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def to_csv(address):  # change txt document to csv file
    twitter_df=pd.read_csv(address, sep="\t",names=['ID','Event','Type','Content'])
    return twitter_df


def remove_pattern(input_txt,pattern):  # a function to remove patterns, like: @xxxxxx, #xxxxx
    r=re.findall(pattern, input_txt)
    for i in r:
        input_txt=re.sub(i, '', input_txt)
    return input_txt  


def preprocess(twitter_df,language):  # remove all unnecessary content in comments
    twitter_df['tidy_Content']=np.vectorize(remove_pattern)(twitter_df['Content'], "@[\w]*")
    twitter_df['tidy_Content']=np.vectorize(remove_pattern)(twitter_df['tidy_Content'], r"#(\w+)")
    twitter_df['tidy_Content']=np.vectorize(remove_pattern)(twitter_df['tidy_Content'], r'http://[a-zA-Z0-9.?/&=:]*')
    if language=='english':
        twitter_df['tidy_Content']=twitter_df['tidy_Content'].str.replace("[^a-zA-Z#]", " ",regex=True)
    twitter_df['tidy_Content']=twitter_df['tidy_Content'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    # a new column for cleaned comments
    token_tweet=twitter_df['tidy_Content'].apply(lambda x: x.split())
    stemmer=PorterStemmer()
    token_tweet=token_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
    # combine all valuable words back to sentences
    detokenizer=MosesDetokenizer()
    for i in range(len(token_tweet)):
        token_tweet[i]=detokenizer.detokenize(token_tweet[i], return_str=True)
    twitter_df['tidy_Content']=token_tweet
    twitter_df['tidy_Content']=twitter_df['tidy_Content'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    return twitter_df


'''def plot_wordcloud(twitter_df,event):  # plot a word cloud for all words
    all_words=" ".join([text for text in twitter_df['tidy_Content'][twitter_df['Event']==event]])
    wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def hashtag_extract(df):  # function to collect hashtags
    hashtags = []
    # Loop over the words in the tweet
    for i in df:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def plot_histogram(HT):  # plot a histogram for all kinds of words
    a = nltk.FreqDist(HT)
    d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()
'''

def preprocess_y(twitter_df,language):  # encode y from positive, negative and neutral to 1,-1,0
    if language=='english':
        tfidf_vectorizer=TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    elif language=='arabic':
        tfidf_vectorizer=TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
    # TF-IDF feature matrix
    tfidf=tfidf_vectorizer.fit_transform(twitter_df['tidy_Content'])
    y_validation=pd.DataFrame(columns = ['Type'])
    y_validation['Type']=twitter_df['Type'].copy()
    #1:positive   0:negative
    for i in range(0,y_validation['Type'].size):
        if y_validation['Type'][i]=='positive':
            y_validation['Type'][i]=1
        elif y_validation['Type'][i]=='negative':
            y_validation['Type'][i]=0
    return y_validation,tfidf


'''def plot_one_event(event,twitter_df):  # plot a word cloud figure and a histogram for one specific event
    plot_wordcloud(twitter_df,event)
    # extracting hashtags from positive tweets
    HT_event= hashtag_extract(twitter_df['Content'][twitter_df['Event']==event])
    # unnesting list
    HT_event=sum(HT_event,[])
    plot_histogram(HT_event)'''


def one_event(event,twitter_df):  # a function to select all comments based on one specific event name
    df=pd.DataFrame(columns=['ID','Type','Content','tidy_Content'])
    for i in range(0,twitter_df.index.size):
        if twitter_df['Event'][i]==event:
            df_new=pd.DataFrame(twitter_df.iloc[i])
            df_new_T=pd.DataFrame(df_new.values.T, index=df_new.columns, columns=df_new.index)
            df=pd.concat([df,df_new_T],axis=0)
            df=df.reset_index(drop=True)
    return df


def knn_design(X_train,X_test, y_train,y_test,k):  # for Knn model
    # classify data
    std = StandardScaler(with_mean=False)
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model
    return accuracy_score(y_test,neigh.predict(X_test))


def knn_model(twitter_df,language,k):  # train, valid and test KNN model, and return accuracy
    y_validation,tfidf=preprocess_y(twitter_df,language)
    x_train, x_test,y_train,y_test=train_test_split(tfidf, y_validation['Type'].astype('int'), random_state=42, test_size=0.2)
    x_train_valid, x_test_valid, y_train_valid, y_test_valid=train_test_split(tfidf[:(y_validation.size*4)//5],y_validation['Type'].astype('int')[:(y_validation.size*4)//5], random_state=None, test_size=0.25)
    # fit and predict
    acc_train=knn_design(x_train_valid, x_train_valid, y_train_valid, y_train_valid,k)
    acc_valid=knn_design(x_train_valid, x_test_valid, y_train_valid, y_test_valid,k)
    acc_test=knn_design(x_train, x_test,y_train,y_test,5)
    return acc_train,acc_valid,acc_test


def KNN_compare(all_event,twitter_df,language,i_max,event_number):#for binary-split task
    score_list=[]
    for i in range(1,i_max):
        #knn = KNeighborsClassifier(n_neighbors=i)#classify data
        #knn.fit(tr_X,tr_Y)#train model
        #pred_i = knn.predict(te_X)#make prediction using test datasets
        tr_avg,v_avg,t_avg=train_all_event(all_event, twitter_df, language,i,event_number)
        score_list.append(v_avg)
    plt.plot(range(1,i_max),score_list,color='pink', linestyle='dashed', marker='o', markerfacecolor='grey',markersize=10)
    plt.title("Accuracy vs. K Value (Binary)")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()#make a chart to show the results


def train_all_event(all_event,twitter_df,language,k,event_number):
    tr_avg_k=0
    v_avg_k=0
    t_avg_k=0
    # train and test all kinds of events
    for event in all_event:
        df=one_event(event,twitter_df)
        acc_tr_k,acc_v_k,acc_t_k=knn_model(df,language,k)
        tr_avg_k+=acc_tr_k
        v_avg_k+=acc_v_k
        t_avg_k+=acc_t_k
    tr_avg_k = tr_avg_k / event_number
    v_avg_k=v_avg_k/event_number
    t_avg_k=t_avg_k/event_number
    return tr_avg_k,v_avg_k,t_avg_k