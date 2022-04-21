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
from sklearn.naive_bayes import MultinomialNB  # naive bayes model
from sklearn.ensemble import RandomForestClassifier  # random forest model
from keras.preprocessing.text import Tokenizer  # LSTM model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import learning_curve


def to_csv(address):  # change txt document to csv file
    data=open(address,'r',encoding='utf-8').readlines()
    ID=[]
    Type=[]
    Content=[]
    for row in data:
        ID.append(row[0:18])
        if not (row[19:26]=="neutral"):
            Type.append(row[19:27])
        else:
            Type.append(row[19:26])
        Content.append(row[28:])
    df_list={"ID":ID,"Type":Type,"Content":Content}
    twitter=pd.DataFrame(df_list,columns=['ID','Type','Content','tidy_Content'])
    return twitter


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


'''def plot_wordcloud(twitter_df,type):  # plot a word cloud for all words
    all_words=" ".join([text for text in twitter_df['tidy_Content'][twitter_df['Type']==type]])
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
    plt.show()'''


def preprocess_y(twitter_df,language):  # encode y from positive, negative and neutral to 1,-1,0
    if language=='english':
        tfidf_vectorizer=TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    elif language=='arabic':
        tfidf_vectorizer=TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
    # TF-IDF feature matrix
    tfidf=tfidf_vectorizer.fit_transform(twitter_df['tidy_Content'])
    y_validation=pd.DataFrame(columns = ['Type'])
    y_validation['Type']=twitter_df['Type'].copy()
    #1:positive   0:negative   2:neutral
    for i in range(0,y_validation['Type'].size):
        if y_validation['Type'][i]=='positive':
            y_validation['Type'][i]=1
        elif y_validation['Type'][i]=='negative':
            y_validation['Type'][i]=0
        elif y_validation['Type'][i]=='neutral':
            y_validation['Type'][i]=2
    return y_validation,tfidf


def naive_bayes(y_validation,tfidf):  # use naive bayes model, split train, validation and test dataset, fit model, and print accuracy
    x_train_valid, x_test_valid, y_train_valid, y_test_valid=train_test_split(tfidf[:4800,:],y_validation['Type'][:4800].values.astype('int'), random_state=0, test_size=0.25)
    x_train, x_test,y_train,y_test=train_test_split(tfidf, y_validation['Type'].astype('int'), random_state=0, test_size=0.2)
    # fit english version data
    nb_valid=MultinomialNB()
    nb_valid.fit(x_train_valid,y_train_valid)
    naive_A_valid=accuracy_score(y_test_valid,nb_valid.predict(x_test_valid))
    print('naive bayes validation accuracy:')
    print(naive_A_valid)
    # fit arabic version data
    nb_test=MultinomialNB()
    draw_learning_curves(x_train, y_train, nb_test, 10, 'NB')
    nb_test.fit(x_train,y_train)
    naive_A_test=accuracy_score(y_test, nb_test.predict(x_test))
    print('naive bayes test accuracy:')
    print(naive_A_test)
    return naive_A_valid,naive_A_test


def random_forest(y_validation,tfidf):  # use random forest model, split train, validation and test dataset, fit model, and print accuracy
    x_train_valid, x_test_valid, y_train_valid, y_test_valid=train_test_split(tfidf[:4800,:], y_validation['Type'][:4800].values.astype('int'), random_state=0, test_size=0.25)
    x_train, x_test,y_train,y_test=train_test_split(tfidf, y_validation['Type'].astype('int'), random_state=0, test_size=0.2)
    # fit english version data
    rf_valid=RandomForestClassifier(n_estimators=500,max_depth=20)
    rf_valid.fit(x_train_valid,y_train_valid)
    random_A_valid=accuracy_score(y_test_valid,rf_valid.predict(x_test_valid))
    print('random forest valid accuracy:')
    print(random_A_valid)
    # fit arabic version data
    rf_test=RandomForestClassifier(n_estimators=500,max_depth=20)
    draw_learning_curves(x_train, y_train, rf_test, 10, 'RF')
    rf_test.fit(x_train,y_train)
    random_A_test=accuracy_score(y_test,rf_test.predict(x_test))
    print('random forest test accuracy:')
    print(random_A_test)
    return random_A_valid,random_A_test


def data_cleaning(text_list):  # clean data for LSTM model
    stopwords_rem=False
    stopwords_en=stopwords.words('english')
    lemmatizer=WordNetLemmatizer()
    tokenizer=TweetTokenizer()
    reconstructed_list=[]
    # manually write word tags
    for each_text in text_list: 
        lemmatized_tokens=[]
        tokens=tokenizer.tokenize(each_text.lower())
        pos_tags=pos_tag(tokens)
        for each_token, tag in pos_tags: 
            if tag.startswith('NN'): 
                pos='n'
            elif tag.startswith('VB'): 
                pos='v'
            elif tag.startswith('JJ'): 
                pos='a'
            elif tag.startswith('R'):
                pos='r'
            lemmatized_token=lemmatizer.lemmatize(each_token, pos)
            if stopwords_rem: # False 
                if lemmatized_token not in stopwords_en: 
                    lemmatized_tokens.append(lemmatized_token)
            else: 
                lemmatized_tokens.append(lemmatized_token)
        # append all the sentences
        reconstructed_list.append(' '.join(lemmatized_tokens))
    return reconstructed_list


def LSTM_model(twitter_df,y_validation,language,epochs):
    # Decompose the data into training sets and test sets
    lstm_df=pd.DataFrame(columns=['tidy_Content'])
    lstm_df['tidy_Content']=np.vectorize(remove_pattern)(twitter_df['Content'], "@[\w]*")
    lstm_df['tidy_Content']=np.vectorize(remove_pattern)(lstm_df['tidy_Content'], r"#(\w+)")
    lstm_df['tidy_Content']=np.vectorize(remove_pattern)(lstm_df['tidy_Content'], r'http://[a-zA-Z0-9.?/&=:]*')
    if language == 'english':
        lstm_df['tidy_Content']=lstm_df['tidy_Content'].str.replace("[^a-zA-Z#]", " ",regex=True)
    lstm_df['tidy_Content']=lstm_df['tidy_Content'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    X_train, X_test,y_train,y_test=train_test_split(lstm_df['tidy_Content'], y_validation['Type'], random_state=0, test_size=0.2)
    # to merge transform data
    X_train=data_cleaning(X_train)
    X_test=data_cleaning(X_test)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size=len(tokenizer.word_index)+1
    print(f'Vocab Size: {vocab_size}')
    X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=40)
    X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=40)
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    # create LSTM model with embedding layer and fit training data
    model=Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,output_dim=100,input_length=40))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dense(3,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=256,epochs=epochs,validation_data=(X_test,y_test))


def draw_learning_curves(X, y, estimator, num_trainings,title):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=2, train_sizes=np.linspace(.1, 1.0, num_trainings))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.title(title + " Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_scores_mean, 'o-', color="g",label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

