import sys
sys.path.append('./A/')
sys.path.append('./B/')
import TaskA as A
import TaskB as B

# ======================================================================================================================
# Task A Data preprocessing
print('Task A English preprocessing:')
twitter_df=A.to_csv('./Datasets/twitter-2016train-A.txt')
twitter_df=A.preprocess(twitter_df,'english')
# Plot word cloud to see the frequency of all words
A.plot_wordcloud(twitter_df,'positive')
A.plot_wordcloud(twitter_df,'negative')
A.plot_wordcloud(twitter_df,'neutral')
# extracting hashtags from positive tweets
HT_positive = A.hashtag_extract(twitter_df['Content'][twitter_df['Type']=='positive'])
# extracting hashtags from negative tweets
HT_negative = A.hashtag_extract(twitter_df['Content'][twitter_df['Type']=='negative'])
# extracting hashtags from neutral tweets
HT_neutral = A.hashtag_extract(twitter_df['Content'][twitter_df['Type']=='neutral'])
# unnesting list
HT_positive=sum(HT_positive,[])
HT_negative=sum(HT_negative,[])
HT_neutral=sum(HT_neutral,[])
# Plot histogram to see the quantity of all words
A.plot_histogram(HT_positive)
A.plot_histogram(HT_negative)
A.plot_histogram(HT_neutral)
y_validation,tfidf=A.preprocess_y(twitter_df,'english')
print('***************************************************************************************************************')
print('Task A Arabic preprocessing:')
twitter_df_a=A.to_csv('./Datasets/twitter-2016train-A-arabic.txt')
twitter_df_a=A.preprocess(twitter_df_a,'arabic')
# Plot word cloud to see the frequency of all words
A.plot_wordcloud(twitter_df_a,'positive')
A.plot_wordcloud(twitter_df_a,'negative')
A.plot_wordcloud(twitter_df_a,'neutral')
# extracting hashtags from positive tweets
HT_positive_a = A.hashtag_extract(twitter_df_a['Content'][twitter_df_a['Type']=='positive'])
# extracting hashtags from negative tweets
HT_negative_a = A.hashtag_extract(twitter_df_a['Content'][twitter_df_a['Type']=='negative'])
# extracting hashtags from neutral tweets
HT_neutral_a = A.hashtag_extract(twitter_df_a['Content'][twitter_df_a['Type']=='neutral'])
# unnesting list
HT_positive_a=sum(HT_positive_a,[])
HT_negative_a=sum(HT_negative_a,[])
HT_neutral_a=sum(HT_neutral_a,[])
# Plot histogram to see the quantity of all words
A.plot_histogram(HT_positive_a)
A.plot_histogram(HT_negative_a)
A.plot_histogram(HT_neutral_a)
y_validation_a,tfidf_a=A.preprocess_y(twitter_df_a,'arabic')
# ======================================================================================================================
# Task A
print('***************************************************************************************************************')
print('FOR ENGLISH:')
naive_A_English_valid,naive_A_English_test=A.naive_bayes(y_validation,tfidf)
random_A_English_valid,random_A_English_test=A.random_forest(y_validation,tfidf)
A.LSTM_model(twitter_df,y_validation,'english')
print('FOR ARABIC:')
naive_A_Arabic_valid,naive_A_Arabic_test=A.naive_bayes(y_validation_a,tfidf_a)
random_A_Arabic_valid,random_A_Arabic_test=A.random_forest(y_validation_a,tfidf_a)
A.LSTM_model(twitter_df_a,y_validation_a,'arabic')
# ======================================================================================================================
# Task B Data preprocessing
print('***************************************************************************************************************')
print('Task B English preprocessing:')
twitter_df=B.to_csv('./Datasets/twitter-2016train-BD.txt')
twitter_df=B.preprocess(twitter_df,'english')
print('Task B Arabic preprocessing:')
twitter_df_a=B.to_csv('./Datasets/twitter-2016train-BD-arabic.txt')
twitter_df_a=B.preprocess(twitter_df_a,'arabic')
# ======================================================================================================================
# Task B
print('***************************************************************************************************************')
print('FOR ENGLISH:')
all_event=twitter_df['Event'].value_counts()
all_event=all_event.index.values
v_avg_k=0
t_avg_k=0
v_avg_s=0
t_avg_s=0
# train and test all kinds of events
for event in all_event:
    df=B.one_event(event,twitter_df)
    #print(event)
    #naive_bayes(y_validation,tfidf)
    #acc_v_s,acc_t_s=svm_model(df,'english')
    acc_v_k,acc_t_k=B.knn_model(df,'english')
    v_avg_k+=acc_v_k
    t_avg_k+=acc_t_k
    #v_avg_s+=acc_v_s
    #t_avg_s+=acc_t_s
v_avg_k=v_avg_k/60
t_avg_k=t_avg_k/60
#v_avg_s=v_avg_s/60
#t_avg_s=t_avg_s/60
print('knn valid average accuracy:')
print(v_avg_k)
print('knn test accuracy:')
print(t_avg_k)
#print('svm valid average accuracy:')
#print(v_avg_s)
#print('svm test accuracy:')
#print(t_avg_s)

print('FOR ARABIC:')
all_event_a=twitter_df_a['Event'].value_counts()
all_event_a=all_event_a.index.values
v_avg_k_a=0
t_avg_k_a=0
v_avg_s_a=0
t_avg_s_a=0
# train and test all kinds of events
for event in all_event_a:
    df=B.one_event(event,twitter_df_a)
    #print(event)
    #acc_v_s,acc_t_s=svm_model(df,'arabic')
    acc_v_k_a,acc_t_k_a=B.knn_model(df,'arabic')
    v_avg_k_a+=acc_v_k_a
    t_avg_k_a+=acc_t_k_a
    #v_avg_s+=acc_v_s
    #t_avg_s+=acc_t_s
v_avg_k_a=v_avg_k_a/34
t_avg_k_a=t_avg_k_a/34
#v_avg_s=v_avg_s/60
#t_avg_s=t_avg_s/60
print('knn valid average accuracy:')
print(v_avg_k_a)
print('knn test accuracy:')
print(t_avg_k_a)
#print('Arabic svm valid average accuracy:')
#print(v_avg_s)
#print('Arabic svm test accuracy:')
#print(t_avg_s)
# ======================================================================================================================
# Print out results with following format:
print('Pick highest accuracy value model:')
print('For task A: random forest(English), naive bayes(Arabic)')
print('For task B: knn')
print('English TA:{},{};TB:{},{};'.format(random_A_English_valid,random_A_English_test, v_avg_k, t_avg_k))
print('Arabic TA:{},{};TB:{},{};'.format(naive_A_English_valid,naive_A_English_test, v_avg_k_a, t_avg_k_a))


