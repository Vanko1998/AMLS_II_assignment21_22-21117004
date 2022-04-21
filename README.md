This project is made up of two parts: task A and task B. Each has its own files to save code: file A is for task A, file B is for task B. File datasets is used to save all raw data, including English and Arabic version of all Twitter comments of 2016. Document main.py can run all the functions in both task A and task B. 

In task A, there are data preprocessing steps and training steps(NB, RF, LSTM)
In task B, there are data preprocessing steps and training steps(KNN)




root file:
    A:
        Task-A.ipynb
        TaskA.py
    B:
        Task-B.ipynb
        TaskB.py
    Datasets:
        twitter-2016train-A.txt
        twitter-2016train-arabic.txt
        twitter-2016train-BD.txt
        twitter-2016train-BD-arabic.txt
    main.py
    README.md



There are 5 parts of files:
    A: all code of task A
    B: all code of task B
    Datasets: all raw data of twitter comments in 2016
    main.py: runnable document which realize all the functions in task A and task B
    README.md:a brief introduction of the whole file
    
   
   
To run the code, use python IDE and run main.py



Necessary packages: 
    pandas
    sys
    re
    numpy
    nltk.stem.porter
    nltk
    sacremoses
    wordcloud
    matplotlib.pyplot
    seaborn
    sklearn.feature_extraction.text
    sklearn.model_selection
    sklearn.naive_bayes
    sklearn.metrics
    sklearn.ensemble
    keras.preprocessing.text
    keras.preprocessing.sequence
    tensorflow.keras.utils
    nltk.stem
    nltk.tokenize
    nltk.corpus
    nltk.tag
    keras.models
    keras
    sklearn
    sklearn.neighbors
    sklearn.preprocessing
