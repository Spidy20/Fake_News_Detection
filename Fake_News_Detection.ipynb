{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TfidfVectorizer Explanation\n",
    "Convert a collection of raw documents to a matrix of TF-IDF features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "TF-IDF where TF means term frequency, and IDF means Inverse Document frequency."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "text = ['Hello Kushal Bhavsar here, I love machine learning','Welcome to the Machine learning hub' ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "vect = TfidfVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "                dtype=<class 'numpy.float64'>, encoding='utf-8',\n",
       "                input='content', lowercase=True, max_df=1.0, max_features=None,\n",
       "                min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None,\n",
       "                smooth_idf=True, stop_words=None, strip_accents=None,\n",
       "                sublinear_tf=False, token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b',\n",
       "                tokenizer=None, use_idf=True, vocabulary=None)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vect.fit(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1.40546511 1.40546511 1.40546511 1.40546511 1.40546511 1.\n",
      " 1.40546511 1.         1.40546511 1.40546511 1.40546511]\n"
     ]
    }
   ],
   "source": [
    "## TF will count the frequency of word in each document. and IDF \n",
    "print(vect.idf_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'hello': 1, 'kushal': 4, 'bhavsar': 0, 'here': 2, 'love': 6, 'machine': 7, 'learning': 5, 'welcome': 10, 'to': 9, 'the': 8, 'hub': 3}\n"
     ]
    }
   ],
   "source": [
    "print(vect.vocabulary_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### A words which is present in all the data, it will have low IDF value. With this unique words will be highlighted using the Max IDF values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Hello Kushal Bhavsar here, I love machine learning'"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "example = text[0]\n",
    "example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.4078241  0.4078241  0.4078241  0.         0.4078241  0.29017021\n",
      "  0.4078241  0.29017021 0.         0.         0.        ]]\n"
     ]
    }
   ],
   "source": [
    "example = vect.transform([example])\n",
    "print(example.toarray())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Here, 0 is present in the which indexed word, which is not available in given sentence."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PassiveAggressiveClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Passive: if correct classification, keep the model; Aggressive: if incorrect classification, update to adjust to this misclassified example."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Passive-Aggressive algorithms are generally used for large-scale learning. It is one of the few ‘online-learning algorithms‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Let's start the work"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.chdir(\"D:/Fake News Detection\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                                              title  \\\n",
       "0        8476                       You Can Smell Hillary’s Fear   \n",
       "1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2        3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3       10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4         875   The Battle of New York: Why This Primary Matters   \n",
       "\n",
       "                                                text label  \n",
       "0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  \n",
       "1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  \n",
       "2  U.S. Secretary of State John F. Kerry said Mon...  REAL  \n",
       "3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  \n",
       "4  It's primary day in New York and front-runners...  REAL  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe = pd.read_csv('news.csv')\n",
    "dataframe.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = dataframe['text']\n",
    "y = dataframe['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       Daniel Greenfield, a Shillman Journalism Fello...\n",
       "1       Google Pinterest Digg Linkedin Reddit Stumbleu...\n",
       "2       U.S. Secretary of State John F. Kerry said Mon...\n",
       "3       — Kaydee King (@KaydeeKing) November 9, 2016 T...\n",
       "4       It's primary day in New York and front-runners...\n",
       "                              ...                        \n",
       "6330    The State Department told the Republican Natio...\n",
       "6331    The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...\n",
       "6332     Anti-Trump Protesters Are Tools of the Oligar...\n",
       "6333    ADDIS ABABA, Ethiopia —President Obama convene...\n",
       "6334    Jeb Bush Is Suddenly Attacking Trump. Here's W...\n",
       "Name: text, Length: 6335, dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       FAKE\n",
       "1       FAKE\n",
       "2       REAL\n",
       "3       FAKE\n",
       "4       REAL\n",
       "        ... \n",
       "6330    REAL\n",
       "6331    FAKE\n",
       "6332    FAKE\n",
       "6333    REAL\n",
       "6334    REAL\n",
       "Name: label, Length: 6335, dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2402    REAL\n",
       "1922    REAL\n",
       "3475    FAKE\n",
       "6197    REAL\n",
       "4748    FAKE\n",
       "        ... \n",
       "4931    REAL\n",
       "3264    REAL\n",
       "1653    FAKE\n",
       "2607    FAKE\n",
       "2732    REAL\n",
       "Name: label, Length: 5068, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)\n",
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2402    REAL\n",
       "1922    REAL\n",
       "3475    FAKE\n",
       "6197    REAL\n",
       "4748    FAKE\n",
       "        ... \n",
       "4931    REAL\n",
       "3264    REAL\n",
       "1653    FAKE\n",
       "2607    FAKE\n",
       "2732    REAL\n",
       "Name: label, Length: 5068, dtype: object"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)\n",
    "tfid_x_train = tfvect.fit_transform(x_train)\n",
    "tfid_x_test = tfvect.transform(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* max_df = 0.50 means \"ignore terms that appear in more than 50% of the documents\".\n",
    "* max_df = 25 means \"ignore terms that appear in more than 25 documents\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,\n",
       "                            early_stopping=False, fit_intercept=True,\n",
       "                            loss='hinge', max_iter=50, n_iter_no_change=5,\n",
       "                            n_jobs=None, random_state=None, shuffle=True,\n",
       "                            tol=0.001, validation_fraction=0.1, verbose=0,\n",
       "                            warm_start=False)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classifier = PassiveAggressiveClassifier(max_iter=50)\n",
    "classifier.fit(tfid_x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 93.69%\n"
     ]
    }
   ],
   "source": [
    "y_pred = classifier.predict(tfid_x_test)\n",
    "score = accuracy_score(y_test,y_pred)\n",
    "print(f'Accuracy: {round(score*100,2)}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[575  40]\n",
      " [ 40 612]]\n"
     ]
    }
   ],
   "source": [
    "cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])\n",
    "print(cf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fake_news_det(news):\n",
    "    input_data = [news]\n",
    "    vectorized_input_data = tfvect.transform(input_data)\n",
    "    prediction = classifier.predict(vectorized_input_data)\n",
    "    print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['REAL']\n"
     ]
    }
   ],
   "source": [
    "fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['FAKE']\n"
     ]
    }
   ],
   "source": [
    "fake_news_det(\"\"\"Go to Article \n",
    "President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  \"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(classifier,open('model.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load the model from disk\n",
    "loaded_model = pickle.load(open('model.pkl', 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fake_news_det1(news):\n",
    "    input_data = [news]\n",
    "    vectorized_input_data = tfvect.transform(input_data)\n",
    "    prediction = loaded_model.predict(vectorized_input_data)\n",
    "    print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['FAKE']\n"
     ]
    }
   ],
   "source": [
    "fake_news_det1(\"\"\"Go to Article \n",
    "President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  \"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['REAL']\n"
     ]
    }
   ],
   "source": [
    "fake_news_det1(\"\"\"U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['REAL']\n"
     ]
    }
   ],
   "source": [
    "fake_news_det('''U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.''')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
