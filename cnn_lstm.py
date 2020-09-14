#Run in google colab.
#---------Nilesh Agarwal---------#
#agarwalnilesh97@gmail.com

#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip
import os
import numpy as np
import csv
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical,plot_model
from keras.layers import Activation, Dense, Dropout,Input,Add,concatenate,LSTM
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.layers import Conv1D,MaxPooling1D,Embedding,GlobalMaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
from sklearn.model_selection import StratifiedKFold
from  sklearn.metrics  import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

final_df = pd.read_csv("multilabel_dataset.csv")

X = final_df.loc[:, 'sentences'].values
Y = final_df.loc[:, 'label'].values
X_phrases = final_df.loc[:, 'vague_phrases'].values
X_cues = []

counter=0
for i in X_phrases:
    counter+=1
    if counter==5000:
        break
    cues = str(i)
    X_cues.append(cues)
X_cues = np.array(X_cues).reshape(4499);

seed = 7
#np.random.seed(seed)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

accuracy=[]
recall=[]
f1_scores=[]
precision=[]

final_predicted_sentences=[]
final_actual_labels=[]
final_predicted_labels=[]
final_actual_cue_words=[]
final_probability_class0=[]
final_probability_class1=[]

for train, test in kfold.split(X, Y, X_cues):
  for i in range(len(X[test])):
        final_predicted_sentences.append(X[test][i])
        final_actual_labels.append(Y[test][i])
        final_actual_cue_words.append(X_cues[test][i])

embedding_index = {}
with open('glove.6B.300d.txt') as f:
  for line in f:
    word, coefs = line.split(maxsplit=1)
    coefs = np.fromstring(coefs,'f',sep=' ')
    embedding_index[word] = coefs

num_labels = 4
vocab_size = 10000
batch_size = 128
embedding_dim = 300
max_len = 297

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
train_sentences_tokenized = tokenizer.texts_to_sequences(X)
X = pad_sequences(train_sentences_tokenized, maxlen=max_len)
Y = to_categorical(final_df['label'])
print(Y.shape)
print(X.shape)

print('Preparing embedding matrix')
num_words = min(vocab_size,len(word_index))+1
embedding_matrix = np.zeros((num_words,embedding_dim))
for word,i in word_index.items():
    if i > vocab_size:
        continue

    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

accuracy=[]
recall=[]
f1_scores=[]
precision=[]

final_predicted_labels=[]
final_probability_class0=[]
final_probability_class1=[]
final_probability_class2=[]
final_probability_class3=[]

Y = final_df['label']
for train, test in kfold.split(X, Y, X_cues):
  Y = to_categorical(final_df['label'])
  embedding_layer = Embedding(num_words,
                            embedding_dim,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_len,
                            trainable = True)
  sequence_input = Input(shape = (max_len,),dtype = 'int32')
  embedded_sequences = embedding_layer(sequence_input)
  x1 = Conv1D(128,3,activation = 'relu')(embedded_sequences)
  x1 = Dropout(0.2)(x1)
  x1 = GlobalMaxPooling1D()(x1)
  x2 = Conv1D(64,2,activation = 'relu')(embedded_sequences)
  x2 = Dropout(0.2)(x2)
  x2 = GlobalMaxPooling1D()(x2)
  x3 = LSTM(128)(embedded_sequences)
  x3 = Dropout(0.2)(x2)
  x = concatenate([x1,x2,x3],axis=1)

  pred = Dense(4,activation = 'softmax')(x)
  model = Model(sequence_input,pred)
  model.summary()
  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.000001, decay=0.0001)

  model.compile(loss = 'binary_crossentropy',optimizer = adam ,metrics = ['accuracy'])
  #model.summary()

  a,b = 0,4
  model.fit(X[train],Y[train][:,a:b],batch_size = batch_size,epochs = 10,validation_data = (X[test],Y[test][:,a:b]))
  scores = model.evaluate(X[test], Y[test][:,a:b], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

  pred = model.predict(X[test])
  final_probability_0 = pred[:,0]
  final_probability_1 = pred[:,1]
  final_probability_2 = pred[:,2]
  final_probability_3 = pred[:,3]

  for i in range(len(X[test])):
    final_predicted_labels.append(pred.argmax(axis=1)[i])
    final_probability_class0.append(final_probability_0[i])
    final_probability_class1.append(final_probability_1[i])
    final_probability_class2.append(final_probability_2[i])
    final_probability_class3.append(final_probability_3[i])

  c_matrix = confusion_matrix(Y[test][:,a:b].argmax(axis=1),pred.argmax(axis=1))
  print(c_matrix)
  print(classification_report(Y[test][:,a:b].argmax(axis=1),pred.argmax(axis=1)))

  plot_model(model, to_file='model.png',show_shapes = True)
  model.save('my_model_cSIN2_tag')

  accuracy.append(accuracy_score(Y[test][:,a:b].argmax(axis=1),pred.argmax(axis=1)))
  print(accuracy)
  f1_scores.append(f1_score(Y[test][:,a:b].argmax(axis=1), pred.argmax(axis=1), average='macro'))
  precision.append(precision_score(Y[test][:,a:b].argmax(axis=1), pred.argmax(axis=1), average='macro'))
  recall.append(recall_score(Y[test][:,a:b].argmax(axis=1), pred.argmax(axis=1), average='macro'))

print("accuracy :: %.4f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
print("f1_score :: %.4f%% (+/- %.2f%%)" % (np.mean(f1_scores), np.std(f1_scores)))
print("precision_score :: %.4f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
print("recall :: %.4f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))

df1 = pd.DataFrame(final_predicted_sentences)
df1.columns = ['Sentences']

df2 = pd.DataFrame(final_actual_labels)
df2.columns = ['True_Labels']

df3 = pd.DataFrame(final_predicted_labels)
df3.columns = ['Predicted_Labels']

df8 = pd.DataFrame(final_actual_cue_words)
df8.columns = ['Speculative_Words']

df4 = pd.DataFrame(final_probability_class0)
df4.columns = ['Probabilty_Score_Class0']

df5 = pd.DataFrame(final_probability_class1)
df5.columns = ['Probabilty_Score_Class1']

df6 = pd.DataFrame(final_probability_class2)
df6.columns = ['Probabilty_Score_Class2']

df7 = pd.DataFrame(final_probability_class3)
df7.columns = ['Probabilty_Score_Class3']

df = df1.join(df2)
df = df.join(df3)
df = df.join(df4)
df = df.join(df5)
df = df.join(df6)

df.to_csv('predicted_PRIVACY-POLICY_vagueness_CNN-LSTM_model.csv',index=False)

options1 = ['0']
options2 = ['1']
# selecting rows based on condition
rslt_df1 = df[(df['True_Labels'] == 1) &
          df['Predicted_Labels'].isin(options1)]

rslt_df2 = df[(df['True_Labels'] == 0) &
          df['Predicted_Labels'].isin(options2)]

frames = [rslt_df1,rslt_df2]
result_df = pd.concat(frames)
#print('\nResult dataframe :\n', r)
result_df.to_csv('incorrect_predictions_PRIVACY-POLICY_vagueness_CNN-LSTM_model.csv',index=False)

options3 = ['1']
options4 = ['0']
# selecting rows based on condition
rslt_df1 = df[(df['True_Labels'] == 1) &
          df['Predicted_Labels'].isin(options3)]

rslt_df2 = df[(df['True_Labels'] == 0) &
          df['Predicted_Labels'].isin(options4)]

frames = [rslt_df1,rslt_df2]
result_df = pd.concat(frames)
#print('\nResult dataframe :\n', r)
result_df.to_csv('correct_predictions_PRIVACY-POLICY_vagueness_CNN-LSTM_model.csv',index=False)
