# Sentiment Analysis #


> **Disclaimer :**
> 
> This project is for educational purposes.
> 
> **Tujuan :**
> 
> Untuk menganalisis kinerja BERT (*Bidirectional Encoder Representations from Transformers*) sebagai metode representasi teks pada model CNN-BiLSTM dan BiLSTM-CNN untuk analisis sentimen berbahasa Indonesia.


## 1. Import Modul ##
Berikut merupakan modul yang akan digunakan dalam analisis ini.
``` python
import re
from keras_tuner.tuners import BayesianOptimization
from keras.layers import Bidirectional, LSTM
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModel
import IPython
```


## 2. Pre-Processing Data ##
Proses ini terdiri atas tiga tahap, yaitu pembersihan data, *one hot encoding*, dan pemisahan data.
### Pembersihan Data ###
Tahapan *pre-processing data* untuk pembersihan data terdiri dari:
1. Mengubah kata singkatan menggunakan kamus dari [penelitian Salsabila, et. al. (2018)](https://github.com/nasalsabila/kamus-alay)
2. Mengubah semua huruf besar menjadi huruf kecil
3. Menghapus www. dan htps://
4. Menghapus @ (*username*) dan # (*hashtag*)
5. Menghapus tanda baca
6. Menghapus angka
7. Menghapus tanda \\, ', ",
8. Menghapus emoji
9. Menghapus spasi berlebih
10. Memisahkan kata berulang
11. Menghapus huruf berulang yang lebih dari dua kali
12. Memeriksa apakah suatu kata terbentuk dari minimal dua huruf

``` python
# Import kamus untuk mengubah singkatan
kamus_alay = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')

# Membuat dictionary untuk memetakan singkatan dari kata
nor_dict = {}
for index, row in kamus_alay.iterrows():
    if row[0] not in nor_dict:
        nor_dict[row[0]] = row[1]

# Pembersihan data
def clean_text(tweet):
    # Mengubah semua huruf menjadi huruf kecil
    tweet = tweet.lower()
    # Menghapus www.* atau https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    # Menghapus @username
    tweet = re.sub('@[^\s]+','',tweet)
    # Menghapus tanda #
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Menghapus tanda baca
    tweet = re.sub(r'[^\w\s]',' ', tweet)
    # Menghapus angka
    tweet = re.sub(r'[\d-]', '', tweet)
    # Menghapus spasi berlebih
    tweet = re.sub('[\s]+', ' ', tweet)
    # Menghapus tanda \, ', dan "
    tweet = tweet.strip('\'"')
    
    # Pembersihan kata
    words = tweet.split()
    tokens=[]
    for ww in words:
        # Memisahkan kata berulang
        for w in re.split(r'[-/\s]\s*', ww):
            # Menghapus huruf berulang yang lebih dari dua kali
            pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
            w = pattern.sub(r"\1\1", w)
            w = w.strip('\'"?,.')
            # Memeriksa apakah suatu kata terbentuk dari minimal dua huruf
            val = re.search(r"^[a-zA-Z][a-zA-Z][a-zA-Z]*$", w)
            if w in nor_dict:
                w = nor_dict[w]
            if w == "rt" or val is None:
                continue
            else:
                tokens.append(w.lower())
    tweet = " ".join(tokens)  
    return tweet

# Pembersihan dataset
tokped['review'] = tokped['review'].map(lambda x: clean_text(x))
tokped = tokped[tokped['review'].apply(lambda x: len(x.split()) >=1)]
data = np.array(tokped['review'])
```

Berikut merupakan contoh hasil dari pembersihan data.
| Sebelum | Sesudah |
|:---:|:---:|
| okelah... mohon maaf apabila saran saya ini kurang berkenan.. sekarang ini negeri kita ada wabah penyakit yg sangat mematikan.. untuk pihak terkait, misalnya seorang jasa pengiriman/ kurir skrg di wajib kan dites kesehatan.. Krn kita sebagai konsumen tdk Thu, apakah barang pesanan kita itu terkontaminasi atau tdk.. Krn kita merasa senang bila pesanan kita sdh sampai dirumah.. dan kita langsung memegangnya..mohonmaaf sekali lg..🙏🙏🙏 | okelah mohon maaf apabila saran saya ini kurang berkenan sekarang ini negeri kita ada wabah penyakit yang sangat mematikan untuk pihak terkait misalnya seorang jasa pengiriman kurir sekarang di wajib kan dites kesehatan karena kita sebagai konsumen tidak tuh apakah barang pesanan kita itu terkontaminasi atau tidak karena kita merasa senang bila pesanan kita sudah sampai dirumah dan kita langsungmemegang nyamohon maaf sekali lagi |

### One Hot Encoding Pada Label ###
Label dari data merepresentasikan sentimen positif dan sentimen negatif yang merupakan variabel kategorik. Maka hal tersebut, dilakukan proses *one hot encoding* untuk mengubah label menjadi variabel numerik. Pada analisis ini, setiap label data akan diubah menjadi vektor berdimensi dua yang bernilai 0 atau 1.

``` python
label = np.array(pd.get_dummies(tokped['sentiment']))
```

Berikut gambaran tahapan ini:
Ulasan | Label | Array
|:---|:---:| :---: |
| Positif | 1 | [0 , 1] |
| Negatif | 0 | [1 , 0] |


### Pemisahan Data ###
Jumlah dari *data training* adalah 80% (3488 data), dan *data testing* sebesar 20% (872 data) dari total seluruh data.
``` python
data_train,data_test,label_train,label_test = train_test_split(data, label, test_size=0.2, stratify=label, random_state=24)
```


## 3. Model dan Tokenizer IndoBERT ##
Representasi data menggunakan metode BERT:
1. Tokenisasi *input* menggunakan BERT tokenizer.
2. Membatasi banyak token pada tiap satu data.
3. Menyamakan panjang dokumen dengan *padding* dan *truncation*.
4. Mengubah token menjadi bilangan bulat sehingga *input* dapat dibaca oleh model BERT dengan tahapan numerikalisasi (*numericalization*), dengan memanfaatkan *vocabulary* dari model WordPiece yang terdiri dari 30,522 pasangan token dan bilangan bulat unik yang bersesuaian.

Pada analisis ini, akan digunakan [model *pretrained* IndoBERT Base (*phase2 - uncased*)](https://huggingface.co/indobenchmark/indobert-base-p2).

### Memanggil model dan tokenizer ###
``` python
bert_model = TFAutoModel.from_pretrained("indobenchmark/indobert-base-p2", trainable=False)
bert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
```

### Melakukan tokenasi pada data
``` python
# Pendefinisian fungsi untuk melakukan tokenisasi pada satu data
def tokenisasi(teks):
      encode_dict = bert_tokenizer(teks,
                                   add_special_tokens = True,
                                   max_length = 128, #maximum token per kalimat = 128
                                   padding = 'max_length',
                                   truncation = True,
                                   return_attention_mask = True,
                                   return_tensors = 'tf',)

      tokenID = encode_dict['input_ids']
      attention_mask = encode_dict['attention_mask']

      return tokenID, attention_mask

# Pendefinisian fungsi untuk mengambil hasil tokenisasi pada semua data
def create_input(data):
    tokenID, input_mask = [], []
    for teks in data:
        token, mask = tokenisasi(teks)
        tokenID.append(token)
        input_mask.append(mask)
    
    return [np.asarray(tokenID, dtype=np.int32).reshape(-1, 128), 
            np.asarray(input_mask, dtype=np.int32).reshape(-1, 128)]
```

### Membuat tokenID dan attention mask untuk data train dan test ###
``` python
X_train = create_input(data_train)
X_test = create_input(data_test)
```

Berikut merupakan contoh hasil dari tokenisasi data.
| Sebelum | Sesudah |
|:---:|:---:|
| okelah mohon maaf apabila saran saya ini kurang berkenan sekarang ini negeri kita ada wabah penyakit yang sangat mematikan untuk pihak terkait misalnya seorang jasa pengiriman kurir sekarang di wajib kan dites kesehatan karena kita sebagai konsumen tidak tuh apakah barang pesanan kita itu terkontaminasi atau tidak karena kita merasa senang bila pesanan kita sudah sampai dirumah dan kita langsung memegang nyamohon maaf sekali lagi | 2, 4595, 212, 2903, 2727, 1496, 3386, 209, 92, 1057, 11686, 747, 92, 1202, 219, 176, 18465, 976, 34, 310, 7706, 90, 1241, 1780, 1330, 596, 1416, 3511, 12065, 747, 26, 2354, 951, 350, 53, 964, 211, 219, 242, 2608, 119, 4080, 937, 963, 6012, 219, 137, 21713, 158, 119, 211, 219, 1259, 3000, 1063, 6012, 219, 259, 493, 8161, 41, 219, 728, 5082, 1107, 2903, 2727, 684, 423, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


## 4. Analisis Sentimen dengan Model CNN-BiLSTM ##
``` python
#Model : BERT + CNN-BiLSTM
def cnn_bilstm(hp):
    #Input layer
    input_token = keras.layers.Input(shape=(128,), dtype=np.int32,
                                        name="input_token")
    input_mask = keras.layers.Input(shape=(128,), dtype=np.int32,
                                   name="input_mask")
    #Embedding
    bert_embedding = bert_model([input_token, input_mask])[0]
    #Convolution layer
    cnn = keras.layers.Conv1D(filters = hp.Int('filters',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                              kernel_size = hp.Int('kernel_size',
                                                    min_value = 3, 
                                                    max_value = 5, 
                                                    step = 1),
                              activation='relu',
                              kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn',
                                                                                    values = [0.01, 0.001])))(bert_embedding)
    
    #Max Pooling layer
    maxpool = keras.layers.MaxPooling1D(pool_size=2)(cnn)
    
    #BiLSTM layer
    bilstm1 = keras.layers.Bidirectional(LSTM(units = hp.Int('units',
                                                     min_value = 100,
                                                     max_value = 200,
                                                     step = 50),
                                      kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_regularizer',
                                                                                         values = [0.01, 0.001])),
                                      recurrent_regularizer=keras.regularizers.l2(hp.Choice('rec_regularizer',
                                                                                            values = [0.01, 0.001]))))(maxpool)
    
    #Dropout layer
    lstm_out = keras.layers.Dropout(0.5)(bilstm1)
 
    #Output layer
    output = keras.layers.Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_dense', values = [0.01, 0.001])))(lstm_out)
    model1 = keras.models.Model(inputs=[input_token, input_mask], outputs=output)
 
    model1.compile(optimizer = keras.optimizers.Adam(1e-3),
                  loss ='categorical_crossentropy',
                  metrics=['accuracy',
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall')])
 
    return model1

#Hyperparameter tuning
tuner = BayesianOptimization(cnn_bilstm,
                             objective = 'val_accuracy', 
                             max_trials = 10,
                             directory = '/content/Hasil',
                             project_name = 'Sentimen-CNN-BiLSTM',
                             overwrite = True)

# Pendefinisian Callback
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, label_train,
             batch_size=32, epochs=10,
             validation_data=(X_test, label_test),
             callbacks=[early_stop, ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print('\nThe hyperparameter search is complete.'
      '\nfilters:', best_hps.get('filters'),
      '\nkernel_size:', best_hps.get('kernel_size'),
      '\nkernel_cnn:', best_hps.get('kernel_cnn'),
      '\nunits:', best_hps.get('units'),
      '\nkernel_regularizer:', best_hps.get('kernel_regularizer'),
      '\nrec_regularizer:', best_hps.get('rec_regularizer'),
      '\nkernel_dense:', best_hps.get('kernel_dense'))

# Mendapatkan model terbaik
model1 = tuner.get_best_models()[0]

# Mendapatkan kinerja model
y_pred = np.argmax(model1.predict(X_test), axis=1)
y = np.argmax(label_test, axis=1)
print('accuracy: ', accuracy_score(y, y_pred), 
      '\nprecicion: ', precision_score(y, y_pred), 
      '\nrecall: ', recall_score(y, y_pred))
```

Setelah dilakukan lima kali *running*, didapatkan hasil sebagai berikut.
| Metrics | Average | Standar Deviasi |
|:---|:---:|:---:|
| Akurasi | 0.872014 | 0.00667 |
| Presisi | 0.888426 | 0.01663 |
| Recall | 0.76755 | 0.01750 |


## 5. Analisis Sentimen dengan Model BiLSTM-CNN ##
``` python
#Model : BERT + BiLSTM-CNN
def bilstm_cnn(hp):
    #Input layer
    input_token = keras.layers.Input(shape=(128,), dtype=np.int32,
                                        name="input_token")
    input_mask = keras.layers.Input(shape=(128,), dtype=np.int32,
                                   name="input_mask")
    #Embedding
    bert_embedding = bert_model([input_token, input_mask])[0]
    
    #BiLSTM layer
    bilstm2 = keras.layers.Bidirectional(LSTM(units = hp.Int('units',
                                                     min_value = 100,
                                                     max_value = 200,
                                                     step = 50),
                                      kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_regularizer',
                                                                                         values = [0.01, 0.001])),
                                      recurrent_regularizer=keras.regularizers.l2(hp.Choice('rec_regularizer',
                                                                                            values = [0.01, 0.001])),return_sequences=True))(bert_embedding)

    #Convolution layer
    cnn = keras.layers.Conv1D(filters = hp.Int('filters',
                                                min_value = 200, 
                                                max_value = 300, 
                                                step = 50),
                              kernel_size = hp.Int('kernel_size',
                                                    min_value = 3, 
                                                    max_value = 5, 
                                                    step = 1),
                              activation='relu',
                              kernel_regularizer = keras.regularizers.l2(hp.Choice('kernel_cnn',
                                                                                    values = [0.01, 0.001])))(bilstm2)
    #Max Pooling layer
    maxpool = keras.layers.GlobalMaxPooling1D()(cnn)
    
    #Dropout layer
    cnn_out = keras.layers.Dropout(0.5)(maxpool)
 
    #Output layer
    output = keras.layers.Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(hp.Choice('kernel_dense', values = [0.01, 0.001])))(cnn_out)
    model2 = keras.models.Model(inputs=[input_token, input_mask], outputs=output)
 
    model2.compile(optimizer = keras.optimizers.Adam(1e-3),
                  loss ='categorical_crossentropy',
                  metrics=['accuracy',
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall')])
 
    return model2

#Hyperparameter tuning
tuner2 = BayesianOptimization(bilstm_cnn,
                             objective = 'val_accuracy', 
                             max_trials = 10,
                             directory = '/content/Hasil',
                             project_name = 'Sentimen-BiLSTM-CNN',
                             overwrite = True)

# Pendefinisian Callback
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner2.search(X_train, label_train,
             batch_size=32, epochs=10,
             validation_data=(X_test, label_test),
             callbacks=[early_stop, ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps2 = tuner2.get_best_hyperparameters()[0]
print('\nThe hyperparameter search is complete.'
      '\nfilters:', best_hps2.get('filters'),
      '\nkernel size:', best_hps2.get('kernel_size'),
      '\nkernel cnn:', best_hps2.get('kernel_cnn'),
      '\nunits:', best_hps2.get('units'),
      '\nkernel regularizer:', best_hps2.get('kernel_regularizer'),
      '\nrec regularizer:', best_hps2.get('rec_regularizer'),
      '\nkernel dense:', best_hps2.get('kernel_dense'))

# Mendapatkan model terbaik
model2 = tuner2.get_best_models()[0]

# Mendapatkan kinerja model
y_pred = np.argmax(model2.predict(X_test), axis=1)
y = np.argmax(label_test, axis=1)
print('accuracy: ', accuracy_score(y, y_pred), 
      '\nprecicion: ', precision_score(y, y_pred), 
      '\nrecall: ', recall_score(y, y_pred))
```

Setelah dilakukan lima kali *running*, didapatkan hasil sebagai berikut.
| Metrics | Average | Standar Deviasi |
|:---|:---:|:---:|
| Akurasi | 0.869032 | 0.00758 |
| Presisi | 0.890962 | 0.02944 |
| Recall | 0.75693 | 0.01918 |


## Kesimpulan ##
Berdasarkan evaluasi model BERT + CNN-BiLSTM dan BERT + BiLSTM-CNN, menghasilkan:
* Nilai **akurasi terbaik** 0.872014 oleh model CNN-BiLSTM.
* Nilai **presisi terbaik** 0.890962 oleh model BiLSTM-CNN.
* Nilai **recall terbaik** 0.76755 oleh model CNN-BiLSTM.


## Referensi ##
* N. Aliyah Salsabila, Y. Ardhito Winatmoko, A. Akbar Septiandri and A. Jamal.(2018). "Colloquial Indonesian Lexicon". *2018 International Conference on Asian Language Processing (IALP)*, pp. 226-229. https://doi.org/10.1109/IALP.2018.8629151.
* Gowandi, T., Murfi, H., & Nurrohmah, S. (2021). Performance analysis of hybrid architectures of deep learning for indonesian sentiment analysis. *Soft Computing in Data Sciences*, 1489, 18–27. https://doi.org/10.1007/978-981-16-7334-4_2.
* Wilie, Bryan et. al. (2020). IndoNLU: *Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. https://huggingface.co/indobenchmark/indobert-base-p2.
