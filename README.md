
# NLP Project

## Summarization using Text Rank
### Ali Mortazavi
In this project, we want to extract important sentences that can summarize the whole text.<br>
We used __Page Rank Algorithm__ for determining the importance of each sentence. In this algorithm, we consider every sentence in the text as a node and then we have to determine the relationship between nodes. To find the relation between each sentence (nodes in the page rank graph), we used word2vec.<br>
First, we trained a word2vec from our data. For determining a sentence vector, we used the average of word2vec of its words.
Then for every document, we ran page rank algorithm then we selected n top sentence as an extractive summary.<br>
(n = ratio *  number_of_document_sentences)<br>
At the end, we used __ROUGE-1__ and __ROUGE-2__ for evaluation. 


Importing Libraries


```python
import tensorflow as tf
import pandas
import numpy as np
import numpy
import sys
import re
import os
import math
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import KeyedVectors
import os.path
from rouge.rouge import rouge_score
import xml.etree.ElementTree as ET
import pandas as pd
```

    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\gensim\utils.py:1209: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

## Creating Word2Vec from Documents

 We collect every sentence from Single Dataset for training the word2vec.


```python
path = './Single -Dataset/Single -Dataset/Source/DUC'
all_files = os.listdir(path)   # imagine you're one directory above test dir

all_sentences = []
for i in range(0, len(all_files)-1):
    file_name = all_files[i]
#     print (path+'\\'+str(file_name))
    
    with open(path+'/'+str(file_name), 'r', encoding='utf-8-sig') as f:
#         print(file_name)
        context = f.read()
        #preprocessing:


        #deleting some character from text
        context = context.replace(":", " ")
        context = context.replace("(", " ")
        context = context.replace(")", " ")
        context = context.replace("،", " ")
        #replace all end sentence delimeter with '.'
        context = context.replace("...", " . ")
        context = context.replace(".", " . ")
        context = context.replace("!", " . ")
        context = context.replace("?", " . ")


        all_tokens = context.split()
        tmp = []
        for token in all_tokens:
            if (token=='.'):
                all_sentences.append(tmp)
                tmp=[]
            else:
                tmp.append(token)


```

Since our dataset is small we select (windows size = 2) and vector (dimension = 8) to avoid overfitting. 


```python
path = get_tmpfile("word2vec.model")
model = Word2Vec(all_sentences, size=8, min_count=1, workers=4, sg=0, hs=0, window=2, iter=100)
```


```python
model.save(".\word2vec1.model")
```


```python
model.wv.save('.\word_vector1.kv')
wv = KeyedVectors.load('.\word_vector1.kv', mmap='r')
```

Now we can see some word vectors.


```python
vector = wv['سلام'] 
vector
```




    array([-0.5535564 ,  0.29107115,  0.8052115 , -0.3384324 , -0.552541  ,
           -0.0543752 , -0.65191317,  0.00579695], dtype=float32)




```python
vector = model.wv["کاهش"]
vector
```




    array([-0.6940553 ,  3.9127538 ,  1.375228  , -2.802112  ,  1.833232  ,
           -1.3731824 , -1.4810753 , -0.26960114], dtype=float32)




```python
model.wv.most_similar("کاهش")
```

    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    [('پایدار', 0.9342570304870605),
     ('مختص', 0.9304113388061523),
     ('تورم', 0.9264208078384399),
     ('فعالان', 0.9204784631729126),
     ('50درصد', 0.9191967248916626),
     ('صرفاً', 0.9188379645347595),
     ('بانک', 0.9144338965415955),
     ('کارکنان', 0.9091541767120361),
     ('مرکزی', 0.9073986411094666),
     ('شدید', 0.9055722951889038)]




```python
tmp = model.wv["کاهش"]- model.wv["افزایش"]+model.wv["زیاد"]
model.similar_by_vector(tmp)
```

    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `similar_by_vector` (Method will be removed in 4.0.0, use self.wv.similar_by_vector() instead).
      
    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    [('لزوم', 0.9157054424285889),
     ('تمسک', 0.9058618545532227),
     ('پيش\u200cبيني\u200cها', 0.8861194849014282),
     ('کرسی', 0.8810957670211792),
     ('اصول', 0.8807306289672852),
     ('اثر', 0.8718945980072021),
     ('انسان', 0.8679296374320984),
     ('عَرضه', 0.8654670715332031),
     ('عرضه', 0.8650482892990112),
     ('نکردنِ', 0.8632205724716187)]




```python
tmp = model.wv["کاهش"]
model.similar_by_vector(tmp)
```

    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\ipykernel_launcher.py:2: DeprecationWarning: Call to deprecated `similar_by_vector` (Method will be removed in 4.0.0, use self.wv.similar_by_vector() instead).
      
    c:\users\ali morty\appdata\local\programs\python\python35\lib\site-packages\gensim\matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int32 == np.dtype(int).type`.
      if np.issubdtype(vec.dtype, np.int):
    




    [('کاهش', 1.0),
     ('پایدار', 0.9342569708824158),
     ('مختص', 0.9304113388061523),
     ('تورم', 0.9264207482337952),
     ('فعالان', 0.9204784035682678),
     ('50درصد', 0.919196605682373),
     ('صرفاً', 0.9188379049301147),
     ('بانک', 0.9144337773323059),
     ('کارکنان', 0.9091541171073914),
     ('مرکزی', 0.9073985815048218)]



## Raeding Word2Vec from file
We also use pretrained word2vec. 


```python
twitter_fa_w2v = dict()
with open('./twitt_wiki_ham_blog.fa.text.100.vec', 'r', encoding='utf-8') as infile:
    first_line = True
    for line in infile:
        if first_line:
            first_line = False
            continue
        tokens = line.split()
        twitter_fa_w2v[tokens[0]] = np.asarray([float(el) for el in tokens[1:]])
        if len(twitter_fa_w2v[tokens[0]]) != 100:  # 100:
            print('Bad line!')
            
```


```python
a = np.array([twitter_fa_w2v["مرد"] - twitter_fa_w2v["زن"]])
b = np.array([twitter_fa_w2v["پسر"] - twitter_fa_w2v["دختر"]])
c  = np.array([(twitter_fa_w2v["مرد"] - twitter_fa_w2v["زن"] - (twitter_fa_w2v["پسر"] - twitter_fa_w2v["دختر"]))])
d = np.concatenate((a.T,b.T,c.T), axis=1)
```

In word2vec, every dimension is correspond to one feature of the word. <br>
In the example below, we see some dimensions are close to zero as we expected. 


```python
d = pd.DataFrame(d, columns=["مرد - زن", "پسر - دختر" , "تفاصل" ])
```


```python
d.head(16)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>مرد - زن</th>
      <th>پسر - دختر</th>
      <th>تفاصل</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.272904</td>
      <td>0.203833</td>
      <td>-0.476737</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.530206</td>
      <td>2.021168</td>
      <td>-1.490962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.028775</td>
      <td>0.967605</td>
      <td>1.061170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491309</td>
      <td>-0.545429</td>
      <td>1.036738</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.590408</td>
      <td>1.684423</td>
      <td>-0.094015</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.475196</td>
      <td>0.584342</td>
      <td>-0.109146</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.165987</td>
      <td>1.721274</td>
      <td>-0.555287</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.205672</td>
      <td>-0.792910</td>
      <td>2.998582</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.701990</td>
      <td>-0.348505</td>
      <td>1.050495</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.362143</td>
      <td>0.622676</td>
      <td>-0.260533</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.391263</td>
      <td>-0.544476</td>
      <td>2.935739</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.259936</td>
      <td>-0.986105</td>
      <td>1.246041</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-1.812315</td>
      <td>-2.305010</td>
      <td>0.492695</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.221498</td>
      <td>0.850249</td>
      <td>-0.628751</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.008472</td>
      <td>-0.172758</td>
      <td>0.164286</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.114126</td>
      <td>-0.293846</td>
      <td>0.179720</td>
    </tr>
  </tbody>
</table>
</div>



## Page Rank Algorithm
We use page rank algorithm to determine the importance of each sentence. 



```python
def __extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes


def __makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()

    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix)  # insert missing columns
    matrix = insertMissingColumns(matrix.T).T  # insert missing rows

    return matrix.fillna(default)


def __ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(numpy.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T


def __normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclideanNorm(series):
    return math.sqrt(series.dot(series))


# PageRank specific functionality:

def __startState(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    startProb = 1.0 / float(len(nodes))
    return pandas.Series({node: startProb for node in nodes})


def __integrateRandomSurfer(nodes, transitionProbs, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transitionProbs.copy().multiply(1.0 - rsp) + alpha


def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    # Clerical work:
    transitionWeights = pandas.DataFrame(transitionWeights)
    nodes = __extractNodes(transitionWeights)
    transitionWeights = __makeSquare(transitionWeights, nodes, default=0.0)
    transitionWeights = __ensureRowsPositive(transitionWeights)

    # Setup:
    state = __startState(nodes)
    transitionProbs = __normalizeRows(transitionWeights)
    transitionProbs = __integrateRandomSurfer(nodes, transitionProbs, rsp)


    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbs)
        delta = state - oldState
        if __euclideanNorm(delta) < epsilon:
            break

    return state
```


```python
def get_sentence_list(context):
    ret = []
    all_tokens = context.split()
    tmp = []
    for token in all_tokens:
        if (token=='.'):
            if (len(tmp)!=0):
                ret.append(tmp)
            tmp=[]
        else:
            tmp.append(token)    
    return ret

```

## Preprocessing
We use only three characters as a boundary for the sentences. (".", "?", "!") <br>
We also remove all other delimiter characters from our data.


```python
def preprocessing(string):
    #deleting some character from text
    context = str(string)
    context = context.replace(":", " ")
    context = context.replace("»", " ")
    context = context.replace("«", " ")
    context = context.replace("(", " ")
    context = context.replace(")", " ")
    context = context.replace("/", " ")
    context = context.replace("،", " ")

    #replace all end sentence delimeter with '.'
    context = context.replace("...", " . ")
    context = context.replace(".", " . ")
    context = context.replace("!", " . ")
    context = context.replace("?", " . ")
    context = context.replace("؟", " . ")
    
    return context
```


```python
def make_graph(sentence_list, similarity_function):
    n = len(sentence_list)
    arr = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if (i!=j):
                a = sentence_list[i]
                b = sentence_list[j]
                if (isinstance(a, float) or isinstance(b, float)):
                    c=0
                else:
                    try:
                        c = similarity_function(a,b)
                    except Exception:
                        print ("OH NO!")
                        print (a)
                        print (b)
                        print ("IIII")
                        print (distance_similariy(a,b))
#                         print (c)
                        print ("/////")
                        sys.exit(0)
                arr[i][j]=c
                arr[j][i]=c
    return arr
```


```python
def sentence2vector(sentence_list, word2vec_vectors, mode="avg"):
    if mode=="avg":
        arr = []
        for sentence in sentence_list:
            sum = 0
            for i in range(0, len(sentence)):
                tmp=0
                try:
                    tmp = word2vec_vectors[sentence[i]]
                    sum+= tmp
                except KeyError:
#                     print ("key error happend")
                    a=2
            sum/=len(sentence)
            if (isinstance(sum, int)):
                arr.append(tmp = word2vec_vectors["کاهش"])
            else:
                arr.append(sum)
        return arr
    return None
```


```python
def make_input_for_page_rank(arr):
    ret = dict()
    n = len (arr)
    for i in range(0, n):
        tmp = dict()
        for j in range(0, n):
            tmp[j]=arr[j][i]
        ret[i]=tmp
    return ret
        
```


```python
def get_len(a):
    return np.sqrt(np.dot(a,a))
```


```python
def consine_similarity(a,b):
    return np.dot(a,b)/(get_len(a)*get_len(b))
```


```python
def list2sentence(list):
    ret = []
    for i in range(0, len(list)):
        str = ""
        for j in range(0, len(list[i])):
            str += list[i][j]+" "
        str = str[:-1]+". "
        ret.append(str)
#     print (ret)
    return ret
            
```


```python
def summerize_text(text, word_vector, ratio=0.2):
    context = preprocessing(text)  
    sentence_list = get_sentence_list(context)
    wv = word_vector
    sentence_vectors = sentence2vector(sentence_list, wv)
    arr = make_graph(sentence_vectors, consine_similarity)
    transitionWeights = make_input_for_page_rank(arr)
    rank_list = powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000)
    zip = []
    for i in range(0, len(rank_list)):
        zip.append([i, rank_list[i]])
    sorted_zip = sorted(zip, key=lambda tup: tup[1], reverse=True)
#     print (sorted_zip)
    tmp_dict = dict()
    for i in range(0, len(sorted_zip)):
        tmp_dict[sorted_zip[i][0]]=i
    resource = list2sentence(sentence_list)
    summary = ""
    summary_sentences=int(len(resource)*ratio)
    for i in range(0, len(resource)):
        if (tmp_dict[i]<summary_sentences):
            summary+=resource[i]
    return summary
    
```


```python
def write_summary (summary, file_name, location='.\our_output\Single\our_summary'):
    save_path = location+"/"+file_name
    
#     print (file_name)
#     print (location)
#     print (save_path)
    if not os.path.exists(location):
        os.makedirs(location)
    file1 = open(save_path, "w", encoding="utf-8-sig")
    file1.write(summary)
    file1.close()

```

## ROUGE Metrics
We used __ROUGE__ metrics to evaluate our results. <br>
__ROUGE-n__ compares n-grams in reference summary and system summary. We reported __precision, recall, f-score__ for ROUGE-1 and ROUGE-2. 



```python
def calculate_ROUGE_metrics (reference, system):
    
    reference= preprocessing(reference)
    system = preprocessing(system)
    reference_tokens = reference.split()
    system_tokens = system.split()
#     print (reference_tokens)
#     print (system_tokens)
#     sys.exit(0)
    #unigram:
        
    ref_set_1 = set()
    for t in reference_tokens:
        ref_set_1.add(t)
    sys_set_1 = set()
    for t in system_tokens:
        sys_set_1.add(t)
    over_lap = 0
    for t in ref_set_1:
        if (t in sys_set_1): over_lap+=1
    uni_precision = over_lap/len(sys_set_1)
    uni_recall = over_lap/len(ref_set_1)
    uni_f1 = 2 * (uni_precision*uni_recall)/(uni_precision+uni_recall)
    #bigram:
    ref_set_2 = set()
    for i in range(0, len(reference_tokens)-1):
        ref_set_2.add((reference_tokens[i], reference_tokens[i+1]))
    sys_set_2 = set()
    for i in range(0, len(system_tokens)-1):
        sys_set_2.add((system_tokens[i], system_tokens[i+1]))
    over_lap = 0
    for (x,y) in ref_set_2:
        if ((x,y) in sys_set_2): 
            over_lap+=1
#     try: 
    bi_precision = over_lap/len(sys_set_2)
    bi_recall = over_lap/len(ref_set_2)
    bi_f1 = 2 * (bi_precision*bi_recall)/(bi_precision+bi_recall)
#     except Exception:
#         print ("SUMMARIES")
#         print ("SYSTEM")
#         print (system_tokens)
#         print ("///\n REFERENCE")
#         print (reference_tokens)
#         sys.exit(0)
    return [uni_f1, uni_precision, uni_recall, bi_f1, bi_precision, bi_recall]
```

In the example below, we see precision, recall, F-score for unigram and bigram.


```python
ref = "A A B"
system = "A B A"
calculate_ROUGE_metrics (ref, system)
```




    [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]




```python
def get_scores (reference_summary, our_summary):
    return calculate_ROUGE_metrics(reference_summary, our_summary)
    
```

## Single Document Dataset
In this section, we want to summerize single document. 


```python
def single_document_summarize(wv, path = './Single -Dataset/Single -Dataset/Source/DUC', output_location ='./our_output/Single/our_summary'):
    all_files = os.listdir(path)   # imagine you're one directory above test dir
    all_sentences = []
    for i in range(0, len(all_files)):
        file_name = all_files[i]
    #     print (path+'\\'+str(file_name))
        with open(path+'/'+str(file_name), 'r', encoding='utf-8-sig') as f:
    #         print(file_name)
            context = f.read()
            summary = summerize_text(context, wv, ratio=0.4)
            write_summary(summary, file_name, location= output_location)
```

## Evaluation For Single Documents


```python

def evaluation (our_summary_path = ".\our_output\Single\our_summary", refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = "./Evaluation/", number_of_print = 4):
    
    our_summaries = os.listdir(our_summary_path)
    reference_summaries = os.listdir(refereence_path)
    evaluation_array = []
    for i in range(0, len(our_summaries)-1):

        file_name = our_summaries[i]
        with open(our_summary_path+'/'+str(file_name), 'r', encoding='utf-8-sig') as f:
            our_values = [] 
            our_summary = f.read()
            for j in range(0, len(reference_summaries)):
                if (reference_summaries[j][:prefix]==file_name[:prefix]):
                    #they are for the same text
                    # print ("EQUALLLL")
                    # print (reference_summaries[i])
                    # print (file_name)
                    with open(refereence_path+'/'+str(reference_summaries[j]), 'r', encoding='utf-8-sig') as g:
                        reference_summary = g.read()
                        # print ("OUR SUMMARY")
                        # print (our_summary)
                        # print ("Their Sumaary")
                        # print (reference_summary)
                        scores = rouge_score.rouge_n(our_summary, reference_summary, n=1)
                        tmp= []
                        for key, value in scores.items():
                            tmp.append(value)
                        scores = rouge_score.rouge_n(our_summary, reference_summary, n=2)
                        for key, value in scores.items():
                            tmp.append(value)
#                         tmp = calculate_ROUGE_metrics(reference_summary, our_summary)
                        try:
                            tmp = get_scores(reference_summary, our_summary)
                            if (number_of_print>0):
                                number_of_print-=1
                                print ("OUR SUMMARY")
                                print (our_summary)
                                print ("REF SUMMARY")
                                print (reference_summary)
                        except Exception:
                            
                            print ("ZERO BIGRAM PROBLEM")
                            print(reference_summaries[j])
                            print (file_name)
                            print (reference_summaries[j][:prefix]==file_name[:prefix])
                            print ("OUR FILE")
                            print (refereence_path+'/'+str(reference_summaries[j]))
                            print (reference_summary)
                            print ("OUR SUMMARY")
                            print (our_summary)
#                             sys.exit(0)
                        # print (tmp)
                        our_values.append(tmp)
            # print("first")
            # print (our_values)
            our_values=np.asarray(our_values)
            our_values=np.average(our_values, axis=0)
            # print ("avg")
            # print (our_values)
            evaluation_array.append(our_values)
            
    
    df = pd.DataFrame(evaluation_array, columns = ["rouge-1 f", "rouge-1 p", "rouge-1 r", "rouge-2 f", "rouge-2 p", "rouge-2 r"]) 
    path = evaluation_path
    df.to_csv(path+'result.csv')
    return df 
```

## Using Our trained word2vec


```python
single_document_summarize(twitter_fa_w2v, output_location ='./our_output/Single/our_word_2_vec_summaries')
```


```python
result1 = evaluation(our_summary_path = '.\our_output\Single\our_word_2_vec_summaries', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\our_word_2_vec_for_single_doc", number_of_print = 4)
```

    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    جدایی نادر از... بالاخره خوب است یا نه
    واکنش ها وتحلیل های متفاوتی پیرامون فیلم جدایی نادر از سیمین و اصغر فرهادی منتشر می شود. 
    گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. 
     اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است 
     و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد. 
    در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند 
    علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد.
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند.
    به همان دلایل ذکر شده روی وی حساب باز کرده اند. 
    حقیقت آنجاست که جدای از این حواشی، اگر جدایی نادر از سیمین در کشور دیگری جز ایران ساخته و اکران می شد هیچ گاه تا این اندازه مورد توجه قرار نمی گرفت. 
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند، شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند 
    حقیقت آنجاست که جدای از این حواشی، اگر جدایی نادر از سیمین در کشور دیگری جز ایران ساخته و اکران می شد هیچ گاه تا این اندازه مورد توجه قرار نمی گرفت
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    واکنش ها وتحلیل های متفاوتی پیرامون فیلم جدایی نادر از سیمین و اصغر فرهادی منتشر می شود. واکنش هایی که بعضا صد در صد در تضاد با یکدیگر هستند. و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها شده.
    
    گروهی معتقدند اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد.
    
     علت دیگر آن نیز مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند، اما این گروه معتقدند وی در آینده پتانسیل اقدامات ساختارشکنانه را خواهد داشت.
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف می کنند و وی را ناجی سینمای ایران می خوانند،این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی ، اجتماعی و حتی سیاسی خود تعریف کرده اند.
    
    در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد، دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود...
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    گروهی معتقدند اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد.و برای اثبات این نظر، به دست دادن وی با زنان نامحرم و نمایش این عمل قبیح دینی در صحناتی از فیلم جدایی نادر از سیمین اشاره می کنند. 
    در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد(اعم از نفی و یا تمجید)، دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود...
    
    
    ZERO BIGRAM PROBLEM
    IRN.CU.13910203.045.E.S.g.f.c.txt
    IRN.CU.13910203.045.txt
    True
    OUR FILE
    .\Single -Dataset\Single -Dataset\Summ\Extractive/IRN.CU.13910203.045.E.S.g.f.c.txt
    كنسرت اركستر آكادمیك تهران در برج میلاد  در راستای پاسداشت روزهای ملی به صحنه می رود 
    خوانندگان نسبت به توانایی اجرایی اركسترال انتخاب شدنددر چهار بخش كنسرت  را اجرا خواهد كرد وسالار عقیلی ، نیما مسیحا ، مانی رهنما و حمید خزاعی در فرم های  سنتی ، پاپ و كلاسیك  ایفای نقش می كنندقیمت بلیت های كنسرت 30 الی 90 هزار تومان در نظر گرفته شده است.اركستر آكادمیك تهران 13 و 14 اردیبهشت به روی صحنه می رود
    
    OUR SUMMARY
    'امین سالمی' رهبر اركستر نیز در این نشست از برگزاری این كنسرت به عنوان تجربه ای تازه یاد كرد و گفت وجود 4 خواننده در این اجرا فضای مطلوب شنیداری با سلیقه های گوناگاون را برای مخاطب ایجاد می كند. 'سالار عقیلی' خواننده بخش موسیقی سنتی با بیان اینكه این كنسرت تجربه های تازه ای را به همراه دارد افزود شركت در این كنسرت به عنوان خواننده بخشی دیگر از ارائه تجربه های متفاوت كاری من به مخاطبان محسوب می شود. عقیلی افزود آشنایی مخاطبان جوان موسیقی پاپ با داشته های موسیقی ملی و جذب حتی تعداد محدودی از شنوندگان به موسیقی جدی را می توان از عمده ترین دلایل همكاری من در این فرم از موسیقی برشمرد. 
    

## Using Twitter_FA word2vec


```python
single_document_summarize(twitter_fa_w2v, output_location ='./our_output/Single/twitter_word_2_vec_summaries')
```


```python
result2 = evaluation(our_summary_path = '.\our_output\Single\\twitter_word_2_vec_summaries', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\\twitter_word_2_vec_for_single_doc", number_of_print = 4)
```

    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    جدایی نادر از... بالاخره خوب است یا نه
    واکنش ها وتحلیل های متفاوتی پیرامون فیلم جدایی نادر از سیمین و اصغر فرهادی منتشر می شود. 
    گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. 
     اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است 
     و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد. 
    در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند 
    علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد.
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند.
    به همان دلایل ذکر شده روی وی حساب باز کرده اند. 
    حقیقت آنجاست که جدای از این حواشی، اگر جدایی نادر از سیمین در کشور دیگری جز ایران ساخته و اکران می شد هیچ گاه تا این اندازه مورد توجه قرار نمی گرفت. 
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند، شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند 
    حقیقت آنجاست که جدای از این حواشی، اگر جدایی نادر از سیمین در کشور دیگری جز ایران ساخته و اکران می شد هیچ گاه تا این اندازه مورد توجه قرار نمی گرفت
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    واکنش ها وتحلیل های متفاوتی پیرامون فیلم جدایی نادر از سیمین و اصغر فرهادی منتشر می شود. واکنش هایی که بعضا صد در صد در تضاد با یکدیگر هستند. و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها شده.
    
    گروهی معتقدند اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد.
    
     علت دیگر آن نیز مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند، اما این گروه معتقدند وی در آینده پتانسیل اقدامات ساختارشکنانه را خواهد داشت.
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف می کنند و وی را ناجی سینمای ایران می خوانند،این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی ، اجتماعی و حتی سیاسی خود تعریف کرده اند.
    
    در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد، دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود...
    
    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد اعم از نفی و یا تمجید دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود. 
    REF SUMMARY
    گروهی معتقدند اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد.و برای اثبات این نظر، به دست دادن وی با زنان نامحرم و نمایش این عمل قبیح دینی در صحناتی از فیلم جدایی نادر از سیمین اشاره می کنند. 
    در هر حال چیزی که این روزها در مورد تحلیل های پیرامون جدایی نادر از سیمین واقعیت دارد(اعم از نفی و یا تمجید)، دلهره و یا امید نسبت به عملکرد و مواضع آینده این کارگردان است که هیچ گاه علنا بیان نمی شود...
    
    
    ZERO BIGRAM PROBLEM
    IRN.CU.13910203.045.E.S.g.f.c.txt
    IRN.CU.13910203.045.txt
    True
    OUR FILE
    .\Single -Dataset\Single -Dataset\Summ\Extractive/IRN.CU.13910203.045.E.S.g.f.c.txt
    كنسرت اركستر آكادمیك تهران در برج میلاد  در راستای پاسداشت روزهای ملی به صحنه می رود 
    خوانندگان نسبت به توانایی اجرایی اركسترال انتخاب شدنددر چهار بخش كنسرت  را اجرا خواهد كرد وسالار عقیلی ، نیما مسیحا ، مانی رهنما و حمید خزاعی در فرم های  سنتی ، پاپ و كلاسیك  ایفای نقش می كنندقیمت بلیت های كنسرت 30 الی 90 هزار تومان در نظر گرفته شده است.اركستر آكادمیك تهران 13 و 14 اردیبهشت به روی صحنه می رود
    
    OUR SUMMARY
    'امین سالمی' رهبر اركستر نیز در این نشست از برگزاری این كنسرت به عنوان تجربه ای تازه یاد كرد و گفت وجود 4 خواننده در این اجرا فضای مطلوب شنیداری با سلیقه های گوناگاون را برای مخاطب ایجاد می كند. 'سالار عقیلی' خواننده بخش موسیقی سنتی با بیان اینكه این كنسرت تجربه های تازه ای را به همراه دارد افزود شركت در این كنسرت به عنوان خواننده بخشی دیگر از ارائه تجربه های متفاوت كاری من به مخاطبان محسوب می شود. عقیلی افزود آشنایی مخاطبان جوان موسیقی پاپ با داشته های موسیقی ملی و جذب حتی تعداد محدودی از شنوندگان به موسیقی جدی را می توان از عمده ترین دلایل همكاری من در این فرم از موسیقی برشمرد. 
    

## Comparasion - Single Document
### Our word2vec Results


```python
result1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rouge-1 f</th>
      <th>rouge-1 p</th>
      <th>rouge-1 r</th>
      <th>rouge-2 f</th>
      <th>rouge-2 p</th>
      <th>rouge-2 r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.618092</td>
      <td>0.534884</td>
      <td>0.765144</td>
      <td>0.434872</td>
      <td>0.363880</td>
      <td>0.580697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.600769</td>
      <td>0.566667</td>
      <td>0.657849</td>
      <td>0.484734</td>
      <td>0.439560</td>
      <td>0.566518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.588950</td>
      <td>0.613333</td>
      <td>0.582231</td>
      <td>0.471829</td>
      <td>0.488889</td>
      <td>0.473384</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.563248</td>
      <td>0.498592</td>
      <td>0.662783</td>
      <td>0.412858</td>
      <td>0.355652</td>
      <td>0.505841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.643913</td>
      <td>0.665517</td>
      <td>0.628607</td>
      <td>0.511866</td>
      <td>0.519048</td>
      <td>0.510824</td>
    </tr>
  </tbody>
</table>
</div>




```python
result1.mean()
```




    rouge-1 f    0.508832
    rouge-1 p    0.482383
    rouge-1 r    0.634011
    rouge-2 f    0.367171
    rouge-2 p    0.353983
    rouge-2 r    0.481875
    dtype: float64



The result for 5 different documents: (r = recall, p = precission, f = f1_score)

### Twitter word2vec Results


```python
result2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rouge-1 f</th>
      <th>rouge-1 p</th>
      <th>rouge-1 r</th>
      <th>rouge-2 f</th>
      <th>rouge-2 p</th>
      <th>rouge-2 r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.618092</td>
      <td>0.534884</td>
      <td>0.765144</td>
      <td>0.434872</td>
      <td>0.363880</td>
      <td>0.580697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.600769</td>
      <td>0.566667</td>
      <td>0.657849</td>
      <td>0.484734</td>
      <td>0.439560</td>
      <td>0.566518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.588950</td>
      <td>0.613333</td>
      <td>0.582231</td>
      <td>0.471829</td>
      <td>0.488889</td>
      <td>0.473384</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.563248</td>
      <td>0.498592</td>
      <td>0.662783</td>
      <td>0.412858</td>
      <td>0.355652</td>
      <td>0.505841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.643913</td>
      <td>0.665517</td>
      <td>0.628607</td>
      <td>0.511866</td>
      <td>0.519048</td>
      <td>0.510824</td>
    </tr>
  </tbody>
</table>
</div>



The average for whole dataset is:


```python
result2.mean()
```




    rouge-1 f    0.506452
    rouge-1 p    0.480294
    rouge-1 r    0.631836
    rouge-2 f    0.364011
    rouge-2 p    0.351220
    rouge-2 r    0.478495
    dtype: float64



## Multi Document
In this section, we summerize multiple document. <br>
In this case, we concatenate all sentences in all documents and then we run __page rank algorithm__ to prioritize every sentence. Then we choose top __k__ sentece as our summary. <br>
### Notice:
#### k  = fixed ratio * number of sentences in the input.



```python
def get_summary (summary_path, the_name):
    all_files = os.listdir(summary_path)
    for file in all_files:
        if file[:6] == the_name:
#             print(file)
            file_path = summary_path + "\\" + file
            with open(file_path, 'r', encoding="utf-8") as content_file:
                extractive_summary = content_file.read()
                return extractive_summary
    return None
```


```python
def read_XML (path):
#     print ("start reading "+ path)
    tree = ET.parse(path)
    root = tree.getroot()
    text = root.find("TEXT").text
    # print(text)
    return text
```


```python
def evaluation_multi (wv, path0 = "Multi - Dataset\Multi - Dataset", compare_with_extractive=True, number_of_print=4):
    all_tracks = os.listdir(path0)
    all_scores = []
    for track in all_tracks:
        track_score = []
        path1 = path0 + "\\" + track
        path2 = path0 + "\\" + track + "\\Source"
        all_directories = os.listdir(path2)
        for dir in all_directories:
            path3 = path2 + "\\" + dir
            if (os.path.isdir(path3)):
                all_docs= os.listdir(path3)
    #             print (all_docs)
                all_doc_context = ""
                for doc in all_docs:
                    source_file_path = path3 + "\\" + doc
                    # print(source_file_path)
                    text = read_XML(source_file_path)
                    all_doc_context+= text
    #             print (all_doc_context)
                summary = summerize_text(all_doc_context, wv, ratio=0.1)
#                 print (summary)
                
                summary_path1 = path1 + "\\" + "Summ"
                all_people = os.listdir(summary_path1)
    #             print (all_people)
                abstractive_summaries = []
                extractive_summaries = []
                for person in all_people:
    #                 print (person)
                    summary_path2 = summary_path1 + "\\" + person
                    summary_path_extractive = summary_path2 + "\\Multi\\Extractive"
                    summary_path_abstractive = summary_path2 + "\\Multi\\Abstractive"
                    extr = get_summary(summary_path_extractive,dir)
                    abstr = get_summary(summary_path_abstractive,dir)
                    abstractive_summaries.append(abstr)
                    extractive_summaries.append(extr)
                # print ("|||||")
                # print (extractive_summaries[0])
                if (compare_with_extractive):
                    summary_set = extractive_summaries
                else:
                    summary_set = abstractive_summaries
                scores = []
                for i in range(0, len(summary_set)):
                    scores.append(get_scores(summary_set[i], summary))
                    if (number_of_print>0):
                        number_of_print-=1
                        print ("-----------------------------------")
                        print ("REF SUMMARY")
                        print (summary_set[i])
                        
                        print ("OUR SUMMARY")
                        print (summary)
                        print ("-----------------------------------")
                        print ("\n\n")
                        
                scores = np.asarray(scores)
                scores = np.average(scores, axis=0)
                
                all_scores.append(scores)
    return pd.DataFrame(all_scores, columns = ["rouge-1 f", "rouge-1 p", "rouge-1 r", "rouge-2 f", "rouge-1 p", "rouge-1 r"])


```

## Using Our trained word2vec


```python
evaluation_df1 = evaluation_multi(wv)
```

    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترین ورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.
    پس از درخشش پولاد مردان وزنه‌برداری ایران در رقابت‌های وزنه‌برداری قهرمانی جهان در رقابت‌های پاریس 2011 و کسب بهترین نتیجه تاریخ وزنه‌برداری برای کشورمان در این رقابت‌ها، نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری به منظور انتخاب بهترین‌ وزنه‌بردار جهان قرار گرفته است در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    
    پیش از ظهر امروز پهلوان بلندآوازه جهان، قهرمان فوق سنگین وزن وزنه‌برداری جهان با حضور استاندار، مدیران کل مازندران، مسئولان شهرستان قائمشهر و استقبال بی‌نظیر و سرشار از تقدیر و شعف مردم مازندران وارد قائمشهر شد.
    
    از تمبر یادبود بهداد سلیمی (قوی‌ترین ‌مرد جهان) که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده با امضای سلیمی و فرماندار قائمشهر رونمایی شد.
    
    طبق پیش‌بینی‌ها، بهداد سلیمی، قوی‌ترین مرد جهان بر سکوی نخست دسته فوق‌سنگین بازی‌های آسیایی گوانجو ایستاد. 
    مجله جهانی فدراسیون بین‌المللی، بهداد سلیمی بهترین را به عنوان بهترین وزنه‌بردار جهان در سال 2010 انتخاب کرد.
     علی پروین در جریان تمرین امروز پرسپولیس درباره قرمانی بهداد سلیمی گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.
    
    وزنه‌برداري قهرماني جهان - تركيه:بهداد سليمي قويترين مرد جهان شد
    : دكتر محمود احمدي نژاد طي پيامي، كسب عنوان قهرماني بهداد سليمي در رقابتهاي جهاني وزنه برداري را تبريك گفت.
    
    سليمي كه از همان روزهاي اول خودش را به عنوان يك رضا‌زاده جديد مطرح كرده بود سرانجام توانست ركورد رضا‌زاده را بشكند.
    بهداد اما به اين چيزها راضي نيست: «خوشحالم كه توانستم ركورد جديدي را به نام خودم ثبت كنم و همچنين به عنوان برترين وزنه‌بردار انتخاب شوم اما كار اصلي‌ام مانده است. من مي‌خواهم طلاي المپيك را بگيرم.»
    وزنه‌برداري قهرماني آسيا – چين:سليمي قهرمان دسته فوق سنگين شد
    رهبر معظم انقلاب :متشكرم، دل ملت ايران را شاد كرديد؛ركورد دنيا با فرياد ياعلي شكست
    رئیس فدراسیون جهانی وزنه برداری برای گفتن تبریک به بهداد سلیمی و حسین رضازاده به پشت صحنه مسابقات رفت.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    
    OUR SUMMARY
    در بین ورزشکاران زن نیز رتبه نخست به خدیجه آزادپور ووشو اختصاص یافت و آرزو معتمدی قایقرانی مه لقا جان بزرگی تیراندازی و سوسن حاجی پور تکواندو در رتبه های بعد قرار گرفتند. سجاد انوشیروانی با اینکه به خاطر کسب مدال برنز بازی‌های آسیایی خدا را شکر می‌کند می‌گوید توقع بیش از این را از خودش داشته است؛ من آمده بودم به این مسابقات تا مدال نقره را بگیرم اما متأسفانه نشد. به گزارش واحد مركزي خبر سليمي در يک ضرب با 208 کيلوگرم دوم شده بود اما در دو ضرب با آسيب ديدگي وزنه‌بردار روس و کنار کشيدن از ادامه مسابقه با بالاي سربردن وزنه245 کيلوگرم دردو ضرب عنوان قهرماني خود را مسجل کرد. در روز پاياني اين مسابقات که حکم انتخابي المپيک 2012 لندن را نيز دارد در دسته فوق سنگين وزن بهداد سليمي از ايران با بلند کردن وزنه 208 کيلوگرمي پس از يوگني چيگيشف از روسيه در جاي دوم قرار گرفت. در حرکت يک ضرب چيگيشف از روسيه با 210 کيلوگرم اول شد و يوداچين از اوکراين با رکورد 205 کيلوگرم در مکان سوم قرار گرفت. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. بهداد در سالي كه گذشت قهرماني آسيا و جوانان جهان را كسب كرد. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در اين وزن هايبو سان از چين دوم شد و اوتا كازومي از ژاپن به مدال برنز رسيد. كيانوش رستمي كه سال گذشته در دسته 77 كيلوگرم وزنه زد و موفق شد مدال برنز حركت يكضرب را به خود اختصاص دهد با نظر كوروش باقري به دسته 85 كيلوگرم صعود كرد تا بدون وزن كم كردن عضلات خود را پر كرده و به سرعت در وزن جديد جا بيفتد. رستمي در ركوردگيري اخير انجام شده در سنندج به ترتيب وزنه‌هاي 171 كيلو در يكضرب و 210 كيلوگرم در دوضرب را مهار كرده تا با ركورد مجموع 381 كيلوگرم از شرايط خوبي براي ايستادن بر يكي از سكوهاي اول تا سوم جهاني در پاريس برخوردار باشد. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. انوشيرواني هم روي سكوي جهاني رفت سجاد انوشيرواني ديگر وزنه‌بردار كشور در دسته فوق‌سنگين ديشب درخشش بهداد سليمي را با كسب مدال برنز حركت يكضرب نقره دوضرب و نقره مجموع تكميل كرد. سجاد كه در آنتاليا بعد از انداختن وزنه 185 اين وزنه و وزنه 191 را در يكضرب بالاي سر برده بود اين بار هم پس از انداختن وزنه 193 در حركت دوم وزنه 193 را بالاي سر برد و بعد هم با مهار وزنه 198 كيلوگرم مدال برنز حركت يكضرب را براي خودش به ارمغان آورد. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. بهداد سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در رقابت‎هاي دسته فوق سنگين مسابقات جهاني آنتاليا سجاد انوشيرواني ديگر وزنه بردار ايران هم نمايش خيره کننده‎اي داشت و با وجود آنکه تنها 2 ماه از عمل جراحي زانويش مي گذشت با مهار وزنه‎هاي 191 کيلوگرم در يک ضرب و 235 کيلوگرم در دوضرب با مجموع 426 کيلوگرم در مکان پنجم جهان ايستاد و 22 امتياز با ارزش تيمي براي ايران در سهميه المپيک 2012 به دست آورد. سلیمی قهرمان ورزنه‌برداری کشور و قوی‌ترین مرد جهان است که در مسابقات اخیر غرب آسیا توانست رتبه ششم خود را به رتبه اول ارتقا داده و سه مدال طلا برای کشورمان به ارمغان آورد. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترینورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.سال گذشته بهداد سلیمی از کشورمان این عنوان را کسب کرد و به احتمال زیاد با توجه به اینکه وزنه‌بردار فوق سنگین کشورمان امسال با شکستن رکورد جهان به عنوان قهرمانی رقابت‌های پاریس 2011 رسید، امسال نیز بخت نخست کسب این عنوان محسوب می‌شود.
    قهرمان وزنه‌برداری جهان امروز با استقبال بی‌نظیر مردم مازندران وارد قائمشهر شد.سلیمی نام مولایش علی (ع) را بر لبان جهانیان به عنوان رمز قدرت، پهلوانی و جاودانگی جاری ساخت.تمبر یادبود بهداد سلیمی در اتحادیه تمبر جهان ثبت شد.بهداد سلیمی در رقابتهای وزنه‌برداری قهرمانی بزرگسالان جهان در آنتالیا عنوان قهرمانی را به دست آورد و در بازیهای آسیایی گوانگجو هم مدال طلای دسته فوق سنگین را به گردن آویخت. 
    علی پروین در جریان تمرین امروز پرسپولیس درباره قرمانی بهداد سلیمی گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.رهبرورییس جمهورطي پيامي، كسب عنوان قهرماني بهداد سليمي در رقابتهاي جهاني وزنه برداري را تبريك گفت.
    معاون رييس جمهوري و رييس سازمان تربيت بدني طي آييني با اهداي 125 سكه بهار آزادي از بهداد سليمي قهرمان سنگين وزن مسابقات وزنه برداري سال 2010 جهان در تركيه تجليل كرد.وزنه‌برداران فوق سنگين كشورمان قهرمان و نايب قهرمان رقابت‌‌هاي وزن 105+ كيلوگرم قهرماني آسيا شدند.بهداد سلیمی وزنه‌بردار تیم ملی کشورمان در جریان رکوردگیری تیم ملی در سنندج رکورد 217 کیلوگرم را به نام خود ثبت کرد.دکتر تاماش آیان رئیس فدراسیون جهانی وزنه برداری پس از شکستن رکورد یکضرب جهان توسط بهداد سلیمی به پشت صحنه مسابقات آمد تا به بهداد و حسین رضازاده تبریک بگوید.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.آیت الله جوادی آملی پس از دریافت این مدال‌ها خطاب به سلیمی گفت: «خداوند ان‌شاالله به شما عزت، شکوه و جلال دنیا و آخرت عطا کند و ‌در برنامه‌های ورزشی خود موفق باشید.»
    
    
    OUR SUMMARY
    در بین ورزشکاران زن نیز رتبه نخست به خدیجه آزادپور ووشو اختصاص یافت و آرزو معتمدی قایقرانی مه لقا جان بزرگی تیراندازی و سوسن حاجی پور تکواندو در رتبه های بعد قرار گرفتند. سجاد انوشیروانی با اینکه به خاطر کسب مدال برنز بازی‌های آسیایی خدا را شکر می‌کند می‌گوید توقع بیش از این را از خودش داشته است؛ من آمده بودم به این مسابقات تا مدال نقره را بگیرم اما متأسفانه نشد. به گزارش واحد مركزي خبر سليمي در يک ضرب با 208 کيلوگرم دوم شده بود اما در دو ضرب با آسيب ديدگي وزنه‌بردار روس و کنار کشيدن از ادامه مسابقه با بالاي سربردن وزنه245 کيلوگرم دردو ضرب عنوان قهرماني خود را مسجل کرد. در روز پاياني اين مسابقات که حکم انتخابي المپيک 2012 لندن را نيز دارد در دسته فوق سنگين وزن بهداد سليمي از ايران با بلند کردن وزنه 208 کيلوگرمي پس از يوگني چيگيشف از روسيه در جاي دوم قرار گرفت. در حرکت يک ضرب چيگيشف از روسيه با 210 کيلوگرم اول شد و يوداچين از اوکراين با رکورد 205 کيلوگرم در مکان سوم قرار گرفت. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. بهداد در سالي كه گذشت قهرماني آسيا و جوانان جهان را كسب كرد. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در اين وزن هايبو سان از چين دوم شد و اوتا كازومي از ژاپن به مدال برنز رسيد. كيانوش رستمي كه سال گذشته در دسته 77 كيلوگرم وزنه زد و موفق شد مدال برنز حركت يكضرب را به خود اختصاص دهد با نظر كوروش باقري به دسته 85 كيلوگرم صعود كرد تا بدون وزن كم كردن عضلات خود را پر كرده و به سرعت در وزن جديد جا بيفتد. رستمي در ركوردگيري اخير انجام شده در سنندج به ترتيب وزنه‌هاي 171 كيلو در يكضرب و 210 كيلوگرم در دوضرب را مهار كرده تا با ركورد مجموع 381 كيلوگرم از شرايط خوبي براي ايستادن بر يكي از سكوهاي اول تا سوم جهاني در پاريس برخوردار باشد. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. انوشيرواني هم روي سكوي جهاني رفت سجاد انوشيرواني ديگر وزنه‌بردار كشور در دسته فوق‌سنگين ديشب درخشش بهداد سليمي را با كسب مدال برنز حركت يكضرب نقره دوضرب و نقره مجموع تكميل كرد. سجاد كه در آنتاليا بعد از انداختن وزنه 185 اين وزنه و وزنه 191 را در يكضرب بالاي سر برده بود اين بار هم پس از انداختن وزنه 193 در حركت دوم وزنه 193 را بالاي سر برد و بعد هم با مهار وزنه 198 كيلوگرم مدال برنز حركت يكضرب را براي خودش به ارمغان آورد. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. بهداد سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در رقابت‎هاي دسته فوق سنگين مسابقات جهاني آنتاليا سجاد انوشيرواني ديگر وزنه بردار ايران هم نمايش خيره کننده‎اي داشت و با وجود آنکه تنها 2 ماه از عمل جراحي زانويش مي گذشت با مهار وزنه‎هاي 191 کيلوگرم در يک ضرب و 235 کيلوگرم در دوضرب با مجموع 426 کيلوگرم در مکان پنجم جهان ايستاد و 22 امتياز با ارزش تيمي براي ايران در سهميه المپيک 2012 به دست آورد. سلیمی قهرمان ورزنه‌برداری کشور و قوی‌ترین مرد جهان است که در مسابقات اخیر غرب آسیا توانست رتبه ششم خود را به رتبه اول ارتقا داده و سه مدال طلا برای کشورمان به ارمغان آورد. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترین ورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.
    به گزارش باشگاه خبری فارس «توانا»، پس از درخشش پولاد مردان وزنه‌برداری ایران دررقابت های پاریس2011نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    به گزارش خبرگزاری فارس از شهرستان قائمشهر، مراسم تجلیل از جهان پهلوان بهداد سلیمی به نام نماد قدرت جهان امروز با شعفی غرورآفرین در سالن همایش‌های شهرداری قائمشهر برگزار شد. 
    مرد نیرومند ایران نام مولایش علی (ع) را بر لبان جهانیان به عنوان رمز قدرت، پهلوانی و جاودانگی جاری ساخت.
    سلیمی نام قائم شهر،شهرمزین به نام حضرت قائم رادرجهان جاودانه کرد.رئیس اداره ورزش وجوانان شهرستان قائم شهرنیزبااشاره به کسب هشت مدال جهانی توسط ورزشکاران قائم شهرازابتدای سال90تاکنون تصریح کرد:کمترشهری درکشورماهمچون قائم شهرصاحب این همه افتخارومدال جهانی است. 
    همچنین طی مراسمی از تمبر یادبود بهداد سلیمی که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده است با امضای بهداد سلیمی و فرماندار قائمشهر رونمایی شد.
    طبق پیش‌بینی‌ها، بهداد سلیمی، قوی‌ترین مرد جهان بر سکوی نخست دسته فوق‌سنگین بازی‌های آسیایی گوانجو ایستاد. 
    بهداد سلیمی،‌ قوی‌ترین مرد جهان برای دریافت جایزه بهترین وزنه‌بردار سال 2010 جهان عازم ترکیه شد.
    علی پروین گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.
    دکتراحمدی نژادطی پیامی این موفقیت بزرگ رابه آقای سلیمی ،خانواده محترم، مربيان گرامي و ملت شريف ايران تبريك عرض کردند.
    رهبر معظم انقلاب در پيامي از تيم ملي وزنه‌برداري براي شاد كردن دل ملت عزيز ايران تشكر كردند. 
    متن پيام حضرت آيت الله خامنه‌اي به شرح ذيل است.
    بسم الله الرحمن الرحيم 
    كاروان اعزامي به مسابقات وزنه‌برداري قهرماني جهان 
    سلام عليكم
    از شما جوانان و مدال آوران غيور به ويژه آقايان بهداد سليمي و كيانوش رستمي، كه با موفقيت خود در اين دوره از مسابقات قهرماني جهان، دل ملت عزيز ايران را شاد كرده‌ايد، تشكر مي‌كنم.
    سيد علي خامنه اي / 22 آبان
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    آیت الله جوادی آملی پس از دریافت این مدال‌ها خطاب به سلیمی گفت: «خداوند ان‌شاالله به شما عزت، شکوه و جلال دنیا و آخرت عطا کند و ‌در برنامه‌های ورزشی خود موفق باشید.»
    
    
    OUR SUMMARY
    در بین ورزشکاران زن نیز رتبه نخست به خدیجه آزادپور ووشو اختصاص یافت و آرزو معتمدی قایقرانی مه لقا جان بزرگی تیراندازی و سوسن حاجی پور تکواندو در رتبه های بعد قرار گرفتند. سجاد انوشیروانی با اینکه به خاطر کسب مدال برنز بازی‌های آسیایی خدا را شکر می‌کند می‌گوید توقع بیش از این را از خودش داشته است؛ من آمده بودم به این مسابقات تا مدال نقره را بگیرم اما متأسفانه نشد. به گزارش واحد مركزي خبر سليمي در يک ضرب با 208 کيلوگرم دوم شده بود اما در دو ضرب با آسيب ديدگي وزنه‌بردار روس و کنار کشيدن از ادامه مسابقه با بالاي سربردن وزنه245 کيلوگرم دردو ضرب عنوان قهرماني خود را مسجل کرد. در روز پاياني اين مسابقات که حکم انتخابي المپيک 2012 لندن را نيز دارد در دسته فوق سنگين وزن بهداد سليمي از ايران با بلند کردن وزنه 208 کيلوگرمي پس از يوگني چيگيشف از روسيه در جاي دوم قرار گرفت. در حرکت يک ضرب چيگيشف از روسيه با 210 کيلوگرم اول شد و يوداچين از اوکراين با رکورد 205 کيلوگرم در مکان سوم قرار گرفت. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. بهداد در سالي كه گذشت قهرماني آسيا و جوانان جهان را كسب كرد. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در اين وزن هايبو سان از چين دوم شد و اوتا كازومي از ژاپن به مدال برنز رسيد. كيانوش رستمي كه سال گذشته در دسته 77 كيلوگرم وزنه زد و موفق شد مدال برنز حركت يكضرب را به خود اختصاص دهد با نظر كوروش باقري به دسته 85 كيلوگرم صعود كرد تا بدون وزن كم كردن عضلات خود را پر كرده و به سرعت در وزن جديد جا بيفتد. رستمي در ركوردگيري اخير انجام شده در سنندج به ترتيب وزنه‌هاي 171 كيلو در يكضرب و 210 كيلوگرم در دوضرب را مهار كرده تا با ركورد مجموع 381 كيلوگرم از شرايط خوبي براي ايستادن بر يكي از سكوهاي اول تا سوم جهاني در پاريس برخوردار باشد. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. انوشيرواني هم روي سكوي جهاني رفت سجاد انوشيرواني ديگر وزنه‌بردار كشور در دسته فوق‌سنگين ديشب درخشش بهداد سليمي را با كسب مدال برنز حركت يكضرب نقره دوضرب و نقره مجموع تكميل كرد. سجاد كه در آنتاليا بعد از انداختن وزنه 185 اين وزنه و وزنه 191 را در يكضرب بالاي سر برده بود اين بار هم پس از انداختن وزنه 193 در حركت دوم وزنه 193 را بالاي سر برد و بعد هم با مهار وزنه 198 كيلوگرم مدال برنز حركت يكضرب را براي خودش به ارمغان آورد. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. بهداد سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در رقابت‎هاي دسته فوق سنگين مسابقات جهاني آنتاليا سجاد انوشيرواني ديگر وزنه بردار ايران هم نمايش خيره کننده‎اي داشت و با وجود آنکه تنها 2 ماه از عمل جراحي زانويش مي گذشت با مهار وزنه‎هاي 191 کيلوگرم در يک ضرب و 235 کيلوگرم در دوضرب با مجموع 426 کيلوگرم در مکان پنجم جهان ايستاد و 22 امتياز با ارزش تيمي براي ايران در سهميه المپيک 2012 به دست آورد. سلیمی قهرمان ورزنه‌برداری کشور و قوی‌ترین مرد جهان است که در مسابقات اخیر غرب آسیا توانست رتبه ششم خود را به رتبه اول ارتقا داده و سه مدال طلا برای کشورمان به ارمغان آورد. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿بهداد سلیمی در رقابتهای وزنه‌برداری قهرمانی بزرگسالان جهان در آنتالیا عنوان قهرمانی را به دست آورد و در بازیهای آسیایی گوانگجو هم مدال طلای دسته فوق سنگین را به گردن آویخت تا از نگاه کاربران سایت فدراسیون جهانی وزنه برداری شایسته ترین فرد برای معرفی به عنوان بهترین وزنه بردار سال 2010 میلادی لقب بگیرد.
    بهداد سلیمی،‌ قوی‌ترین مرد جهان برای دریافت جایزه بهترین وزنه‌بردار سال 2010 جهان عازم ترکیه شد.علي سعيدلو ضمن تقدير از عملكرد سليمي در رقابت هاي جهاني آنتاليا تركيه، به منظور قدرداني از تلاش اين نماينده كشورمان در كسب مدال طلا وزنه برداري سنگين وزن جهان، 125 سكه بهار آزادي به وي هديه داد.
    در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني، مقام معظم رهبري در پيام كوتاهي، اين پيروزي ارزشمند را به وي تبريک گفتند. 
    سليمي ‌همه شرايط لازم براي انتخاب شدن به عنوان چهره جوان برتر ورزش در سال 89 را دارد.
    او جزو طلايي‌هاي گوانگجو بود و در اسفند ماه نيز در رقابت‌هاي كشوري ركورد رضازاده در حركت يك ضرب را شكست. اگر چه ركورد حركت دو ضرب رئيس فدراسيون وزنه‌برداري دست نخورده باقي ماند
    در برنامه‌اي كه در دستور كار بهداد سليمي قرار گرفته، هم مدال‌هاي طلاي جهاني و المپيك گنجانده شده و هم خلق ركوردهاي تازه‌اي براي وزنه‌برداري دنيا تا براي سال‌ها هيچكس نتواند به اين ركوردها نيز نزديك شود.  
    بهداد سلیمی وزنه‌بردار تیم ملی کشورمان در جریان رکوردگیری تیم ملی در سنندج رکورد 217 کیلوگرم را به نام خود ثبت کرد.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    تيم ملي وزنه‌برداري، در جدول مدالي هفتادونهمين دوره مسابقات جهاني وزنه‌برداري  نايب قهرمان جهان شد. در جدول امتيازي هم كه از اهميت بيشتري برخوردار است تيم ملي وزنه‌برداري كشورمان با  براي اولين بار در تاريخ، روي سكوي سومي دنيا ايستاد.رهبر معظم انقلاب در پيامي از تيم ملي وزنه‌برداري براي شاد كردن دل ملت عزيز ايران تشكر كردند. 
    رئیس فدراسیون جهانی وزنه برداری پس از شکستن رکورد یکضرب جهان توسط بهداد سلیمی به پشت صحنه مسابقات آمد تا به بهداد و حسین رضازاده تبریک بگوید.
    پس از درخشش وزنه‌برداری ایران در رقابت‌های وزنه‌برداری قهرمانی جهان در رقابت‌های پاریس 2011 و ، نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری به منظور انتخاب بهترین‌ وزنه‌بردار جهان قرار گرفته در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    قهرمان فوق سنگین وزن وزنه‌برداری جهان باحضور مسئولان شهرستان قائمشهر و استقبال بی‌نظیر  مردم مازندران وارد قائمشهر شد.
    فرماندار قائمشهر با اشاره به یا علی گویان سلیمی در مسابقات جهانی وزنه‌برداری فرانسه در کشور کفر و نفاق و در مقابل دیدگان میلیون‌ها انسان بر زبانش جاری شد، 
    حاوی پیام‌های ارزشمندبرای همه ملت‌های جهان به ویژه مسلمانان دانست. 
    نماینده مردم قائمشهر توسل به علی (ع) سلیمی را الگویی دانست که در ذهن جهان ایجاد شد و نشانگر طینت پاک و اهل بیت‌مدارانه این پهلوان است
    معاون سیاسی امنیتی استاندار مازندران نیز در این مراسم، بهداد سلیمی را قهرمان نامی جهان اسلام معرفی کرد 
    رئیس اداره ورزش و جوانان شهرستان قائمشهر نیزمدال ارزشمند جهان پهلوان بهداد سلیمی، اوج عظمت غرور و قدرت ایران را در جهان به اهتزاز درآورد
    همچنین طی مراسمی از تمبر یادبود بهداد سلیمی که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده است با امضای بهداد سلیمی و فرماندار قائمشهر رونمایی شد.
    
    
    OUR SUMMARY
    در بین ورزشکاران زن نیز رتبه نخست به خدیجه آزادپور ووشو اختصاص یافت و آرزو معتمدی قایقرانی مه لقا جان بزرگی تیراندازی و سوسن حاجی پور تکواندو در رتبه های بعد قرار گرفتند. سجاد انوشیروانی با اینکه به خاطر کسب مدال برنز بازی‌های آسیایی خدا را شکر می‌کند می‌گوید توقع بیش از این را از خودش داشته است؛ من آمده بودم به این مسابقات تا مدال نقره را بگیرم اما متأسفانه نشد. به گزارش واحد مركزي خبر سليمي در يک ضرب با 208 کيلوگرم دوم شده بود اما در دو ضرب با آسيب ديدگي وزنه‌بردار روس و کنار کشيدن از ادامه مسابقه با بالاي سربردن وزنه245 کيلوگرم دردو ضرب عنوان قهرماني خود را مسجل کرد. در روز پاياني اين مسابقات که حکم انتخابي المپيک 2012 لندن را نيز دارد در دسته فوق سنگين وزن بهداد سليمي از ايران با بلند کردن وزنه 208 کيلوگرمي پس از يوگني چيگيشف از روسيه در جاي دوم قرار گرفت. در حرکت يک ضرب چيگيشف از روسيه با 210 کيلوگرم اول شد و يوداچين از اوکراين با رکورد 205 کيلوگرم در مکان سوم قرار گرفت. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. بهداد در سالي كه گذشت قهرماني آسيا و جوانان جهان را كسب كرد. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در اين وزن هايبو سان از چين دوم شد و اوتا كازومي از ژاپن به مدال برنز رسيد. كيانوش رستمي كه سال گذشته در دسته 77 كيلوگرم وزنه زد و موفق شد مدال برنز حركت يكضرب را به خود اختصاص دهد با نظر كوروش باقري به دسته 85 كيلوگرم صعود كرد تا بدون وزن كم كردن عضلات خود را پر كرده و به سرعت در وزن جديد جا بيفتد. رستمي در ركوردگيري اخير انجام شده در سنندج به ترتيب وزنه‌هاي 171 كيلو در يكضرب و 210 كيلوگرم در دوضرب را مهار كرده تا با ركورد مجموع 381 كيلوگرم از شرايط خوبي براي ايستادن بر يكي از سكوهاي اول تا سوم جهاني در پاريس برخوردار باشد. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. انوشيرواني هم روي سكوي جهاني رفت سجاد انوشيرواني ديگر وزنه‌بردار كشور در دسته فوق‌سنگين ديشب درخشش بهداد سليمي را با كسب مدال برنز حركت يكضرب نقره دوضرب و نقره مجموع تكميل كرد. سجاد كه در آنتاليا بعد از انداختن وزنه 185 اين وزنه و وزنه 191 را در يكضرب بالاي سر برده بود اين بار هم پس از انداختن وزنه 193 در حركت دوم وزنه 193 را بالاي سر برد و بعد هم با مهار وزنه 198 كيلوگرم مدال برنز حركت يكضرب را براي خودش به ارمغان آورد. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. بهداد سليمي پس از حسين رضازاده نخستين وزنه بردار ايراني است که در دسته فوق سنگين مسابقات جهاني بزرگسالان براي ايران افتخارآفريني مي کند. در رقابت‎هاي دسته فوق سنگين مسابقات جهاني آنتاليا سجاد انوشيرواني ديگر وزنه بردار ايران هم نمايش خيره کننده‎اي داشت و با وجود آنکه تنها 2 ماه از عمل جراحي زانويش مي گذشت با مهار وزنه‎هاي 191 کيلوگرم در يک ضرب و 235 کيلوگرم در دوضرب با مجموع 426 کيلوگرم در مکان پنجم جهان ايستاد و 22 امتياز با ارزش تيمي براي ايران در سهميه المپيک 2012 به دست آورد. سلیمی قهرمان ورزنه‌برداری کشور و قوی‌ترین مرد جهان است که در مسابقات اخیر غرب آسیا توانست رتبه ششم خود را به رتبه اول ارتقا داده و سه مدال طلا برای کشورمان به ارمغان آورد. 
    -----------------------------------
    
    
    
    

## Using Twitter_FA word2vec


```python
evaluation_df2 = evaluation_multi(twitter_fa_w2v)
```

    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترین ورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.
    پس از درخشش پولاد مردان وزنه‌برداری ایران در رقابت‌های وزنه‌برداری قهرمانی جهان در رقابت‌های پاریس 2011 و کسب بهترین نتیجه تاریخ وزنه‌برداری برای کشورمان در این رقابت‌ها، نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری به منظور انتخاب بهترین‌ وزنه‌بردار جهان قرار گرفته است در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    
    پیش از ظهر امروز پهلوان بلندآوازه جهان، قهرمان فوق سنگین وزن وزنه‌برداری جهان با حضور استاندار، مدیران کل مازندران، مسئولان شهرستان قائمشهر و استقبال بی‌نظیر و سرشار از تقدیر و شعف مردم مازندران وارد قائمشهر شد.
    
    از تمبر یادبود بهداد سلیمی (قوی‌ترین ‌مرد جهان) که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده با امضای سلیمی و فرماندار قائمشهر رونمایی شد.
    
    طبق پیش‌بینی‌ها، بهداد سلیمی، قوی‌ترین مرد جهان بر سکوی نخست دسته فوق‌سنگین بازی‌های آسیایی گوانجو ایستاد. 
    مجله جهانی فدراسیون بین‌المللی، بهداد سلیمی بهترین را به عنوان بهترین وزنه‌بردار جهان در سال 2010 انتخاب کرد.
     علی پروین در جریان تمرین امروز پرسپولیس درباره قرمانی بهداد سلیمی گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.
    
    وزنه‌برداري قهرماني جهان - تركيه:بهداد سليمي قويترين مرد جهان شد
    : دكتر محمود احمدي نژاد طي پيامي، كسب عنوان قهرماني بهداد سليمي در رقابتهاي جهاني وزنه برداري را تبريك گفت.
    
    سليمي كه از همان روزهاي اول خودش را به عنوان يك رضا‌زاده جديد مطرح كرده بود سرانجام توانست ركورد رضا‌زاده را بشكند.
    بهداد اما به اين چيزها راضي نيست: «خوشحالم كه توانستم ركورد جديدي را به نام خودم ثبت كنم و همچنين به عنوان برترين وزنه‌بردار انتخاب شوم اما كار اصلي‌ام مانده است. من مي‌خواهم طلاي المپيك را بگيرم.»
    وزنه‌برداري قهرماني آسيا – چين:سليمي قهرمان دسته فوق سنگين شد
    رهبر معظم انقلاب :متشكرم، دل ملت ايران را شاد كرديد؛ركورد دنيا با فرياد ياعلي شكست
    رئیس فدراسیون جهانی وزنه برداری برای گفتن تبریک به بهداد سلیمی و حسین رضازاده به پشت صحنه مسابقات رفت.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    
    OUR SUMMARY
    کشور سرآمد و ممتاز دانست و در ادامه با تشکر از مربی سلیمی استاد غزالیان که نقش بی‌بدیلی در قهرمانی این پهلوان داشت خاطرنشان کرد سلیمی نمونه بارز تفکر بسیجی در ورزش است که قلبش برای نظام و آرمان‌های آن می‌طپد. بهداد سليمي كه بايد با فريادهاي روحيه‌بخش حسين رضازاده از كنار تخته براي شكستن ركورد وي آماده شود در اين‌باره به خبرنگار اعزامي جام‌جم گفت خوشبختانه ديگر مشكلي از بابت سرماخوردگي و گلو درد ندارم و وزنم در حال بازگشت به همان شرايط مطلوب قبلي است. اما نظر كوروش باقري در اين باره چيست؟ سرمربي تيم ملي گفت اگر ركورد حريفان بهداد بالا باشد براي وزنه اول او 201 كيلوگرم را انتخاب مي‌كنم بعد 206 و در آخر 214. اما اگر ببينم بهداد با وزنه كمتري هم در صدر قرار مي‌گيرد ريسك نمي‌كنم. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در رقابت‌هاي دوضرب نيز سليمي با مهار وزنه 250 كيلوگرمي در انتخاب دوم با مجموع 458 كيلوگرم به مقام قهرماني آسيا رسيد و عنوان قهرماني بازي‌هاي آسيايي 2010 گوانگجو را تكرار كرد. دعوت از كوروش باقري براي تصدي تيم ملي وزنه‌برداري با توجه به شناخت خوب و اعتماد چند ساله‌اي كه بين او و رضازاده وجود داشت حركت وزنه‌برداري و قهرمانان اين رشته را شتاب بخشيد تا وزنه‌برداري در 2 سال اخير بدون حاشيه به جلو قدم برداشته و قهرمانان خود را براي دستيابي به مدال‌هاي جهاني و المپيك حريص كند. پشتوانه قهرمانان وزنه‌برداري هر چند وزنه‌برداري ايران بايد سال‌ها از اسم و اعتبار حسين رضازاده در جهت مجد و عظمت خود بهره بگيرد و اصولا اين قهرمان سرمايه مادي و معنوي اين رشته به حساب مي‌آيد اما شانس بزرگ قهرمانان امروز تيم ملي اين است كه كوله باري از تجارب قهرماني را ضامن موفقيت و بزرگي خود در رقابت‌هاي آسيايي و جهاني ببينند. جدا از رضازاده كه از بالا روند آمادگي و تمرينات تيم ملي را رصد مي‌كند كوروش باقري قهرمان سال 2001 جهان و كسي كه در المپيك سيدني تا مرز مدال‌آوري پيش رفت و تنها به‌واسطه چند صد گرم اضافه وزن در جايگاه چهارم ايستاد نيز امروز در مقام سرمربي تيم ملي همه چيز را در كنترل خود دارد و به دقت روند پيشرفت سرمايه‌هاي وزنه‌برداري را دنبال مي‌كند. هر چند او گفته است براي المپيك لندن نيز به وزنه 220 كيلويي حمله خواهد كرد اما با توجه به اين‌كه بهداد ركورد دوضرب خود را نيز به نسبت جهاني سال گذشته در آنتالياي تركيه 15كيلويي افزايش داده و به مرز 260 كيلوگرم رسيده است انتظار مي‌رود كه اين وزنه‌بردار شايسته مازندراني در پاريس با حاشيه امنيتي خوب به نسبت حريفان جهاني دومين قهرماني خود را جشن بگيرد. چشم اميد به مدال‌آوري 2 جوان البته وزنه‌برداري ايران كه با 8 وزنه‌بردار در رقابت‌هاي جهاني فرانسه شركت مي‌كند تا با كسب امتيازهاي لازم 6سهميه كامل حضور در المپيك لندن را از آن خود كند جدا از بهداد سليمي در سنگين وزن روي مدال‌آوري 2جوان خوش‌استيل در 2 وزن 85 و 94 كيلوگرم نيز حساب مي‌كند. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. سليمي براي حركت آخر خود با انتخاب وزنه 260 كيلوگرم اين بار به ركورد مجموع حسين رضازاده كه 11 سال پيش و در المپيك سيدني با حدنصاب 472 به ثبت رسيده بود حمله كرد كه موفق نبود و با مجموع 464 كيلوگرم پاياني خوش را براي وزنه‌برداري ايران رقم زد. انوشيرواني در حركت دو ضرب هم كار خودش را با 228 كيلوگرم آغاز كرد و در حركت دوم هم وزنه 233 كيلوگرم را با موفقيت بالاي سر برد تا اين بار براي بدست آوردن مدال نقره دو ضرب و مجموع به وزنه 241 حمله كند. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي در حرکت دوضرب هم با درخواست وزنه 241 کيلوگرم ديرتر از ساير رقبايش روي تخته آمد و با بلند کردن اين وزنه و با توجه به مصدوميت چگيشف روس شانس خود را براي کسب عنوان قهرماني جهان افزايش داد. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. سليمي در حرکت دوضرب دسته فوق سنگين پس از ماتياس اشتاينر آلماني و قهرمان المپيک 2008 پکن که رکورد 246 کيلوگرم را براي خود به ثبت رسانده بود با رکورد 245 کيلوگرم به دومين مدال نقره خود دست يافت و درنهايت با مجموع 453 کيلوگرم و 13 کيلوگرم برتري نسبت به وزنه بردار آلماني ضمن کسب مدال طلاي مجموع دوحرکت روي سکوي قهرماني ايستاد. در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني مقام معظم رهبري در پيام كوتاهي اين پيروزي ارزشمند را به وي تبريک گفتند. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترینورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.سال گذشته بهداد سلیمی از کشورمان این عنوان را کسب کرد و به احتمال زیاد با توجه به اینکه وزنه‌بردار فوق سنگین کشورمان امسال با شکستن رکورد جهان به عنوان قهرمانی رقابت‌های پاریس 2011 رسید، امسال نیز بخت نخست کسب این عنوان محسوب می‌شود.
    قهرمان وزنه‌برداری جهان امروز با استقبال بی‌نظیر مردم مازندران وارد قائمشهر شد.سلیمی نام مولایش علی (ع) را بر لبان جهانیان به عنوان رمز قدرت، پهلوانی و جاودانگی جاری ساخت.تمبر یادبود بهداد سلیمی در اتحادیه تمبر جهان ثبت شد.بهداد سلیمی در رقابتهای وزنه‌برداری قهرمانی بزرگسالان جهان در آنتالیا عنوان قهرمانی را به دست آورد و در بازیهای آسیایی گوانگجو هم مدال طلای دسته فوق سنگین را به گردن آویخت. 
    علی پروین در جریان تمرین امروز پرسپولیس درباره قرمانی بهداد سلیمی گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.رهبرورییس جمهورطي پيامي، كسب عنوان قهرماني بهداد سليمي در رقابتهاي جهاني وزنه برداري را تبريك گفت.
    معاون رييس جمهوري و رييس سازمان تربيت بدني طي آييني با اهداي 125 سكه بهار آزادي از بهداد سليمي قهرمان سنگين وزن مسابقات وزنه برداري سال 2010 جهان در تركيه تجليل كرد.وزنه‌برداران فوق سنگين كشورمان قهرمان و نايب قهرمان رقابت‌‌هاي وزن 105+ كيلوگرم قهرماني آسيا شدند.بهداد سلیمی وزنه‌بردار تیم ملی کشورمان در جریان رکوردگیری تیم ملی در سنندج رکورد 217 کیلوگرم را به نام خود ثبت کرد.دکتر تاماش آیان رئیس فدراسیون جهانی وزنه برداری پس از شکستن رکورد یکضرب جهان توسط بهداد سلیمی به پشت صحنه مسابقات آمد تا به بهداد و حسین رضازاده تبریک بگوید.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.آیت الله جوادی آملی پس از دریافت این مدال‌ها خطاب به سلیمی گفت: «خداوند ان‌شاالله به شما عزت، شکوه و جلال دنیا و آخرت عطا کند و ‌در برنامه‌های ورزشی خود موفق باشید.»
    
    
    OUR SUMMARY
    کشور سرآمد و ممتاز دانست و در ادامه با تشکر از مربی سلیمی استاد غزالیان که نقش بی‌بدیلی در قهرمانی این پهلوان داشت خاطرنشان کرد سلیمی نمونه بارز تفکر بسیجی در ورزش است که قلبش برای نظام و آرمان‌های آن می‌طپد. بهداد سليمي كه بايد با فريادهاي روحيه‌بخش حسين رضازاده از كنار تخته براي شكستن ركورد وي آماده شود در اين‌باره به خبرنگار اعزامي جام‌جم گفت خوشبختانه ديگر مشكلي از بابت سرماخوردگي و گلو درد ندارم و وزنم در حال بازگشت به همان شرايط مطلوب قبلي است. اما نظر كوروش باقري در اين باره چيست؟ سرمربي تيم ملي گفت اگر ركورد حريفان بهداد بالا باشد براي وزنه اول او 201 كيلوگرم را انتخاب مي‌كنم بعد 206 و در آخر 214. اما اگر ببينم بهداد با وزنه كمتري هم در صدر قرار مي‌گيرد ريسك نمي‌كنم. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در رقابت‌هاي دوضرب نيز سليمي با مهار وزنه 250 كيلوگرمي در انتخاب دوم با مجموع 458 كيلوگرم به مقام قهرماني آسيا رسيد و عنوان قهرماني بازي‌هاي آسيايي 2010 گوانگجو را تكرار كرد. دعوت از كوروش باقري براي تصدي تيم ملي وزنه‌برداري با توجه به شناخت خوب و اعتماد چند ساله‌اي كه بين او و رضازاده وجود داشت حركت وزنه‌برداري و قهرمانان اين رشته را شتاب بخشيد تا وزنه‌برداري در 2 سال اخير بدون حاشيه به جلو قدم برداشته و قهرمانان خود را براي دستيابي به مدال‌هاي جهاني و المپيك حريص كند. پشتوانه قهرمانان وزنه‌برداري هر چند وزنه‌برداري ايران بايد سال‌ها از اسم و اعتبار حسين رضازاده در جهت مجد و عظمت خود بهره بگيرد و اصولا اين قهرمان سرمايه مادي و معنوي اين رشته به حساب مي‌آيد اما شانس بزرگ قهرمانان امروز تيم ملي اين است كه كوله باري از تجارب قهرماني را ضامن موفقيت و بزرگي خود در رقابت‌هاي آسيايي و جهاني ببينند. جدا از رضازاده كه از بالا روند آمادگي و تمرينات تيم ملي را رصد مي‌كند كوروش باقري قهرمان سال 2001 جهان و كسي كه در المپيك سيدني تا مرز مدال‌آوري پيش رفت و تنها به‌واسطه چند صد گرم اضافه وزن در جايگاه چهارم ايستاد نيز امروز در مقام سرمربي تيم ملي همه چيز را در كنترل خود دارد و به دقت روند پيشرفت سرمايه‌هاي وزنه‌برداري را دنبال مي‌كند. هر چند او گفته است براي المپيك لندن نيز به وزنه 220 كيلويي حمله خواهد كرد اما با توجه به اين‌كه بهداد ركورد دوضرب خود را نيز به نسبت جهاني سال گذشته در آنتالياي تركيه 15كيلويي افزايش داده و به مرز 260 كيلوگرم رسيده است انتظار مي‌رود كه اين وزنه‌بردار شايسته مازندراني در پاريس با حاشيه امنيتي خوب به نسبت حريفان جهاني دومين قهرماني خود را جشن بگيرد. چشم اميد به مدال‌آوري 2 جوان البته وزنه‌برداري ايران كه با 8 وزنه‌بردار در رقابت‌هاي جهاني فرانسه شركت مي‌كند تا با كسب امتيازهاي لازم 6سهميه كامل حضور در المپيك لندن را از آن خود كند جدا از بهداد سليمي در سنگين وزن روي مدال‌آوري 2جوان خوش‌استيل در 2 وزن 85 و 94 كيلوگرم نيز حساب مي‌كند. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. سليمي براي حركت آخر خود با انتخاب وزنه 260 كيلوگرم اين بار به ركورد مجموع حسين رضازاده كه 11 سال پيش و در المپيك سيدني با حدنصاب 472 به ثبت رسيده بود حمله كرد كه موفق نبود و با مجموع 464 كيلوگرم پاياني خوش را براي وزنه‌برداري ايران رقم زد. انوشيرواني در حركت دو ضرب هم كار خودش را با 228 كيلوگرم آغاز كرد و در حركت دوم هم وزنه 233 كيلوگرم را با موفقيت بالاي سر برد تا اين بار براي بدست آوردن مدال نقره دو ضرب و مجموع به وزنه 241 حمله كند. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي در حرکت دوضرب هم با درخواست وزنه 241 کيلوگرم ديرتر از ساير رقبايش روي تخته آمد و با بلند کردن اين وزنه و با توجه به مصدوميت چگيشف روس شانس خود را براي کسب عنوان قهرماني جهان افزايش داد. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. سليمي در حرکت دوضرب دسته فوق سنگين پس از ماتياس اشتاينر آلماني و قهرمان المپيک 2008 پکن که رکورد 246 کيلوگرم را براي خود به ثبت رسانده بود با رکورد 245 کيلوگرم به دومين مدال نقره خود دست يافت و درنهايت با مجموع 453 کيلوگرم و 13 کيلوگرم برتري نسبت به وزنه بردار آلماني ضمن کسب مدال طلاي مجموع دوحرکت روي سکوي قهرماني ايستاد. در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني مقام معظم رهبري در پيام كوتاهي اين پيروزي ارزشمند را به وي تبريک گفتند. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿ورزشی نویسان ایران با اختصاص بیشترین آراء، بهداد سلیمی و خدیجه آزادپور را به عنوان برترین ورزشکاران ایران در سال ۲۰۱۱ معرفی کردند.
    به گزارش باشگاه خبری فارس «توانا»، پس از درخشش پولاد مردان وزنه‌برداری ایران دررقابت های پاریس2011نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    به گزارش خبرگزاری فارس از شهرستان قائمشهر، مراسم تجلیل از جهان پهلوان بهداد سلیمی به نام نماد قدرت جهان امروز با شعفی غرورآفرین در سالن همایش‌های شهرداری قائمشهر برگزار شد. 
    مرد نیرومند ایران نام مولایش علی (ع) را بر لبان جهانیان به عنوان رمز قدرت، پهلوانی و جاودانگی جاری ساخت.
    سلیمی نام قائم شهر،شهرمزین به نام حضرت قائم رادرجهان جاودانه کرد.رئیس اداره ورزش وجوانان شهرستان قائم شهرنیزبااشاره به کسب هشت مدال جهانی توسط ورزشکاران قائم شهرازابتدای سال90تاکنون تصریح کرد:کمترشهری درکشورماهمچون قائم شهرصاحب این همه افتخارومدال جهانی است. 
    همچنین طی مراسمی از تمبر یادبود بهداد سلیمی که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده است با امضای بهداد سلیمی و فرماندار قائمشهر رونمایی شد.
    طبق پیش‌بینی‌ها، بهداد سلیمی، قوی‌ترین مرد جهان بر سکوی نخست دسته فوق‌سنگین بازی‌های آسیایی گوانجو ایستاد. 
    بهداد سلیمی،‌ قوی‌ترین مرد جهان برای دریافت جایزه بهترین وزنه‌بردار سال 2010 جهان عازم ترکیه شد.
    علی پروین گفت: بهداد سلیمی عمدا رکورد مجموع وزنه برداری دنیا را در مسابقات قهرمانی جهان نشکست تا این کار را در المپیک لندن انجام دهد.
    دکتراحمدی نژادطی پیامی این موفقیت بزرگ رابه آقای سلیمی ،خانواده محترم، مربيان گرامي و ملت شريف ايران تبريك عرض کردند.
    رهبر معظم انقلاب در پيامي از تيم ملي وزنه‌برداري براي شاد كردن دل ملت عزيز ايران تشكر كردند. 
    متن پيام حضرت آيت الله خامنه‌اي به شرح ذيل است.
    بسم الله الرحمن الرحيم 
    كاروان اعزامي به مسابقات وزنه‌برداري قهرماني جهان 
    سلام عليكم
    از شما جوانان و مدال آوران غيور به ويژه آقايان بهداد سليمي و كيانوش رستمي، كه با موفقيت خود در اين دوره از مسابقات قهرماني جهان، دل ملت عزيز ايران را شاد كرده‌ايد، تشكر مي‌كنم.
    سيد علي خامنه اي / 22 آبان
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    آیت الله جوادی آملی پس از دریافت این مدال‌ها خطاب به سلیمی گفت: «خداوند ان‌شاالله به شما عزت، شکوه و جلال دنیا و آخرت عطا کند و ‌در برنامه‌های ورزشی خود موفق باشید.»
    
    
    OUR SUMMARY
    کشور سرآمد و ممتاز دانست و در ادامه با تشکر از مربی سلیمی استاد غزالیان که نقش بی‌بدیلی در قهرمانی این پهلوان داشت خاطرنشان کرد سلیمی نمونه بارز تفکر بسیجی در ورزش است که قلبش برای نظام و آرمان‌های آن می‌طپد. بهداد سليمي كه بايد با فريادهاي روحيه‌بخش حسين رضازاده از كنار تخته براي شكستن ركورد وي آماده شود در اين‌باره به خبرنگار اعزامي جام‌جم گفت خوشبختانه ديگر مشكلي از بابت سرماخوردگي و گلو درد ندارم و وزنم در حال بازگشت به همان شرايط مطلوب قبلي است. اما نظر كوروش باقري در اين باره چيست؟ سرمربي تيم ملي گفت اگر ركورد حريفان بهداد بالا باشد براي وزنه اول او 201 كيلوگرم را انتخاب مي‌كنم بعد 206 و در آخر 214. اما اگر ببينم بهداد با وزنه كمتري هم در صدر قرار مي‌گيرد ريسك نمي‌كنم. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در رقابت‌هاي دوضرب نيز سليمي با مهار وزنه 250 كيلوگرمي در انتخاب دوم با مجموع 458 كيلوگرم به مقام قهرماني آسيا رسيد و عنوان قهرماني بازي‌هاي آسيايي 2010 گوانگجو را تكرار كرد. دعوت از كوروش باقري براي تصدي تيم ملي وزنه‌برداري با توجه به شناخت خوب و اعتماد چند ساله‌اي كه بين او و رضازاده وجود داشت حركت وزنه‌برداري و قهرمانان اين رشته را شتاب بخشيد تا وزنه‌برداري در 2 سال اخير بدون حاشيه به جلو قدم برداشته و قهرمانان خود را براي دستيابي به مدال‌هاي جهاني و المپيك حريص كند. پشتوانه قهرمانان وزنه‌برداري هر چند وزنه‌برداري ايران بايد سال‌ها از اسم و اعتبار حسين رضازاده در جهت مجد و عظمت خود بهره بگيرد و اصولا اين قهرمان سرمايه مادي و معنوي اين رشته به حساب مي‌آيد اما شانس بزرگ قهرمانان امروز تيم ملي اين است كه كوله باري از تجارب قهرماني را ضامن موفقيت و بزرگي خود در رقابت‌هاي آسيايي و جهاني ببينند. جدا از رضازاده كه از بالا روند آمادگي و تمرينات تيم ملي را رصد مي‌كند كوروش باقري قهرمان سال 2001 جهان و كسي كه در المپيك سيدني تا مرز مدال‌آوري پيش رفت و تنها به‌واسطه چند صد گرم اضافه وزن در جايگاه چهارم ايستاد نيز امروز در مقام سرمربي تيم ملي همه چيز را در كنترل خود دارد و به دقت روند پيشرفت سرمايه‌هاي وزنه‌برداري را دنبال مي‌كند. هر چند او گفته است براي المپيك لندن نيز به وزنه 220 كيلويي حمله خواهد كرد اما با توجه به اين‌كه بهداد ركورد دوضرب خود را نيز به نسبت جهاني سال گذشته در آنتالياي تركيه 15كيلويي افزايش داده و به مرز 260 كيلوگرم رسيده است انتظار مي‌رود كه اين وزنه‌بردار شايسته مازندراني در پاريس با حاشيه امنيتي خوب به نسبت حريفان جهاني دومين قهرماني خود را جشن بگيرد. چشم اميد به مدال‌آوري 2 جوان البته وزنه‌برداري ايران كه با 8 وزنه‌بردار در رقابت‌هاي جهاني فرانسه شركت مي‌كند تا با كسب امتيازهاي لازم 6سهميه كامل حضور در المپيك لندن را از آن خود كند جدا از بهداد سليمي در سنگين وزن روي مدال‌آوري 2جوان خوش‌استيل در 2 وزن 85 و 94 كيلوگرم نيز حساب مي‌كند. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. سليمي براي حركت آخر خود با انتخاب وزنه 260 كيلوگرم اين بار به ركورد مجموع حسين رضازاده كه 11 سال پيش و در المپيك سيدني با حدنصاب 472 به ثبت رسيده بود حمله كرد كه موفق نبود و با مجموع 464 كيلوگرم پاياني خوش را براي وزنه‌برداري ايران رقم زد. انوشيرواني در حركت دو ضرب هم كار خودش را با 228 كيلوگرم آغاز كرد و در حركت دوم هم وزنه 233 كيلوگرم را با موفقيت بالاي سر برد تا اين بار براي بدست آوردن مدال نقره دو ضرب و مجموع به وزنه 241 حمله كند. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي در حرکت دوضرب هم با درخواست وزنه 241 کيلوگرم ديرتر از ساير رقبايش روي تخته آمد و با بلند کردن اين وزنه و با توجه به مصدوميت چگيشف روس شانس خود را براي کسب عنوان قهرماني جهان افزايش داد. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. سليمي در حرکت دوضرب دسته فوق سنگين پس از ماتياس اشتاينر آلماني و قهرمان المپيک 2008 پکن که رکورد 246 کيلوگرم را براي خود به ثبت رسانده بود با رکورد 245 کيلوگرم به دومين مدال نقره خود دست يافت و درنهايت با مجموع 453 کيلوگرم و 13 کيلوگرم برتري نسبت به وزنه بردار آلماني ضمن کسب مدال طلاي مجموع دوحرکت روي سکوي قهرماني ايستاد. در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني مقام معظم رهبري در پيام كوتاهي اين پيروزي ارزشمند را به وي تبريک گفتند. 
    -----------------------------------
    
    
    
    -----------------------------------
    REF SUMMARY
    ﻿بهداد سلیمی در رقابتهای وزنه‌برداری قهرمانی بزرگسالان جهان در آنتالیا عنوان قهرمانی را به دست آورد و در بازیهای آسیایی گوانگجو هم مدال طلای دسته فوق سنگین را به گردن آویخت تا از نگاه کاربران سایت فدراسیون جهانی وزنه برداری شایسته ترین فرد برای معرفی به عنوان بهترین وزنه بردار سال 2010 میلادی لقب بگیرد.
    بهداد سلیمی،‌ قوی‌ترین مرد جهان برای دریافت جایزه بهترین وزنه‌بردار سال 2010 جهان عازم ترکیه شد.علي سعيدلو ضمن تقدير از عملكرد سليمي در رقابت هاي جهاني آنتاليا تركيه، به منظور قدرداني از تلاش اين نماينده كشورمان در كسب مدال طلا وزنه برداري سنگين وزن جهان، 125 سكه بهار آزادي به وي هديه داد.
    در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني، مقام معظم رهبري در پيام كوتاهي، اين پيروزي ارزشمند را به وي تبريک گفتند. 
    سليمي ‌همه شرايط لازم براي انتخاب شدن به عنوان چهره جوان برتر ورزش در سال 89 را دارد.
    او جزو طلايي‌هاي گوانگجو بود و در اسفند ماه نيز در رقابت‌هاي كشوري ركورد رضازاده در حركت يك ضرب را شكست. اگر چه ركورد حركت دو ضرب رئيس فدراسيون وزنه‌برداري دست نخورده باقي ماند
    در برنامه‌اي كه در دستور كار بهداد سليمي قرار گرفته، هم مدال‌هاي طلاي جهاني و المپيك گنجانده شده و هم خلق ركوردهاي تازه‌اي براي وزنه‌برداري دنيا تا براي سال‌ها هيچكس نتواند به اين ركوردها نيز نزديك شود.  
    بهداد سلیمی وزنه‌بردار تیم ملی کشورمان در جریان رکوردگیری تیم ملی در سنندج رکورد 217 کیلوگرم را به نام خود ثبت کرد.
    قوی ترین مرد جهان  قهرمان وزنه‌برداری کشورمان و قوی‌ترین مرد جهان مدال‌های طلای مسابقات غرب آسیای خود را تقدیم آیت ا‌لله جوادی آملی کرد.
    تيم ملي وزنه‌برداري، در جدول مدالي هفتادونهمين دوره مسابقات جهاني وزنه‌برداري  نايب قهرمان جهان شد. در جدول امتيازي هم كه از اهميت بيشتري برخوردار است تيم ملي وزنه‌برداري كشورمان با  براي اولين بار در تاريخ، روي سكوي سومي دنيا ايستاد.رهبر معظم انقلاب در پيامي از تيم ملي وزنه‌برداري براي شاد كردن دل ملت عزيز ايران تشكر كردند. 
    رئیس فدراسیون جهانی وزنه برداری پس از شکستن رکورد یکضرب جهان توسط بهداد سلیمی به پشت صحنه مسابقات آمد تا به بهداد و حسین رضازاده تبریک بگوید.
    پس از درخشش وزنه‌برداری ایران در رقابت‌های وزنه‌برداری قهرمانی جهان در رقابت‌های پاریس 2011 و ، نام بهداد سلیمی و کیانوش رستمی، دو طلایی ایران در رقابت‌های پاریس در نظرسنجی که در سایت فدراسیون جهانی وزنه‌برداری به منظور انتخاب بهترین‌ وزنه‌بردار جهان قرار گرفته در کنار 6 وزنه بردار دیگر به چشم می‌خورد.
    قهرمان فوق سنگین وزن وزنه‌برداری جهان باحضور مسئولان شهرستان قائمشهر و استقبال بی‌نظیر  مردم مازندران وارد قائمشهر شد.
    فرماندار قائمشهر با اشاره به یا علی گویان سلیمی در مسابقات جهانی وزنه‌برداری فرانسه در کشور کفر و نفاق و در مقابل دیدگان میلیون‌ها انسان بر زبانش جاری شد، 
    حاوی پیام‌های ارزشمندبرای همه ملت‌های جهان به ویژه مسلمانان دانست. 
    نماینده مردم قائمشهر توسل به علی (ع) سلیمی را الگویی دانست که در ذهن جهان ایجاد شد و نشانگر طینت پاک و اهل بیت‌مدارانه این پهلوان است
    معاون سیاسی امنیتی استاندار مازندران نیز در این مراسم، بهداد سلیمی را قهرمان نامی جهان اسلام معرفی کرد 
    رئیس اداره ورزش و جوانان شهرستان قائمشهر نیزمدال ارزشمند جهان پهلوان بهداد سلیمی، اوج عظمت غرور و قدرت ایران را در جهان به اهتزاز درآورد
    همچنین طی مراسمی از تمبر یادبود بهداد سلیمی که به ثبت جهانی upu اتحادیه تمبر ایران و جهان نیز رسیده است با امضای بهداد سلیمی و فرماندار قائمشهر رونمایی شد.
    
    
    OUR SUMMARY
    کشور سرآمد و ممتاز دانست و در ادامه با تشکر از مربی سلیمی استاد غزالیان که نقش بی‌بدیلی در قهرمانی این پهلوان داشت خاطرنشان کرد سلیمی نمونه بارز تفکر بسیجی در ورزش است که قلبش برای نظام و آرمان‌های آن می‌طپد. بهداد سليمي كه بايد با فريادهاي روحيه‌بخش حسين رضازاده از كنار تخته براي شكستن ركورد وي آماده شود در اين‌باره به خبرنگار اعزامي جام‌جم گفت خوشبختانه ديگر مشكلي از بابت سرماخوردگي و گلو درد ندارم و وزنم در حال بازگشت به همان شرايط مطلوب قبلي است. اما نظر كوروش باقري در اين باره چيست؟ سرمربي تيم ملي گفت اگر ركورد حريفان بهداد بالا باشد براي وزنه اول او 201 كيلوگرم را انتخاب مي‌كنم بعد 206 و در آخر 214. اما اگر ببينم بهداد با وزنه كمتري هم در صدر قرار مي‌گيرد ريسك نمي‌كنم. فردا و در فوق‌سنگين ما با سجاد انوشيرواني اين شانس را داريم كه مدال نقره را هم به دست آوريم. اگر بهداد با اين همه موفقيت چهره سال نشود پس چه كسي بشود؟ به گزارش فارس در روز پاياني رقابت‌هاي وزنه‌برداري قهرماني آسيا كه به ميزباني چين در حال برگزاري است بهداد سليمي وزنه‌بردار فوق سنگين كشور‌مان ابتدا موفق شد در حركت يك‌ضرب به مدال طلاي اين رقابت‌ها دست پيدا كند. در رقابت‌هاي دوضرب نيز سليمي با مهار وزنه 250 كيلوگرمي در انتخاب دوم با مجموع 458 كيلوگرم به مقام قهرماني آسيا رسيد و عنوان قهرماني بازي‌هاي آسيايي 2010 گوانگجو را تكرار كرد. دعوت از كوروش باقري براي تصدي تيم ملي وزنه‌برداري با توجه به شناخت خوب و اعتماد چند ساله‌اي كه بين او و رضازاده وجود داشت حركت وزنه‌برداري و قهرمانان اين رشته را شتاب بخشيد تا وزنه‌برداري در 2 سال اخير بدون حاشيه به جلو قدم برداشته و قهرمانان خود را براي دستيابي به مدال‌هاي جهاني و المپيك حريص كند. پشتوانه قهرمانان وزنه‌برداري هر چند وزنه‌برداري ايران بايد سال‌ها از اسم و اعتبار حسين رضازاده در جهت مجد و عظمت خود بهره بگيرد و اصولا اين قهرمان سرمايه مادي و معنوي اين رشته به حساب مي‌آيد اما شانس بزرگ قهرمانان امروز تيم ملي اين است كه كوله باري از تجارب قهرماني را ضامن موفقيت و بزرگي خود در رقابت‌هاي آسيايي و جهاني ببينند. جدا از رضازاده كه از بالا روند آمادگي و تمرينات تيم ملي را رصد مي‌كند كوروش باقري قهرمان سال 2001 جهان و كسي كه در المپيك سيدني تا مرز مدال‌آوري پيش رفت و تنها به‌واسطه چند صد گرم اضافه وزن در جايگاه چهارم ايستاد نيز امروز در مقام سرمربي تيم ملي همه چيز را در كنترل خود دارد و به دقت روند پيشرفت سرمايه‌هاي وزنه‌برداري را دنبال مي‌كند. هر چند او گفته است براي المپيك لندن نيز به وزنه 220 كيلويي حمله خواهد كرد اما با توجه به اين‌كه بهداد ركورد دوضرب خود را نيز به نسبت جهاني سال گذشته در آنتالياي تركيه 15كيلويي افزايش داده و به مرز 260 كيلوگرم رسيده است انتظار مي‌رود كه اين وزنه‌بردار شايسته مازندراني در پاريس با حاشيه امنيتي خوب به نسبت حريفان جهاني دومين قهرماني خود را جشن بگيرد. چشم اميد به مدال‌آوري 2 جوان البته وزنه‌برداري ايران كه با 8 وزنه‌بردار در رقابت‌هاي جهاني فرانسه شركت مي‌كند تا با كسب امتيازهاي لازم 6سهميه كامل حضور در المپيك لندن را از آن خود كند جدا از بهداد سليمي در سنگين وزن روي مدال‌آوري 2جوان خوش‌استيل در 2 وزن 85 و 94 كيلوگرم نيز حساب مي‌كند. پس از آن‌كه همه وزنه‌برداران سنگين وزن دنيا روي تخته آمدند نوبت هنرنمايي بهداد سليمي رسيد تا با همان انتخاب اول كادر فني و مهار وزنه 201 مدال طلاي خود را به گردن بياويزد. پس از آن نوبت شكستن ركورد جهان بود؛ ركوردي كه 8 سال پيش حسين رضازاده با مهار وزنه 213 كيلوگرمي به نام خود به ثبت رسانده بود اما انتخاب وزنه آخر حركت يكضرب مسابقات جهاني پاريس 214 كيلوگرم بود. اين وزنه‌بردار 22 ساله قائمشهري در حركت دوضرب هم اقتدار خود را به رخ جهانيان كشيد و با مهار وزنه 241 و 250 كيلوگرم براحتي مدال طلاي دوضرب و مجموع اين حركت را از آن خود كرد. سليمي براي حركت آخر خود با انتخاب وزنه 260 كيلوگرم اين بار به ركورد مجموع حسين رضازاده كه 11 سال پيش و در المپيك سيدني با حدنصاب 472 به ثبت رسيده بود حمله كرد كه موفق نبود و با مجموع 464 كيلوگرم پاياني خوش را براي وزنه‌برداري ايران رقم زد. انوشيرواني در حركت دو ضرب هم كار خودش را با 228 كيلوگرم آغاز كرد و در حركت دوم هم وزنه 233 كيلوگرم را با موفقيت بالاي سر برد تا اين بار براي بدست آوردن مدال نقره دو ضرب و مجموع به وزنه 241 حمله كند. سجاد در تمرينات تنها يك بار موفق به مهار وزنه 240 شده بود اما با غيرتي مثال زدني اين وزنه را هم بالاي سر نگه داشت تا دو مدال نقره با ارزش ديگر از آوردگاه جهاني پاريس به دست بياورد. سليمي در حرکت دوضرب هم با درخواست وزنه 241 کيلوگرم ديرتر از ساير رقبايش روي تخته آمد و با بلند کردن اين وزنه و با توجه به مصدوميت چگيشف روس شانس خود را براي کسب عنوان قهرماني جهان افزايش داد. در ادامه رقابت‎هاي دوضرب آرتين اوداچي ديگر رقيب سرسخت سليمي هم از ناحيه پا دچار آسيب ديدگي شد تا سليمي با بلند کردن وزنه 245 کيلوگرمي ضمن کسب عنوان قهرماني جهان بعد از 4 سال عنوان قويترين مرد جهان را نيز دوباره به ايران بازگرداند. سليمي در حرکت دوضرب دسته فوق سنگين پس از ماتياس اشتاينر آلماني و قهرمان المپيک 2008 پکن که رکورد 246 کيلوگرم را براي خود به ثبت رسانده بود با رکورد 245 کيلوگرم به دومين مدال نقره خود دست يافت و درنهايت با مجموع 453 کيلوگرم و 13 کيلوگرم برتري نسبت به وزنه بردار آلماني ضمن کسب مدال طلاي مجموع دوحرکت روي سکوي قهرماني ايستاد. در پي قهرماني بهداد سليمي وزنه بردار جوان و شايسته ايران در مسابقات جهاني مقام معظم رهبري در پيام كوتاهي اين پيروزي ارزشمند را به وي تبريک گفتند. 
    -----------------------------------
    
    
    
    

## Comparision 
### Our word2vec Results


```python
evaluation_df1.mean()
```




    rouge-1 f    0.319519
    rouge-1 p    0.264351
    rouge-1 r    0.468186
    rouge-2 f    0.139348
    rouge-1 p    0.117168
    rouge-1 r    0.222390
    dtype: float64



### Twitter word2vec Results


```python
evaluation_df2.mean()
```




    rouge-1 f    0.329835
    rouge-1 p    0.262846
    rouge-1 r    0.518164
    rouge-2 f    0.153218
    rouge-1 p    0.121955
    rouge-1 r    0.271698
    dtype: float64




```python
evaluation_df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rouge-1 f</th>
      <th>rouge-1 p</th>
      <th>rouge-1 r</th>
      <th>rouge-2 f</th>
      <th>rouge-1 p</th>
      <th>rouge-1 r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.230121</td>
      <td>0.182439</td>
      <td>0.319256</td>
      <td>0.054114</td>
      <td>0.039348</td>
      <td>0.090804</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.397606</td>
      <td>0.337436</td>
      <td>0.497798</td>
      <td>0.199047</td>
      <td>0.162953</td>
      <td>0.264761</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.350850</td>
      <td>0.288998</td>
      <td>0.499024</td>
      <td>0.154530</td>
      <td>0.121146</td>
      <td>0.251841</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.384825</td>
      <td>0.348571</td>
      <td>0.483169</td>
      <td>0.196868</td>
      <td>0.175736</td>
      <td>0.262583</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.378019</td>
      <td>0.316496</td>
      <td>0.548719</td>
      <td>0.141726</td>
      <td>0.117105</td>
      <td>0.238854</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.295098</td>
      <td>0.265987</td>
      <td>0.425980</td>
      <td>0.119174</td>
      <td>0.112674</td>
      <td>0.187653</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.403254</td>
      <td>0.358070</td>
      <td>0.564286</td>
      <td>0.247586</td>
      <td>0.225880</td>
      <td>0.350221</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.393967</td>
      <td>0.357720</td>
      <td>0.529827</td>
      <td>0.243227</td>
      <td>0.232941</td>
      <td>0.330378</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.438006</td>
      <td>0.469042</td>
      <td>0.474979</td>
      <td>0.269203</td>
      <td>0.313692</td>
      <td>0.286282</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.415241</td>
      <td>0.447985</td>
      <td>0.456930</td>
      <td>0.225227</td>
      <td>0.264273</td>
      <td>0.250845</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.292540</td>
      <td>0.211236</td>
      <td>0.484971</td>
      <td>0.122833</td>
      <td>0.082997</td>
      <td>0.242637</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.357208</td>
      <td>0.275362</td>
      <td>0.548739</td>
      <td>0.178582</td>
      <td>0.131720</td>
      <td>0.313321</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.342998</td>
      <td>0.251678</td>
      <td>0.575396</td>
      <td>0.149742</td>
      <td>0.102365</td>
      <td>0.309751</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.325049</td>
      <td>0.223948</td>
      <td>0.636985</td>
      <td>0.134171</td>
      <td>0.085351</td>
      <td>0.355699</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.292180</td>
      <td>0.187565</td>
      <td>0.670889</td>
      <td>0.153239</td>
      <td>0.094142</td>
      <td>0.418982</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.353631</td>
      <td>0.336185</td>
      <td>0.469204</td>
      <td>0.185900</td>
      <td>0.197429</td>
      <td>0.250466</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.374788</td>
      <td>0.341194</td>
      <td>0.491610</td>
      <td>0.201803</td>
      <td>0.179450</td>
      <td>0.298444</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.309848</td>
      <td>0.282586</td>
      <td>0.460783</td>
      <td>0.132612</td>
      <td>0.135525</td>
      <td>0.202287</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.390439</td>
      <td>0.391346</td>
      <td>0.450811</td>
      <td>0.202568</td>
      <td>0.223907</td>
      <td>0.234225</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.283796</td>
      <td>0.276737</td>
      <td>0.329663</td>
      <td>0.129094</td>
      <td>0.130284</td>
      <td>0.156123</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.272880</td>
      <td>0.178076</td>
      <td>0.662444</td>
      <td>0.130949</td>
      <td>0.081367</td>
      <td>0.412154</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.369703</td>
      <td>0.271722</td>
      <td>0.595891</td>
      <td>0.166398</td>
      <td>0.114017</td>
      <td>0.321128</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.297488</td>
      <td>0.223701</td>
      <td>0.460496</td>
      <td>0.089229</td>
      <td>0.063646</td>
      <td>0.157013</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.264182</td>
      <td>0.188842</td>
      <td>0.450290</td>
      <td>0.099136</td>
      <td>0.065971</td>
      <td>0.204934</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.387313</td>
      <td>0.303718</td>
      <td>0.547057</td>
      <td>0.215110</td>
      <td>0.163347</td>
      <td>0.322725</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.327951</td>
      <td>0.226955</td>
      <td>0.604855</td>
      <td>0.150604</td>
      <td>0.098773</td>
      <td>0.327069</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.372348</td>
      <td>0.290794</td>
      <td>0.527903</td>
      <td>0.158696</td>
      <td>0.119689</td>
      <td>0.243673</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.345432</td>
      <td>0.255000</td>
      <td>0.561312</td>
      <td>0.158561</td>
      <td>0.112876</td>
      <td>0.284477</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.330018</td>
      <td>0.238240</td>
      <td>0.541336</td>
      <td>0.145722</td>
      <td>0.097261</td>
      <td>0.293637</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.353698</td>
      <td>0.264392</td>
      <td>0.562232</td>
      <td>0.181013</td>
      <td>0.129485</td>
      <td>0.327534</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.415820</td>
      <td>0.357576</td>
      <td>0.498579</td>
      <td>0.213281</td>
      <td>0.176854</td>
      <td>0.270190</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.315897</td>
      <td>0.231742</td>
      <td>0.508360</td>
      <td>0.135081</td>
      <td>0.093151</td>
      <td>0.253166</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.285098</td>
      <td>0.198566</td>
      <td>0.520645</td>
      <td>0.122121</td>
      <td>0.081873</td>
      <td>0.248947</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.300342</td>
      <td>0.204990</td>
      <td>0.569215</td>
      <td>0.138282</td>
      <td>0.087880</td>
      <td>0.331820</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.304730</td>
      <td>0.204499</td>
      <td>0.620669</td>
      <td>0.154248</td>
      <td>0.098738</td>
      <td>0.368831</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.345792</td>
      <td>0.268953</td>
      <td>0.491243</td>
      <td>0.176978</td>
      <td>0.130580</td>
      <td>0.279214</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.264602</td>
      <td>0.177892</td>
      <td>0.547707</td>
      <td>0.115218</td>
      <td>0.073482</td>
      <td>0.292878</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.334482</td>
      <td>0.253746</td>
      <td>0.515692</td>
      <td>0.166403</td>
      <td>0.119608</td>
      <td>0.292694</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.312662</td>
      <td>0.213913</td>
      <td>0.583959</td>
      <td>0.117657</td>
      <td>0.076294</td>
      <td>0.259288</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.299035</td>
      <td>0.202673</td>
      <td>0.590854</td>
      <td>0.113535</td>
      <td>0.072895</td>
      <td>0.269512</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.318495</td>
      <td>0.297761</td>
      <td>0.355938</td>
      <td>0.119900</td>
      <td>0.113287</td>
      <td>0.134819</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.287146</td>
      <td>0.198976</td>
      <td>0.524616</td>
      <td>0.113709</td>
      <td>0.073684</td>
      <td>0.252526</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.263553</td>
      <td>0.167943</td>
      <td>0.623996</td>
      <td>0.110197</td>
      <td>0.066129</td>
      <td>0.336854</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.298762</td>
      <td>0.238275</td>
      <td>0.410696</td>
      <td>0.061606</td>
      <td>0.046266</td>
      <td>0.095209</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.321963</td>
      <td>0.233269</td>
      <td>0.530246</td>
      <td>0.150160</td>
      <td>0.101942</td>
      <td>0.292351</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.297289</td>
      <td>0.205587</td>
      <td>0.554629</td>
      <td>0.130665</td>
      <td>0.084940</td>
      <td>0.294274</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.279436</td>
      <td>0.193333</td>
      <td>0.517679</td>
      <td>0.110088</td>
      <td>0.072345</td>
      <td>0.239113</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.312001</td>
      <td>0.211024</td>
      <td>0.627066</td>
      <td>0.176853</td>
      <td>0.115789</td>
      <td>0.402900</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.289324</td>
      <td>0.209019</td>
      <td>0.516771</td>
      <td>0.134197</td>
      <td>0.092638</td>
      <td>0.283345</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.314851</td>
      <td>0.280531</td>
      <td>0.366795</td>
      <td>0.164075</td>
      <td>0.144022</td>
      <td>0.196451</td>
    </tr>
  </tbody>
</table>
</div>



## Extra Section
In this section we want to calculate the probability for a random surfer to be in a single sentence. It is the sum of probability for a random surfer to be in its words.<br>
So we use every single word as a node in the __page rank algorithm__. We run the algorithm and find importance of every word in the docuemnt. <br>
We define the importance of each sentence to be the sum of importance of each word in it. Then we use these numbers to determine which sentence has more information and should be used in the summary. 


```python
def get_all_words_and_classes(sentence_list, wv):
    all_words = []
    sentence_classes = []
    counter = 0
    for sentence in sentence_list:
        sentence_class_instance = Sentence()
        sentence_class_instance.start_index=counter
        for word in sentence:
            try:
                all_words.append(wv[word])
            except Exception:
                all_words.append(np.zeros(100))
            counter+=1
        sentence_class_instance.end_index=counter
        sentence_classes.append(sentence_class_instance)
    return  all_words, sentence_classes
```


```python
def distance_similariy(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.exp(-np.sqrt(np.dot(a-b,a-b)))         
```


```python
def summerize_text_extended(text, word_vector, ratio=0.2, compare_function = consine_similarity):
    wv = word_vector
    context = preprocessing(text)  
    sentence_list = get_sentence_list(context)
    all_word_vector, snt_clss_list = get_all_words_and_classes(sentence_list, wv)
    arr = make_graph(all_word_vector, compare_function)
    transitionWeights = make_input_for_page_rank(arr)
    rank_list = powerIteration(transitionWeights, rsp=0.15, epsilon=0.001, maxIterations=1000)
    counter=0
    for snt_cls in snt_clss_list:
        while (snt_cls.start_index <= counter <snt_cls.end_index):
            snt_cls.weight += rank_list[counter]
            counter+=1
    weight_list = [s.weight for s in snt_clss_list]
    
    rank_list = weight_list   
    zip = []
    for i in range(0, len(rank_list)):
        zip.append([i, rank_list[i]])
    sorted_zip = sorted(zip, key=lambda tup: tup[1], reverse=True)
#     print (sorted_zip)
    tmp_dict = dict()
    for i in range(0, len(sorted_zip)):
        tmp_dict[sorted_zip[i][0]]=i
    resource = list2sentence(sentence_list)
    summary = ""
    summary_sentences=int(len(resource)*ratio)
    for i in range(0, len(resource)):
        if (tmp_dict[i]<summary_sentences):
            summary+=resource[i]
    return summary
    
```


```python
def single_document_summarize_extended(wv, path = './Single -Dataset/Single -Dataset/Source/DUC', output_location ='./our_output/Single/our_summary/extended'
                                      ,compare_function=distance_similariy):
    all_files = os.listdir(path)   # imagine you're one directory above test dir
    all_sentences = []
    for i in range(0, len(all_files)):
        file_name = all_files[i]
    #     print (path+'\\'+str(file_name))
        with open(path+'/'+str(file_name), 'r', encoding='utf-8-sig') as f:
    #         print(file_name)
            context = f.read()
#             print (context)
            summary = summerize_text_extended(context, wv, ratio=0.4, compare_function=compare_function)
            write_summary(summary, file_name, location= output_location)
```


```python
class Sentence:
    def __init__(self):
        self.start_index = -1
        self.end_index = -1
        self.weight = 0
```


```python
single_document_summarize_extended(twitter_fa_w2v)
```


```python
result4 = evaluation(our_summary_path = '.\our_output\Single\our_summary\extended', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\our_word_2_vec_for_single_doc", number_of_print = 1)
```

    OUR SUMMARY
    و اما آن چه که موجب این تفاوت در آرا و نظرات شده متن اصلی اتفاقات و رویدادها نیست بلکه حواشی موجود در آن سبب این همه تنوع در تحلیل ها و مواضع شده گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. گروهی معتقدند اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است و قصدی برای مهاجرت ندارد اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد پس چه لزومی دارد از وی یک چهره بی بدیل فرهنگی ساخته شود و امکانات رسانه های تصویری و فرهنگی جمهوری اسلامی در خدمت مشهور شدن و مطرح کردن وی در سطح جامعه قرار بگیرند. در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد که گرچه آنها را زیاد ابراز نمی کند اما این گروه معتقدند وی در آینده پتانسیل اقدامات و مواضع ساختارشکنانه را خواهد داشت. در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند و وی را سفیر و ناجی سینمای ایران می خوانند شاید از بیان علت اصلی این تمجید ابا دارند این گروه دوم اتفاقا به همان دلایل ذکر شده روی وی حساب باز کرده اند این گروه شاید اصغر فرهادی را در راستای اهداف فرهنگی اجتماعی و حتی سیاسی خود تعریف کرده اند و معتقدند تفکر و رویکرد وی بیش از آنکه در خدمت گفتمان و اهداف انقلاب باشد می تواند بدون تقابل مستقیم به نقد صریح و رد پیش فرض های آموزه های دینی بپردازد و این تفکر و اندیشه اگر با جوایزی چون اسکار و حضوردر فستیوال های جهانی اعتبار یابد می تواند اثرگذاری مضاعفی داشته باشد. 
    REF SUMMARY
    جدایی نادر از... بالاخره خوب است یا نه
    واکنش ها وتحلیل های متفاوتی پیرامون فیلم جدایی نادر از سیمین و اصغر فرهادی منتشر می شود. 
    گروهی از سایت ها و نشریاتی که غالبا این فیلم را نفی کرده اند در واقع از بیان یک حقیقت طفره می روند و آن حقیقت اتفاقا ارتباطی با اصل فیلمنامه و خود فیلم ندارد. 
     اصغر فرهادی گرچه بارها اعلام کرده موطن وی ایران است 
     و قصدی برای مهاجرت ندارد، اما رویکرد و تفکر وی قرابت چندانی هم با گفتمان فرهنگی و هنری مد نظر انقلاب اسلامی ندارد و نمی توان روی وی به عنوان یک کارگردان انقلابی و متعهد حساب باز کرد. 
    در واقع این گروه از بیان این مطلب که به عقیده آنها اصغر فرهادی و گفتمانش تا حدی بر خلاف آموزه های دینی هستند طفره می روند و ترجیح می دهند این مخالفت را در قالب نقد فیلمنامه و احتمالا برداشت های سوئی که از فیلم می توان داشت بیان کنند 
    علت دیگر آن نیز شاید مواضع سیاسی انتقادی اصغر فرهادی باشد.
    در سوی دیگر نیز گروهی که این روزها به شدت از وی تعریف و تمجید می کنند.
    به همان دلایل ذکر شده روی وی حساب باز کرده اند. 
    حقیقت آنجاست که جدای از این حواشی، اگر جدایی نادر از سیمین در کشور دیگری جز ایران ساخته و اکران می شد هیچ گاه تا این اندازه مورد توجه قرار نمی گرفت. 
    
    


```python
result4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rouge-1 f</th>
      <th>rouge-1 p</th>
      <th>rouge-1 r</th>
      <th>rouge-2 f</th>
      <th>rouge-2 p</th>
      <th>rouge-2 r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.586464</td>
      <td>0.489947</td>
      <td>0.763853</td>
      <td>0.437030</td>
      <td>0.349853</td>
      <td>0.625015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.691734</td>
      <td>0.613115</td>
      <td>0.820927</td>
      <td>0.586131</td>
      <td>0.512632</td>
      <td>0.729597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.635872</td>
      <td>0.584000</td>
      <td>0.713743</td>
      <td>0.524990</td>
      <td>0.488000</td>
      <td>0.583175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.607153</td>
      <td>0.509677</td>
      <td>0.783216</td>
      <td>0.480189</td>
      <td>0.385214</td>
      <td>0.683837</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.632843</td>
      <td>0.529787</td>
      <td>0.793260</td>
      <td>0.511531</td>
      <td>0.421583</td>
      <td>0.658419</td>
    </tr>
  </tbody>
</table>
</div>




```python
result4.mean()
```




    rouge-1 f    0.515624
    rouge-1 p    0.459479
    rouge-1 r    0.693743
    rouge-2 f    0.381027
    rouge-2 p    0.343915
    rouge-2 r    0.544564
    dtype: float64



## Comparision
In this section we compare different models. <br>
In model 1, we used our trained word embedding. (Our dataset was really small, so we used only 8 dimension for vectors) <br>
In model 2, we used pre-trained word embedding. <br>
In model 3, we used extended version for page rank (sum of importance of words in a sentence)<br>
Model 3 has better performance in recall and f-score. 


```python
df = pd.concat([result1.mean(), result2.mean(), result4.mean()], axis=1)
df.columns = ["trained_word_embedding", "twitter_word_embedding","extended mode"]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trained_word_embedding</th>
      <th>twitter_word_embedding</th>
      <th>extended mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rouge-1 f</th>
      <td>0.508832</td>
      <td>0.506452</td>
      <td>0.515624</td>
    </tr>
    <tr>
      <th>rouge-1 p</th>
      <td>0.482383</td>
      <td>0.480294</td>
      <td>0.459479</td>
    </tr>
    <tr>
      <th>rouge-1 r</th>
      <td>0.634011</td>
      <td>0.631836</td>
      <td>0.693743</td>
    </tr>
    <tr>
      <th>rouge-2 f</th>
      <td>0.367171</td>
      <td>0.364011</td>
      <td>0.381027</td>
    </tr>
    <tr>
      <th>rouge-2 p</th>
      <td>0.353983</td>
      <td>0.351220</td>
      <td>0.343915</td>
    </tr>
    <tr>
      <th>rouge-2 r</th>
      <td>0.481875</td>
      <td>0.478495</td>
      <td>0.544564</td>
    </tr>
  </tbody>
</table>
</div>



## Future Work
Since LSTM has a good ability to model short term information, it can be used for modeling each sentence. <br>
So we can use bi-directional LSTM and choose concatenation of center word hidden state for bi-directional LSTM as a vector representation for every sentence. Then we again run __page rank algorihtm__ to determine the importance of each sentence. <br>

