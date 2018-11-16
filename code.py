
# coding: utf-8

# # NLP Project
# 
# ## Summarization using Text Rank
# ### Ali Mortazavi
# In this project, we want to extract important sentences that can summarize the whole text.<br>
# We used __Page Rank Algorithm__ for determining the importance of each sentence. In this algorithm, we consider every sentence in the text as a node and then we have to determine the relationship between nodes. To find the relation between each sentence (nodes in the page rank graph), we used word2vec.<br>
# First, we trained a word2vec from our data. For determining a sentence vector, we used the average of word2vec of its words.
# Then for every document, we ran page rank algorithm then we selected n top sentence as an extractive summary.<br>
# (n = ratio *  number_of_document_sentences)<br>
# At the end, we used __ROUGE-1__ and __ROUGE-2__ for evaluation. 
# 

# Importing Libraries

# In[2]:


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


# ## Creating Word2Vec from Documents

#  We collect every sentence from Single Dataset for training the word2vec.

# In[3]:


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


# Since our dataset is small we select (windows size = 2) and vector (dimension = 8) to avoid overfitting. 

# In[4]:


path = get_tmpfile("word2vec.model")
model = Word2Vec(all_sentences, size=8, min_count=1, workers=4, sg=0, hs=0, window=2, iter=100)


# In[5]:


model.save(".\word2vec1.model")


# In[6]:


model.wv.save('.\word_vector1.kv')
wv = KeyedVectors.load('.\word_vector1.kv', mmap='r')


# Now we can see some word vectors.

# In[7]:


vector = wv['سلام'] 
vector


# In[8]:


vector = model.wv["کاهش"]
vector


# In[9]:


model.wv.most_similar("کاهش")


# In[10]:


tmp = model.wv["کاهش"]- model.wv["افزایش"]+model.wv["زیاد"]
model.similar_by_vector(tmp)


# In[11]:


tmp = model.wv["کاهش"]
model.similar_by_vector(tmp)


# ## Raeding Word2Vec from file
# We also use pretrained word2vec. 

# In[12]:


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
            


# In[13]:


a = np.array([twitter_fa_w2v["مرد"] - twitter_fa_w2v["زن"]])
b = np.array([twitter_fa_w2v["پسر"] - twitter_fa_w2v["دختر"]])
c  = np.array([(twitter_fa_w2v["مرد"] - twitter_fa_w2v["زن"] - (twitter_fa_w2v["پسر"] - twitter_fa_w2v["دختر"]))])
d = np.concatenate((a.T,b.T,c.T), axis=1)


# In word2vec, every dimension is correspond to one feature of the word. <br>
# In the example below, we see some dimensions are close to zero as we expected. 

# In[14]:


d = pd.DataFrame(d, columns=["مرد - زن", "پسر - دختر" , "تفاصل" ])


# In[259]:


d.head(16)


# ## Page Rank Algorithm
# We use page rank algorithm to determine the importance of each sentence. 
# 

# In[75]:


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


# In[17]:


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


# ## Preprocessing
# We use only three characters as a boundary for the sentences. (".", "?", "!") <br>
# We also remove all other delimiter characters from our data.

# In[56]:


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


# In[148]:


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


# In[20]:


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


# In[21]:


def make_input_for_page_rank(arr):
    ret = dict()
    n = len (arr)
    for i in range(0, n):
        tmp = dict()
        for j in range(0, n):
            tmp[j]=arr[j][i]
        ret[i]=tmp
    return ret
        


# In[22]:


def get_len(a):
    return np.sqrt(np.dot(a,a))


# In[23]:


def consine_similarity(a,b):
    return np.dot(a,b)/(get_len(a)*get_len(b))


# In[24]:


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
            


# In[25]:


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
    


# In[26]:


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


# ## ROUGE Metrics
# We used __ROUGE__ metrics to evaluate our results. <br>
# __ROUGE-n__ compares n-grams in reference summary and system summary. We reported __precision, recall, f-score__ for ROUGE-1 and ROUGE-2. 
# 

# In[27]:


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


# In the example below, we see precision, recall, F-score for unigram and bigram.

# In[28]:


ref = "A A B"
system = "A B A"
calculate_ROUGE_metrics (ref, system)


# In[29]:


def get_scores (reference_summary, our_summary):
    return calculate_ROUGE_metrics(reference_summary, our_summary)
    


# ## Single Document Dataset
# In this section, we want to summerize single document. 

# In[30]:


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


# ## Evaluation For Single Documents

# In[247]:



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


# ## Using Our trained word2vec

# In[234]:


single_document_summarize(twitter_fa_w2v, output_location ='./our_output/Single/our_word_2_vec_summaries')


# In[248]:


result1 = evaluation(our_summary_path = '.\our_output\Single\our_word_2_vec_summaries', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\our_word_2_vec_for_single_doc", number_of_print = 4)


# ## Using Twitter_FA word2vec

# In[218]:


single_document_summarize(twitter_fa_w2v, output_location ='./our_output/Single/twitter_word_2_vec_summaries')


# In[249]:


result2 = evaluation(our_summary_path = '.\our_output\Single\\twitter_word_2_vec_summaries', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\\twitter_word_2_vec_for_single_doc", number_of_print = 4)


# ## Comparasion - Single Document
# ### Our word2vec Results

# In[251]:


result1.head()


# In[252]:


result1.mean()


# The result for 5 different documents: (r = recall, p = precission, f = f1_score)

# ### Twitter word2vec Results

# In[250]:


result2.head()


# The average for whole dataset is:

# In[260]:


result2.mean()


# ## Multi Document
# In this section, we summerize multiple document. <br>
# In this case, we concatenate all sentences in all documents and then we run __page rank algorithm__ to prioritize every sentence. Then we choose top __k__ sentece as our summary. <br>
# ### Notice:
# #### k  = fixed ratio * number of sentences in the input.
# 

# In[32]:


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


# In[33]:


def read_XML (path):
#     print ("start reading "+ path)
    tree = ET.parse(path)
    root = tree.getroot()
    text = root.find("TEXT").text
    # print(text)
    return text


# In[34]:


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


# ## Using Our trained word2vec

# In[226]:


evaluation_df1 = evaluation_multi(wv)


# ## Using Twitter_FA word2vec

# In[228]:


evaluation_df2 = evaluation_multi(twitter_fa_w2v)


# ## Comparision 
# ### Our word2vec Results

# In[227]:


evaluation_df1.mean()


# ### Twitter word2vec Results

# In[229]:


evaluation_df2.mean()


# In[230]:


evaluation_df2


# ## Extra Section
# In this section we want to calculate the probability for a random surfer to be in a single sentence. It is the sum of probability for a random surfer to be in its words.<br>
# So we use every single word as a node in the __page rank algorithm__. We run the algorithm and find importance of every word in the docuemnt. <br>
# We define the importance of each sentence to be the sum of importance of each word in it. Then we use these numbers to determine which sentence has more information and should be used in the summary. 

# In[216]:


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


# In[217]:


def distance_similariy(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.exp(-np.sqrt(np.dot(a-b,a-b)))         


# In[218]:


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
    


# In[219]:


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


# In[220]:


class Sentence:
    def __init__(self):
        self.start_index = -1
        self.end_index = -1
        self.weight = 0


# In[ ]:


single_document_summarize_extended(twitter_fa_w2v)


# In[253]:


result4 = evaluation(our_summary_path = '.\our_output\Single\our_summary\extended', refereence_path =  ".\Single -Dataset\Single -Dataset\Summ\Extractive", prefix = 19, 
                evaluation_path = ".\Evaluation\our_word_2_vec_for_single_doc", number_of_print = 1)


# In[254]:


result4.head()


# In[255]:


result4.mean()


# ## Comparision
# In this section we compare different models. <br>
# In model 1, we used our trained word embedding. (Our dataset was really small, so we used only 8 dimension for vectors) <br>
# In model 2, we used pre-trained word embedding. <br>
# In model 3, we used extended version for page rank (sum of importance of words in a sentence)<br>
# Model 3 has better performance in recall and f-score. 

# In[256]:


df = pd.concat([result1.mean(), result2.mean(), result4.mean()], axis=1)
df.columns = ["trained_word_embedding", "twitter_word_embedding","extended mode"]
df


# ## Future Work
# Since LSTM has a good ability to model short term information, it can be used for modeling each sentence. <br>
# So we can use bi-directional LSTM and choose concatenation of center word hidden state for bi-directional LSTM as a vector representation for every sentence. Then we again run __page rank algorihtm__ to determine the importance of each sentence. <br>
# 
