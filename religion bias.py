#  This programmer aims to analyze religion bias in countries within most used languages  #

"""Packages"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import codecs
import os
import gc



def Import(lang, path):
    Target_lang = lang
    # print(len(model.words))
    word_list = pd.read_csv('wordlists/religion_Translations/' + lang + '.csv')
    # model = FastText.load_fasttext_format("D:\Data\Word Embedding\Vectors/" + path)

    ##########  Import Word Lists  ##################

    word_list = np.array(word_list)
    occupations = []
    Islam = []  # man
    Christianity = []  #  woman   this is woman bias
    Terrorism = []
    LenOfIslam = 18  # The length of lists for male and female
    LenOfChristianity = 15  # The length of lists for male and female
    LenOfNatrual = 48
    print(word_list[:, :])
    print(len(word_list[:, 1]))

    for i in range(0, LenOfIslam):
        Islam.append(word_list[i][0])
    print(Islam)
    for i in range(0, LenOfChristianity):
        Christianity.append(word_list[i][1])
    Christianity.append(word_list[i][1])
    for i in range(0, LenOfNatrual):
        Terrorism.append(word_list[i][2])
    print(Terrorism)

    print('''This is the Translated result of Terrorism word list for this language''')
    print("Import Loading")
    model = FastText.load_fasttext_format("D:\Data\Word Embedding\Vectors/" + path, encoding='utf-8')
    # model = KeyedVectors.load_word2vec_format("D:\Data\Word Embedding\Vectors/" + path, encoding='utf-8')
    return model, Islam, Christianity, Terrorism


def Get_filenames(dir):
    filenames = os.listdir(dir)
    return filenames


########################################################

def Religious_Bias(model, Islam, Christianity, Terrorism):
    '''Regious Bias is used to compute Christianity Bias, which means if the value is positive,
    it prefers Christianity'''

    '''Get corr word embedding values according to translation csv from we files'''
    Group_Isam_vec = []
    for item in Islam:
        try:
            Group_Isam_vec.append(model[item])
        except:
            continue
    Group_Isam_vec = np.array(Group_Isam_vec)
    missing = []

    # Get vectors of test word group
    Group_Christianity_vec = []
    for item in Christianity:

        try:
            Group_Christianity_vec.append(model[item])
        except:
            continue
    Group_Christianity_vec = np.array(Group_Christianity_vec)

    Group_Terrorism_vec = []
    for item in Terrorism:
        try:
            Group_Terrorism_vec.append(model[item])
        except:
            print('lost',item)
            continue
    Group_Terrorism_vec = np.array(Group_Terrorism_vec)


    error = 0
    # Make sure the set of info is correct.
    #  Compute the average vectors for men and women group
    Isam = np.mean(np.array(Group_Isam_vec), axis=0)
    Christianity = np.mean(np.array(Group_Christianity_vec), axis=0)
    Bias=[]
    for i in range(0, len(Group_Terrorism_vec)):
        Bias.append((np.linalg.norm(np.subtract(Group_Terrorism_vec[i], Isam))) - (np.linalg.norm(
            np.subtract(Group_Terrorism_vec[i], Christianity))))
    Avg_Bias = np.mean(Bias,axis=0)

    '''Test'''
    print('This is info of Terrorism')
    print(Group_Terrorism_vec)
    print(len(Group_Terrorism_vec))
    print(Bias)
    #Bias = np.linalg.norm(Isam) - np.linalg.norm(Christianity)

    return Avg_Bias


if __name__ == '__main__':
    path_dict = pd.read_csv('Table/chosen langs.csv')  # dict for lang and their file names

    '''Loop'''
    '''Single Trail'''
    single_path = pd.read_csv('Table/single trail.csv')
    All_vec = list(single_path.iloc[:,1])
    print(single_path.loc[:,:])
    print(All_vec)
    a=input()
    Bias_array=[]
    lang_array=[]
    print(single_path)
    for i in range(len(All_vec)):  # Only process those vec which are in our lists
        if All_vec[i] in single_path[:, 1]:  # if we have the vec, we start processing
            print(All_vec[i])
            for m in range(len(single_path)):  # get corresponding language name
                if single_path[m, 1] == All_vec[i]:
                    lang = single_path[m, 0]
                    path = single_path[m, 1]
                    print(lang, path)
                    model, Male, Female, Terrorism = Import(lang, path)
                    temp_bias = Religious_Bias(model, Male, Female, Terrorism)
                    print(lang,'   ',temp_bias)
                    Bias_array.append(temp_bias)
                    lang_array.append(lang)
                    del model
                    gc.collect()
    #print(Bias_array,len(Bias_array))
    #print(lang_array,len(lang_array))

    goon=input()








    #for i in range(len(path_dict)):
        #All_vec.append(path_dict.iloc[i, 1])
   # print(All_vec)
    All_vec = Get_filenames("D:\Data\Word Embedding\Vectors")
    path_dict = np.array(path_dict)
    Bias_array=[]
    lang_array=[]

    print(All_vec)
    print(path_dict[:, 1])
    print('Press any to go on')
    x = input()
    for i in range(len(All_vec)):  # Only process those vec which are in our lists
        if All_vec[i] in path_dict[:, 1]:  # if we have the vec, we start processing
            print(All_vec[i])
            for m in range(len(path_dict)):  # get corresponding language name
                if path_dict[m, 1] == All_vec[i]:
                    lang = path_dict[m, 0]
                    path = path_dict[m, 1]
                    print(lang, path)
                    model, Male, Female, Target_lang = Import(lang, path)
                    temp_bias = Religious_Bias(model, Male, Female, Target_lang)
                    print(lang,'   ',temp_bias)
                    Bias_array.append(temp_bias)
                    lang_array.append(lang)
                    del model
                    gc.collect()
    print(Bias_array,len(Bias_array))
    print(lang_array,len(lang_array))
    Bias_array=pd.DataFrame(Bias_array)
    lang_array.append('Vietnames')
    Bias_array.append(0.019574583)
    lang_array.append('Norwegian (Bokmål)')
    Bias_array.append(0.11692768)
    Bias_array.to_csv('test.csv')
    try:
        print(Bias_array[-2:])
        print(lang_array[-2:])
        file = np.column_stack((lang_array,Bias_array))
        file=pd.DataFrame(file)
        file.to_csv('Info for plot.csv')
    except:
        print('Error')
    print('Done')


'''Extra Computation'''
'''
model, Isam, Christianity, Target_lang = Import('Vietnames','cc.vi.300',)
print(Religious_Bias(model, Isam, Christianity, Target_lang))

model, Isam, Christianity, Target_lang = Import('Norwegian (Bokmål)','cc.no.300',)
print(Religious_Bias(model, Isam, Christianity, Target_lang))

print('Done')
a=input()
'''