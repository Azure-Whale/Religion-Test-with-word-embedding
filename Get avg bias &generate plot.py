import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# set the directory path, then you will get all files inside it


a=pd.read_csv('Regreesion&lists/Arabic.csv')
def Get_filenames(dir):
    filenames = os.listdir(dir)
    return filenames


def Plot(xlabel, ylabel, x_axis, y_axis,path):

    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.title('Religion Bias over 40 countries')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # sns.regplot(x=Difference_Index, y=Bias)
    plt.scatter(x_axis, y_axis, alpha=0.6, color='red')
    plt.xticks()
    plt.xticks(rotation=90)
    plt.savefig(path+'sample.png')
    plt.show()
    plt.cla()


def main():
    path = 'Regreesion&lists'
    files = Get_filenames(path)
    f_=[]
    sum_=[]
    for f in files:
        if f[-3:len(f)]=='csv':
            #print('yes')
            p=path+'/'+f
            #print(f)
            table = pd.read_csv(p)
            f_.append(f)
            sum = np.mean(table['bias'])
            sum_.append(sum)
    Info=[]
    for lang, bias in zip(f_, sum_):
        lang=lang.replace('.csv','')
        print(lang,bias)
        Info.append([lang, bias])
    print(f_)
    #print(sum_)
    #print(Info)
    Info=np.array(Info)

    Table=pd.DataFrame({'Lang':Info[:,0],'Bias':Info[:,1]})
    Table.to_csv('Plot/Coun-Bias.csv',index=False)

    file = pd.read_csv('Plot/Coun-Bias.csv')
    xlabel = 'Lang'
    ylabel = 'Christian Terror Bias'
    x_axis = file.iloc[:, 0]
    y_axis = file.iloc[:, 1]
    Plot(xlabel,ylabel,x_axis,y_axis,path='Plot/')

main()

