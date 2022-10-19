import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from env import user, password, host
import acquire
import prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 
import json
from typing import Dict, List, Optional, Union, cast
from IPython.display import display
from ipywidgets import IntProgress
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests import get
from env import github_token, github_username
import numpy as np
import Modeling
import unicodedata
import re
from wordcloud import WordCloud

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
α = .05
alpha= .05
from sklearn.impute import SimpleImputer
#image
from IPython.display import Image
from IPython.core.display import HTML 
#test
import scipy.stats as stats




#acquire----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username
import os
import json
from typing import Dict, List, Optional, Union, cast
from IPython.display import display
from ipywidgets import IntProgress
import requests
import pandas as pd
from bs4 import BeautifulSoup
import acquire
import time
from requests import get
from env import github_token, github_username
import matplotlib.pyplot as plt
import seaborn as sns 
import prepare
import numpy as np

REPOS = [
    "gocodeup/codeup-setup-script",
    "gocodeup/movies-application",
    "torvalds/linux",
]

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )
    
    
    
def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data(REPOS) -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
    json.dump(data, open("data.json", "w"), indent=1)
    
    
    
def acquire_data():
    list_rep1 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep1.append(repo.text)
    #
    list_rep2 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=2&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep2.append(repo.text)
    #
    list_rep3 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=3&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep3.append(repo.text)
            
    list_rep4 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=4&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep4.append(repo.text)
    list_rep5 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=5&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep5.append(repo.text)
    list_rep6 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=6&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep6.append(repo.text)
    list_rep7 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=7&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep7.append(repo.text)
    list_rep8 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=8&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep8.append(repo.text)
    list_rep9 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=9&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep9.append(repo.text)
    list_rep10 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=10&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep10.append(repo.text)
    list_rep11 = []

    for i in range(0,1):
        response = get('https://github.com/search?o=desc&p=11&q=stars%3A%3E1&s=forks&type=Repositories'.format(i))
        soup = BeautifulSoup(response.content, 'html.parser')

        for repo in soup.find_all('a', class_ = 'v-align-middle'):
            list_rep11.append(repo.text)
            
    df=list_rep1+list_rep2+list_rep3+list_rep4+list_rep5+list_rep6+list_rep7+list_rep8+list_rep9+list_rep10+list_rep11
    df = scrape_github_data(df)
    df=pd.DataFrame(df)
    return df


def csv_git():
    urlfile="https://raw.githubusercontent.com/Ibarra-Shenck/NLP-Project/main/github_forked.csv"

    mydata=pd.read_csv(urlfile)
    return mydata




def remove_stopwords(article_processed,words_to_add=[],words_to_remove=[]):
    ''' 
    takes in string, and two lists
    creates list of words to remove from nltk, modifies as dictated in arguements
    prints result of processing
    returns resulting string
    '''
    from nltk.corpus import stopwords
    #create the stopword list
    stopwords_list = stopwords.words("english")
    #modify stopword list
    [stopwords_list.append(word) for word in words_to_add]
    [stopwords_list.remove(word) for word in words_to_remove]
    #remove using stopword list
    words = article_processed.split()
    filtered_words = [w for w in words if w not in stopwords_list]
    #filtered_words =[word for word in article_processed if word not in stopwords_list]
    #print("removed ",len(article_processed)-len(filtered_words), "words")
    #join back
    article_without_stopwords = " ".join(filtered_words)
    return article_without_stopwords

def lemmatize(article):
    ''' 
    input article
    makes object, applies to string, and returns results
    '''
    import nltk
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    #use lemmatizer
    lemmatized = [wnl.lemmatize(word) for word in article.split()]
    #join words back together
    article_lemmatized = " ".join(lemmatized)
    return article_lemmatized

def stem(article):
    ''' 
    input string
    create object, apply it to the each in string, rejoin and return
    '''
    import nltk
    #create porter stemmer
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in article.split()]
    #join words back together
    article_stemmed = " ".join(stems)
    return article_stemmed

def tokenize(article0):
    ''' 
    input string
    creates object, returns string after object affect
    '''
    import nltk
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    #use the tokenizer
    article = tokenize.tokenize(article0,return_str=True)
    return article

def basic_clean(article0):
    ''' 
    input string
    lowers cases, makes "normal" characters, and removes anything not expected
    returns article
    '''
    import unicodedata
    import re
    #lower cases
    if isinstance(article0, float):
        article = str(article0).lower()
    else:
        article = article0.lower()
    ## decodes to change to "normal" characters after encoding to ascii from a unicode normalize
    article = unicodedata.normalize("NFKD",article).encode("ascii","ignore").decode("utf-8")
    # removes anything not lowercase, number, single quote, or a space
    article = re.sub(r'[^a-z0-9\'\s]','',article)
    return article








#prepare----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import acquire





def prepare_df(df):
    df = df[df.language.isna()==False]
    df["clean"] = [remove_stopwords(tokenize(basic_clean(each))) for each in df.readme_contents]
    df["stemmed"] = df.clean.apply(stem)
    df["lemmatized"] = df.clean.apply(lemmatize)
    # make a clean column (can drop other column later)
    df["count_set_lem"] = df["lemmatized"].str.strip().apply(set).apply(len)
    #create a list of low counts
    low_lang_count = df.language.value_counts(normalize=True)[df.language.value_counts(normalize=True).lt(.07)].index.tolist()
    df['clean_lang'] = np.where(df['language'].isin(low_lang_count),'Other',df['language'])
    #see the results
    df["clean_lang"].value_counts()
    df=df.drop(columns=('language'))
    count_feature_list=[]
    for each in df.clean_lang.unique():
        df[f"count_most_common_{each}"] = ""
        count_feature_list.append(f"count_most_common_{each}")
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    for each in count_feature_list:
        for row in df.index:
            match_list = lang[lang["Language"] == (each.split("_")[-1])]["most_common"].values[0]
            df[each].loc[row] = sum(map(lambda x: list(df["lemmatized"].loc[row].split()).count(x),match_list))
            #df[each] = each.split("_")[-1:]
    return df



def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test



#
#
#
#
#
#
#



def remove_stopwords(article_processed,words_to_add=[],words_to_remove=[]):
    ''' 
    takes in string, and two lists
    creates list of words to remove from nltk, modifies as dictated in arguements
    prints result of processing
    returns resulting string
    '''
    from nltk.corpus import stopwords
    #create the stopword list
    stopwords_list = stopwords.words("english")
    #modify stopword list
    [stopwords_list.append(word) for word in words_to_add]
    [stopwords_list.remove(word) for word in words_to_remove]
    #remove using stopword list
    words = article_processed.split()
    filtered_words = [w for w in words if w not in stopwords_list]
    #filtered_words =[word for word in article_processed if word not in stopwords_list]
    #print("removed ",len(article_processed)-len(filtered_words), "words")
    #join back
    article_without_stopwords = " ".join(filtered_words)
    return article_without_stopwords

def lemmatize(article):
    ''' 
    input article
    makes object, applies to string, and returns results
    '''
    import nltk
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    #use lemmatizer
    lemmatized = [wnl.lemmatize(word) for word in article.split()]
    #join words back together
    article_lemmatized = " ".join(lemmatized)
    return article_lemmatized

def stem(article):
    ''' 
    input string
    create object, apply it to the each in string, rejoin and return
    '''
    import nltk
    #create porter stemmer
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in article.split()]
    #join words back together
    article_stemmed = " ".join(stems)
    return article_stemmed

def tokenize(article0):
    ''' 
    input string
    creates object, returns string after object affect
    '''
    import nltk
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    #use the tokenizer
    article = tokenize.tokenize(article0,return_str=True)
    return article

def basic_clean(article0):
    ''' 
    input string
    lowers cases, makes "normal" characters, and removes anything not expected
    returns article
    '''
    import unicodedata
    import re
    #lower cases
    if isinstance(article0, float):
        article = str(article0).lower()
    else:
        article = article0.lower()
    ## decodes to change to "normal" characters after encoding to ascii from a unicode normalize
    article = unicodedata.normalize("NFKD",article).encode("ascii","ignore").decode("utf-8")
    # removes anything not lowercase, number, single quote, or a space
    article = re.sub(r'[^a-z0-9\'\s]','',article)
    return article

def basic_pipeline(codeup=True,news=True,words_keep=[],words_drop=[]):
    '''
    
    '''
    import acquire
    import pandas as pd

    #acquire
    news_df = pd.DataFrame(acquire.get_news_articles())
    codeup_df = pd.DataFrame(acquire.get_blog_content("https://codeup.com/blog/"))

    if codeup:
        codeup_df.rename(columns={"content":"original"},inplace=True)
        codeup_df["clean"] = [remove_stopwords(tokenize(basic_clean(each)),words_to_add=words_keep,words_to_remove=words_drop) for each in codeup_df.original]
        codeup_df["stemmed"] = codeup_df.clean.apply(stem)
        codeup_df["lemmatized"] = codeup_df.clean.apply(lemmatize)

    if news:
        news_df.rename(columns={"content":"original"},inplace=True),news_df.drop(columns="category",inplace=True)
        news_df["clean"] = [remove_stopwords(tokenize(basic_clean(each)),words_to_add=words_keep,words_to_remove=words_drop) for each in news_df.original]
        news_df["stemmed"] = news_df.clean.apply(stem)
        news_df["lemmatized"] = news_df.clean.apply(lemmatize)

    return codeup_df,news_df











#graph------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from env import user, password, host
import acquire
import prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 
import os
import json
from typing import Dict, List, Optional, Union, cast
from IPython.display import display
from ipywidgets import IntProgress
import requests
import pandas as pd
from bs4 import BeautifulSoup
import acquire
import time
from requests import get
from env import github_token, github_username
import matplotlib.pyplot as plt
import seaborn as sns 
import prepare
import numpy as np
import acquire
import Modeling
import unicodedata
import re
import json
from wordcloud import WordCloud

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import acquire

def explore_ttest_lang_setcount(df,population_name="clean_lang",numerical_feature="count_set_lem"):
    ''' 
    input df dataset and two strings (discrete and continous)
    does a ttest prints results, plots relation
    returns nothing
    '''
    import scipy.stats as stats

    has_similar_list=[]
    not_similar_list=[]

    for sample_name in df[population_name].unique():
        # sets variables
        alpha = .05
        print(numerical_feature,"<-target |",population_name,"<-population name |",sample_name,"<-sample name")

        #sets null hypothesis
        H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding Non-Repeating Words"
        Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding Non-Repeating Words"

        #runs test and prints results
        t, p = stats.ttest_1samp( df[df[population_name] == sample_name][numerical_feature], df[numerical_feature].mean())
        if p > alpha:
            print("We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
            has_similar_list.append(sample_name)
        else:
            print("We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
            not_similar_list.append(sample_name)
        print("----------")

    #plot the results
    plt.figure(figsize=(24,12))
    plt.suptitle(f"Sample Values Compared for Non-Repeating Words", fontsize=12, y=0.99)
    i=0
    for feature in df[population_name].unique():
        temp1=df.copy()
        #plots out a grouping of the features
        i+=1
        ax = plt.subplot(2,3,i)
        temp1[population_name] = np.where(temp1[population_name]==feature,feature,"Other Languages")
        temp1[[numerical_feature,population_name]].groupby(population_name).agg("mean").plot.bar(rot=0,color="white",edgecolor="grey",linewidth=5,ax=ax)
        ax.axhline(y=temp1[numerical_feature].mean(),label=f"Non-Repeating Words Mean {(round(temp1[numerical_feature].mean(),3))}",color="black",linewidth=3)
        ax.set_ylabel("% of Non-Repeating Words")
        plt.legend()
        ax.set_title(f"{feature} means Compared in relation to Count of Non-Repeating Words in Readme",fontsize=8)
    plt.show()
    
    print(f"The ones that are similar in value -> {*has_similar_list,} \nThe ones not similar in value -> {*not_similar_list,}")






def pie_chart1(df):
    labels = pd.concat([df.clean_lang.value_counts(),df.clean_lang.value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    plt.figure(figsize=(16,16))
    mylabels = df['clean_lang']
    textprops = {"fontsize":15}
    textprops = {"fontsize":15}
    plt.pie(labels.percent, labels = labels.index, textprops=textprops, autopct='%.1f%%')
    plt.legend()
    plt.title('Overall Language Distribution',fontsize=18)
    plt.show() 
    
    
    
    
def pie_chart(train):
    plt.figure(figsize=(18,8))
    sns.histplot(train.clean_lang, color= 'red')
    plt.show()
    
    
    
    
def graph2(df):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    sns.catplot(data=lang, x="count_set_words", y="Language", kind="bar",height=11,aspect=1.5)
    plt.title('Total count of words')
    plt.show()
    print(lang.count_set_words)
    
    
    
    
def java_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    java = ' '.join(train[train.clean_lang == 'Java'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(java, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='red', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring java bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
    
    
def Python_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    Python = ' '.join(train[train.clean_lang == 'Python'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Python, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='blue', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring python bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def HTML_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    HTML = ' '.join(train[train.clean_lang == 'HTML'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(HTML, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='red', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring HTML bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def JavaScript_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    JavaScript = ' '.join(train[train.clean_lang == 'JavaScript'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(JavaScript, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='purple', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring JavaScript bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def Ruby_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    Ruby = ' '.join(train[train.clean_lang == 'Ruby'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Ruby, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='yellow', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Ruby bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def other_bigrams(df,train):
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    Other = ' '.join(train[train.clean_lang == 'Other'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Other, 2))
                      .value_counts()
                      .head(20))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='green', width=.9, figsize=(10, 6))

    plt.title('20 Most frequently occuring Other bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])

def chi2_for_lang(train):
    ''' 
    input the df dataset, string of the target, and a list of featues to run through
    does a chi2 test for indepenance (proportionality) and plots the results
    no return
    '''
    # creates a column of the payments for easy analysis, runs a crosstab to put into a chi2 independancy test.
    # produces observed and expected values
    # returns the chi2 and pval for the whole set
    import scipy.stats as stats

    target="clean_lang"
    features=train.iloc[:,-6:].columns.tolist()

    df1 = pd.crosstab(train[target].unique(),features)
    for row in df1.index:
        for col in df1.columns:
            df1[col].loc[row] = train[train[target]==row][col].sum()

    chi2, p, degf, expected = stats.chi2_contingency(df1)

    alpha = .05
    H0 = (f"Languages is not different in the distribution of Common Unique Count")
    H1 = (f"Languages is different in the distribution of Common Unique Count")
    #print('Observed')
    #print(df1.values)
    #print('---\nExpected')
    dfexpected = df1.copy()
    for i in range(len(dfexpected)):
        dfexpected.iloc[i] = expected[i]
    #print(dfexpected.values)
    print(f'---\nchi^2 = {chi2:.4f}, p = {p:.5f}, degf = {degf}')
    if p>alpha:
        print(f"due to p={p:.5f} > α={alpha} we fail to reject our null hypothesis\n({H0})")
    else:
        print(f"due to p = {p:.5f} < α = {alpha} we reject our null hypothesis\n( ", '\u0336'.join(H0) + '\u0336' , ")")

    #plot the results
    plt.figure(figsize=(24,10))
    plt.suptitle(f"Common Unique Words in Language for each Language", fontsize=16, y=0.99)

    for x,col in enumerate(df1.T.columns):
        ax = plt.subplot(2,3,x+1)
        pd.concat({'Expected': dfexpected.T[col], 'Observed': df1.T[col]}, axis=1).\
            plot.barh(color={"Observed": "grey", "Expected": "pink"}, edgecolor="black",ax=ax)
        ax.set_ylabel("Count")
        ax.set_title(f'{col} values') # Title with column name.
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.8,
                    hspace=0.4)
    plt.show()




#modeling------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def score_models(X_train, y_train, X_validate, y_validate):
    '''
    Score multiple models on train and validate datasets.
    Print classification reports to decide on a model to test.
    Return each trained model, so I can choose one to test.
    models = dt_model1, rf_model, knn1_model.
    '''
    dt_model1 = DecisionTreeClassifier(max_depth = 7, random_state = 123)
    rf_model = RandomForestClassifier(min_samples_leaf = 1, max_depth = 10)
    knn1_model = KNeighborsClassifier()
    models = [dt_model1, rf_model, knn1_model]
    for model in models:
        model.fit(X_train, y_train)
        actual_train = y_train
        predicted_train = model.predict(X_train)
        actual_validate = y_validate
        predicted_validate = model.predict(X_validate)
        print(model)
        print('')
        print('train score: ')
        print(classification_report(actual_train, predicted_train))
        print('validate score: ')
        print(classification_report(actual_validate, predicted_validate))
        print('________________________')
        print('')
    return dt_model1, rf_model, knn1_model


def getting_ready(train,validate,test):
    ''' drop columns and spliting into x_train,y_train,x_val,y_val,x_test,y_test'''
    X_train = train.drop(columns=['repo','readme_contents','clean','stemmed','lemmatized','clean_lang'])
    y_train = train.clean_lang

    X_validate = validate.drop(columns=['repo','readme_contents','clean','stemmed','lemmatized','clean_lang'])
    y_validate = validate.clean_lang

    X_test = test.drop(columns=['repo','readme_contents','clean','stemmed','lemmatized','clean_lang'])
    y_test = test.clean_lang
    
    return (X_train,y_train,X_validate,y_validate,X_test,y_test)


def best_model(X_test,y_test):
    '''acquiring the best model aka decisiontree testing'''
    #best model we created 
    dt_model1 = DecisionTreeClassifier(max_depth = 7, random_state = 123)
    dt_model1.fit(X_test, y_test)
    actual_test = y_test
    predicted_test = dt_model1.predict(X_test)
    print(classification_report(actual_test, predicted_test))

def baseline1(train):
    # determine the percentage of customers that churn/do not churn
    baseline = train.y.value_counts().nlargest(1) / train.shape[0]
    print(f'My baseline accuracy is {round(baseline.values[0] * 100,2)}%.')
    
    
#modeling_graph-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def decision_tree_graph(X_test, y_test):
    # Create the tree
    tree = DecisionTreeClassifier(max_depth=7, random_state=123)

    # Fit the model on train
    tree = tree.fit(X_test, y_test)

    # Use the model
    # We'll evaluate the model's performance on train, first
    y_predictions = tree.predict(X_test)
    print('Accuracy of Decision Tree classifier on training set: {:.3f}'
      .format(tree.score(X_test, y_test)))

    # Visualizing the tree
    fig, ax = plt.subplots(figsize=(12,6), dpi = 300)
    plot_tree(tree, feature_names=X_test.columns, class_names=y_test.unique(), filled=True, fontsize=7)
    plt.show()




#stats---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def stat_test(df, train):
    '''
    Perform 1 sample t-test comparing mean length of original
    README file per language to the overall average length (all languages)
    set the significance level to 0.05
    '''
    lang_dict={"Language":[],"Words":[]}
    for lang in df["clean_lang"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(df[df["clean_lang"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    alpha = 0.05
    overall_mean_length_readme = lang[count_set_words].mean()
    for l in train.language.unique():
        sample = train[train.language == l]
        t,p = stats.ttest_1samp(sample[count_set_words], overall_mean_length_readme)
        print(l, round(t,5), p<alpha)
        
    return stat_test