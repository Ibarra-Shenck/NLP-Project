import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
import time
import pandas as pd
import numpy as np

from env import github_token, github_username


headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    print(url)
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
    #time.sleep(2)
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


def scrape_github_data(REPOS=[]) -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    print(len(REPOS),"count of repos")
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)

def get_repo_names():
    import random
    names = []
    langs = ['javascript', 'python', 'java', 'HTML', 'C++', 'Ruby']
    for lang in langs:
        page=1
        while page <= 15:
            ########https://github.com/search?l=python&o=1&q=desc&q=stars%3A%3E0&s=forks&type=Repositories?spoken_language_code=en
            ########https://github.com/search?o=1&p=2&q=stars%3A%3E0&s=forks&type=Repositories
            url = f'https://github.com/search?l={lang}&o=1&p={page}&q=stars%3A%3E0&s=forks&type=Repositories?spoken_language_code=en'
            soup = BeautifulSoup(requests.get(url).content, 'html.parser')
            repos = soup.select('a.v-align-middle')
            print(len(repos),"amount of repos on page")
            while len(repos) == 0:
                time.sleep(random.random()*10)
                print("not grabbing, and going to wait a few seconds and try again")
                url = f'https://github.com/search?l={lang}&o=1&p={page}&q=stars%3A%3E0&s=forks&type=Repositories?spoken_language_code=en'
                soup = BeautifulSoup(requests.get(url).content, 'html.parser')
                repos = soup.select('a.v-align-middle')
                print(len(repos),"amount of repos on page")
            print("grab")
            for r in repos:
                repo_name = r['href']
                names.append(repo_name)

            print('finishing page '+str(page))
            page += 1
            print(len(names), "length of repo list")
            #time.sleep(random.random()*5)
    names = [i[1:] for i in names]
    return names

def to_update_or_not_to_udpate(update_flag=True,list_repo=[]):
    ''' 
    optional inputs of updating
    if updating it will pull from most forked repos, at given (default 100 count), then make a csv
    otherwise pulls from csv
    '''
    import acquire
    import time
    from os.path import exists
    from datetime import datetime
    import random
    
    if update_flag:
        dictionary_of_repos = scrape_github_data(list_repo)
        df = pd.DataFrame(dictionary_of_repos)
        df.to_csv(f'github_forked.csv', index=False)
    else:
        if exists('github_forked.csv'):
            df = pd.read_csv('github_forked.csv')
        else:
            print("can not find file, please update instead")
        
    return df

def get_repo_names_val_test():
    import random
    names = []
    langs = ['javascript', 'python', 'java', 'HTML', 'C++', 'Ruby']
    for lang in langs:
        page=1
        while page <= 1:
            ########https://github.com/search?l=python&o=1&q=desc&q=stars%3A%3E0&s=forks&type=Repositories?spoken_language_code=en
            ########https://github.com/search?l=JavaScript&o=desc&p=2&q=stars%3A%3E0&s=updated&type=Repositories?spoken_language_code=en
            url = f'https://github.com/search?l={lang}&o=desc&p={page}&q=stars%3A%3E0&s=updated&type=Repositories?spoken_language_code=en'
            soup = BeautifulSoup(requests.get(url).content, 'html.parser')
            repos = soup.select('a.v-align-middle')
            print(len(repos),"amount of repos on page")
            while len(repos) == 0:
                time.sleep(random.random()*30)
                print("not grabbing, and going to wait a few seconds and try again")
                soup = BeautifulSoup(requests.get(url).content, 'html.parser')
                repos = soup.select('a.v-align-middle')
                print(len(repos),"amount of repos on page")
            print("grab")
            for r in repos:
                repo_name = r['href']
                names.append(repo_name)

            print('finishing page '+str(page))
            page += 1
            print(len(names), "length of repo list")
            #time.sleep(random.random()*5)
    names = [i[1:] for i in names]
    return names

def get_validate_test(update_flag=True,list_repo=[]):
    ''' 
    optional inputs of updating
    if updating it will pull from most forked repos, at given (default 100 count), then make a csv
    otherwise pulls from csv
    '''
    import acquire
    import time
    from os.path import exists
    from datetime import datetime
    import random
    
    if update_flag:
        dictionary_of_repos = scrape_github_data(list_repo)
        df = pd.DataFrame(dictionary_of_repos)
        df.to_csv(f'github_val_test.csv', index=False)
    else:
        if exists('github_val_test.csv'):
            df = pd.read_csv('github_val_test.csv')
        else:
            print("can not find file, please update instead")
        
    return df













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
    

def prep_work(df):   
    df = df[df.language.isna()==False]
    print("dropped na")
    df["clean"] = [remove_stopwords(tokenize(basic_clean(each))) for each in df.readme_contents]
    print("cleaned")
    df["stemmed"] = df.clean.apply(stem)
    print("stemmed")
    df["lemmatized"] = df.clean.apply(lemmatize)
    print("lemmed")


    #create a list of low counts
    low_lang_count = df.language.value_counts(normalize=True)[df.language.value_counts(normalize=True).lt(.08)].index.tolist()

    # make a clean column (can drop other column later)
    df['clean_lang'] = np.where(df['language'].isin(low_lang_count),'Other',df['language'])
    #see the results
    df["clean_lang"].value_counts()

    # get count of unique words in each readme (set unique not unique to itself)
    df["count_set_lem"] = df["lemmatized"].str.strip().apply(set).apply(len)


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
        most_common = looped_series[looped_series > looped_series.quantile(.995)]
        most_common_list.append(most_common.index.tolist())

    #unique_words = set.intersection(*map(set,most_common_list))

    lang["most_common"] = pd.Series(most_common_list)

    for iter_num1,iter_list1 in enumerate(lang["most_common"]):
        for iter_num2,iter_list2 in enumerate(lang["most_common"]):
            #print(iter_num1,iter_num2)
            if not iter_num2 == iter_num1:
                temp_list=[]
                for word in lang["most_common"].loc[iter_num1]:
                    if word not in (list(set(lang["most_common"].loc[iter_num1]) & set(lang["most_common"].loc[iter_num2]))):
                        temp_list.append(word)
                lang["most_common"].loc[iter_num1]=temp_list
                
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)

    for each in count_feature_list:
        for row in df.index:
            match_list = lang[lang["Language"] == (each.split("_")[-1])]["most_common"].values[0]
            df[each].loc[row] = sum(map(lambda x: list(df["lemmatized"].loc[row].split()).count(x),match_list))
            #df[each] = each.split("_")[-1:]


    return lang, df, most_common_list, count_feature_list, low_lang_count