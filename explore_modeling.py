from wrangle import *
import matplotlib.pyplot as plt


def chi2_for_lang(df):
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
    features=df.iloc[:,-6:].columns.tolist()

    df1 = pd.crosstab(df[target].unique(),features)
    for row in df1.index:
        for col in df1.columns:
            df1[col].loc[row] = df[df[target]==row][col].sum()

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

def explore_ttest_lang_setcount(df,population_name="clean_lang",numerical_feature="count_set_lem"):
    ''' 
    input df dataset and two strings (discrete and continous)
    does a ttest prints results, plots relation
    returns nothing
    '''
    import scipy.stats as stats
    import matplotlib.pyplot as plt

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
        plt.legend(loc="lower left")
        ax.set_title(f"{feature} means Compared in relation to Count of Non-Repeating Words in Readme",fontsize=8)
    plt.show()
    
    print(f"The ones that are similar in value -> {*has_similar_list,} \nThe ones not similar in value -> {*not_similar_list,}")


def get_val_test(df,lang,update=False):
    if update:
        val_test_repos = get_repo_names_val_test()
        validate_test_df = get_validate_test(True,val_test_repos)
    else:
        validate_test_df = get_validate_test(False,)

    validate_test_df = validate_test_df[validate_test_df.language.isna()==False]
    validate_test_df = validate_test_df[validate_test_df.readme_contents.isna()==False]
    print("dropped na")
    validate_test_df["clean"] = [remove_stopwords(tokenize(basic_clean(each))) for each in validate_test_df.readme_contents]
    print("cleaned")
    validate_test_df["stemmed"] = validate_test_df.clean.apply(stem)
    print("stemmed")
    validate_test_df["lemmatized"] = validate_test_df.clean.apply(lemmatize)
    print("lemmed")

    validate_test_df = validate_test_df[validate_test_df.lemmatized!=""]

    # make a clean column (can drop other column later)
    validate_test_df['clean_lang'] = np.where(validate_test_df['language'].isin(df["clean_lang"].unique()),validate_test_df['language'],'Other',)

    # get count of unique words in each readme (set unique not unique to itself)
    validate_test_df["count_set_lem"] = validate_test_df["lemmatized"].str.strip().apply(set).apply(len)


    count_feature_list=[]
    for each in validate_test_df.clean_lang.unique():
        validate_test_df[f"count_most_common_{each}"] = ""
        count_feature_list.append(f"count_most_common_{each}")


    for each in count_feature_list:
        for row in validate_test_df.index:
            match_list = lang[lang["Language"] == (each.split("_")[-1])]["most_common"].values[0]
            validate_test_df[each].loc[row] = sum(map(lambda x: list(validate_test_df["lemmatized"].loc[row].split()).count(x),match_list))
            #validate_test_df[each] = each.split("_")[-1:]

    return validate_test_df

def split_data(df,target="clean_lang"):
    ''' 
    takes in dataframe
    uses train test split on data frame using test size of 2, returns train_validate, test
    uses train test split on train_validate using test size of .3, returns train and validate
    returns train, validate test
    '''
    from sklearn.model_selection import train_test_split

    validate, test = train_test_split(df, test_size= .49, random_state=514,stratify = df[target])
    print(validate.shape, test.shape)
    return validate, test

def scale_split_data (train, validate, test, cols_to_scale=[]):
    ''' 
    takes in your three datasets
    applies minmax scaler to them using dtypes of number
    fits to those columns
    applies to copies of datasets
    returns datasets scaled
    '''
    from sklearn.preprocessing import MinMaxScaler  

    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #fit to data
    scaler.fit(train[cols_to_scale])

    # apply
    train_scaled[cols_to_scale] = scaler.transform(train[cols_to_scale])
    validate_scaled[cols_to_scale] =  scaler.transform(validate[cols_to_scale])
    test_scaled[cols_to_scale] =  scaler.transform(test[cols_to_scale])

    return train_scaled, validate_scaled, test_scaled

def prep_for_modeling(df,lang):

    validate_test_df = get_val_test(df,lang,False)
    #print("got data, cleaned and processed")
    validate, test = split_data(validate_test_df,target="clean_lang")
    #print("split validate and test")
    train_scaled, validate_scaled, test_scaled = scale_split_data (df, validate, test, cols_to_scale=df.iloc[:,-7:].columns)
    #print("scaled")
    # Build Model
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features = 55, ngram_range=(1, 3))

    words_train = tfidf.fit_transform(df['lemmatized'])
    X_train_b = pd.DataFrame(words_train.todense(), columns=tfidf.get_feature_names_out())

    words_validate = tfidf.transform(validate['lemmatized'])
    X_validate_b = pd.DataFrame(words_validate.todense(), columns=tfidf.get_feature_names_out())

    words_test = tfidf.transform(test['lemmatized'])
    X_test_b = pd.DataFrame(words_test.todense(), columns=tfidf.get_feature_names_out())

    ##setting the data sets based on features want to include
    X_train = train_scaled.iloc[:,-7:]
    y_train = train_scaled["clean_lang"]

    X_validate = validate_scaled.iloc[:,-7:]
    y_validate = validate_scaled["clean_lang"]

    X_test = test_scaled.iloc[:,-7:]
    y_test = test_scaled["clean_lang"]

    X_train = pd.concat([X_train.reset_index(),X_train_b],axis=1)
    X_train.drop(columns=["index"],inplace=True)
    X_validate = pd.concat([X_validate.reset_index(),X_validate_b],axis=1)
    X_validate.drop(columns=["index"],inplace=True)
    X_test = pd.concat([X_test.reset_index(),X_test_b],axis=1)
    X_test.drop(columns=["index"],inplace=True)

    return X_train,X_validate,X_test,y_train,y_validate,y_test,train_scaled,validate_scaled,test_scaled

def init_modeling(X_train,X_validate,y_train,y_validate):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score
    import sklearn

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier


    ## doing a basic decison tree classification
    ## fits the model, plots it, predicts off the training, and produces results (classification and confusion matrix)

    decision_tree_results = pd.DataFrame(columns=["leafs","acc_train","acc_val"])
    for i in range(1,10):
        clf = DecisionTreeClassifier(max_depth= i, random_state= 123, criterion="gini")
        clf = clf.fit(X_train,y_train)
        y_pred_train_dt = clf.predict(X_train)
        y_pred_val_dt = clf.predict(X_validate)
        acc_train = accuracy_score(y_pred_train_dt, y_train)
        acc_val = accuracy_score(y_pred_val_dt, y_validate)
        decision_tree_results.loc[len(decision_tree_results.index)]=[i,acc_train,acc_val]
    optimized_leafs=int(decision_tree_results[decision_tree_results.acc_val==decision_tree_results.acc_val.max()]["leafs"].min())
    clf = DecisionTreeClassifier(max_depth= int(optimized_leafs), random_state= 123, criterion="gini")
    clf = clf.fit(X_train,y_train)
    y_pred_train_dt = clf.predict(X_train)
    y_pred_val_dt = clf.predict(X_validate)
    #############################################

        ## logistic regression classifier, played with values until i found some i liked
    ## produces confusion and classification reports

    logreg = LogisticRegression(C=.1)#, class_weight={0:1, 1:99}, random_state=123, intercept_scaling=1, solver='lbfgs', max_iter=1000000000)
    logreg.fit(X_train,y_train)

    y_pred_train_array = logreg.predict(X_train)

    y_pred_logreg_val = logreg.predict(X_validate)
    #############################################

    ## random forest classifier, i played with the values until i settled on these
    ## fits the data, predicts on the training, and runs classification and confusion reports
    rf_results = pd.DataFrame(columns=["leafs","depth","acc_train","acc_val"])
    for i in range(1,8):
        for j in range(1,8):
            rf = RandomForestClassifier(bootstrap=True,class_weight=None,min_samples_leaf=i,n_estimators=100,max_depth=j,random_state=123)
            rf.fit(X_train,y_train)
            y_pred_rf = rf.predict(X_train)
            y_pred_rf_val = rf.predict(X_validate)
            acc_train = accuracy_score(y_pred_rf, y_train)
            acc_val = accuracy_score(y_pred_rf_val, y_validate)
            rf_results.loc[len(rf_results.index)]=[i,j,acc_train,acc_val]

    opt_leaf = int(rf_results[rf_results.acc_val==rf_results.acc_val.max()]["leafs"].min())
    opt_depth = int(rf_results[rf_results.acc_val==rf_results.acc_val.max()]["depth"].min())

    rf = RandomForestClassifier(bootstrap=True,class_weight=None,min_samples_leaf=opt_leaf,n_estimators=100,max_depth=opt_depth,random_state=123)
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_train)
    y_pred_rf_val = rf.predict(X_validate)


    ## kmeans classifier, 
    ## fits the data, predicts on the training, and runs classification and confusion reports
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(X_train, y_train)
    y_pred_train_neigh = neigh.predict(X_train)
    y_pred_val_neigh = neigh.predict(X_validate)

    #print(accuracy_score(y_pred_train_neigh, y_train))
    #print(accuracy_score(y_pred_val_neigh, y_validate))


    print(classification_report(y_train, y_pred_train_dt), "\t Decision Tree classification report on train set")
    print(classification_report(y_validate, y_pred_val_dt), "\t Decision Tree classification report on validate set")
    print("----------------")
    print(classification_report(y_train, y_pred_rf),"\t Random Forest train classification report")
    print(classification_report(y_validate, y_pred_rf_val),"\t Random Forest validate classification report")
    print("----------------")
    print(classification_report(y_train, y_pred_train_array),"\t Logistic Regression train classification")
    print(classification_report(y_validate, y_pred_logreg_val),"\t Logistic Regression validate classification")
    print("----------------")

    print(classification_report(y_train, y_pred_train_neigh),"\t KMeans train classification")
    print(classification_report(y_validate, y_pred_val_neigh),"\t KMeans validate classification")
    return optimized_leafs

def rfe(predictors_x,target_y,n_features):
    ''' 
    takes in the predictors (X) (predictors_x), the target (y) (target_y), and the number of features to select (k) 
    returns the names of the top k selected features based on the Recursive Feature Elimination class. and a ranked df
    '''

    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    model = LinearRegression()
    rfe = RFE(model,n_features_to_select=n_features)
    rfe.fit(predictors_x,target_y)

    print(pd.DataFrame({"rfe_ranking":rfe.ranking_},index=predictors_x.columns).sort_values("rfe_ranking")[:n_features])
    X_train_transformed = pd.DataFrame(rfe.transform(predictors_x),columns=predictors_x.columns[rfe.get_support()],index=predictors_x.index)
    X_train_transformed.head(3)

    var_ranks = rfe.ranking_
    var_names = predictors_x.columns.tolist()

    rfe_ranked = pd.DataFrame({'Var': var_names, 'Rank': var_ranks}).sort_values("Rank")
    
    return rfe_ranked