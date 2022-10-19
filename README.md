# NLP - README Language Classification

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts, statistical tests, and various NLP tools to drive classification models; improving baseline model results.

> Construct models to predict `Language` (Class)
* Language: Primary Language the Repository was written, the Class Feature
* A observation of the Language as identified by the Class Feature

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute recorded presentation

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Classification model that accurately predicts `Language` using various techniques to create and guide feature selection for modeling  
>
> Find key drivers of `Language`
>
> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.
>
> Make recommendations on what works or doesn't work in predicting `Language`, and insights gained from Feature Creation

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
>
> 5 minute recording of a presentation of final notebook</br>
>
> 2-5 Slides as an Executive Report


## Data Dictionary
|       Target             |           Datatype       |     Definition      |
|:-------------------------|:------------------------:|-------------------:|  
Language              | 487 non-null  String   |   Classifier for Language

|       Feature            |           Datatype       |     Definition      |
|:------------------|:------------------:|--------------:|  
|repo                          |487 non-null    object| Name of the repo contents are found in at GITHUB  |
|language                      |487 non-null    object| The primary language the REPO was written in   |
|readme_contents               |487 non-null    object| Contents of the README  |
|clean                         |487 non-null    object| a tokenized, regex'd, simple cleaned return of the README  |
|stemmed                       |487 non-null    object| Stemmed version of the README  |
|lemmatized                    |487 non-null    object| Lemmatized version of the README  |
|clean_lang                    |487 non-null    object| Remap of the Language feature moving the lower percentile languages into `other`  |
|count_set_lem                 |487 non-null    int64 | count of non-repeating words in the lemmatized repo  |
|count_most_common_JavaScript  |487 non-null    object| count of words that match for Common Unique Words for Language `Javascript` |
|count_most_common_Python      |487 non-null    object| count of words that match for Common Unique Words for Language `Python` |
|count_most_common_Java        |487 non-null    object| count of words that match for Common Unique Words for Language `Java` |
|count_most_common_HTML        |487 non-null    object| count of words that match for Common Unique Words for Language `HTML` |
|count_most_common_Other       |487 non-null    object| count of words that match for Common Unique Words for Language `Other` |
|count_most_common_Ruby        |487 non-null    object| count of words that match for Common Unique Words for Language `Ruby` |

-----                    




# Initial Questions and Hypotheses

##  `Hypothesis 1 -` Are the mean count_values of Unique Words in Each Language equal to the Population in relation to Non-Repeating Words for the ReadMes**

> $H_0$: The mean values of `Non-Repeating Words(Readme)_langauge` will not be signifcantly different from `Non-Repeating Words(Readme)_population`.    
>
> $H_a$: Rejection of Null ~~The mean values of `Non-Repeating Words(Readme)_langauge` will not be signifcantly different from `Non-Repeating Words(Readme)_population`.~~  
> - Conclusion: There is enough evidence to reject our null hypothesis for SOME cases

##  `Hypothesis 2 -` Does each language have unique bigrams?

##  `Hypothesis 3 -` Is the distribution of Readmes similar for each Language?

##  `Hypothesis 4 -` Is the distribution of Count of Common words unique to a language a cood indicator of that language (Porportionality)**

> $H_0$: The distribution of `Non-Repeating Words(Readme)_langauge` will be consistent between languages.    
> $H_a$: Rejection of Null ~~The distribution of `Non-Repeating Words(Readme)_langauge` will be consistent between languages..~~  
> alpha = .05  
> - Conclusion: There is enough evidence to reject our null hypothesis for ALL cases

## Summary of Key Findings and Takeaways
 - Feature `Count_Most_Common_JavaScript` has good significance in determining ... JavaScript!
 - RFE Engineer Feature will also be useful in modeling as a way to reduce amount of Word Features (count)
 - We have isolated to a few main languages to help improve sample population to increase ability to predict
 - Features to direct in our modeling phase will include a blend of count for specific words as well as grouped words
-----
</br></br></br>

# Pipeline Walkthrough
## Plan
> Create and build out project README  
> Create required as well as supporting project modules and notebooks
* `wrangle.py`, `explore_modeling.py`,  `mvp.ipynb`
* `github_forked.csv`, `github_val_test.py`
> Decide which features to import   
> Decide how to deal with outliers, anomolies 

> Common words for each language
- Decide on which languages to use when crafting common sets
- Create language feature sets
- Add language labels as features  
> Statistical testing based on Common Unique Words for each Language
- Create functions that iterate through statistical tests
- Organize in Explore section 
> Explore
- Visualize language differences to gauge impact
- Rank languages based on statistical weight
> Modeling
* Create functions that automate iterative model testing
    - Adjust parameters and feature makes
* Handle acquire, prepare/split and scaling in wrangle/modelings
> Verify docstring is implemented for each function within all notebooks and modules 
 

## Acquire
> Acquired csv data from appropriate sources (or use refesh ability to download new repos from github)
* Create local .csv of raw data upon initial acquisition for later use if did not use supplied
* Review data on a preliminary basis
* Set Target Variable
> Add appropriate artifacts into `wrangle.py`

## Prepare
> Clean and basic exploration: 
* Stem and Lem the articles
> Handle any possible threats of data leakage (move acquisition of this data to modeling phase)
* Scale data (MinMaxScaler)


## Explore
> Bivariate exploration
* Investigate and visualize features against `Language`
> Identify additional possible areas for feature engineering (encoding counts)
* Use testing and visualizations to determine which features are significant in determining difference in `Language`
> Multivariate:
* Visuals exploring features as they relate to `Language`
> Statistical Analysis:
* Two tailed T-Test (Sample vs Population) for discrete vs continous
* Chi^2 for discrete vs discrete
> Collect and collate section *Takeaways*

## Model
> Ensure all data is scaled  
> Set up comparison for evaluation metrics and model descriptions    
> Set Baseline Prediction and evaluate Accuracy and F1 scores  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only single features
> Choose **Three** Best Models to add to final report

>Choose **one** model to evaluate on Test set
* Decision Tree
* Depth: 4
* Features: As determined by explore section and RFE recommendations
> Collect and collate section *Takeaways*

## Deliver
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
> Created recorded presentation for delivery
----
</br>

## Project Reproduction Requirements
> Requires personal `env.py` file containing github api and user-name (obtained in github)
> Steps:
* Fully examine this `README.md`
* Download `github_forked.csv`, `github_val_test.py`, `wrangle.py, explore_modeling.py`, and `mvp.ipynb` to working directory
* Run `mvp.ipynb`

## Canva slide show:
https://www.canva.com/design/DAFPfr48sw8/WsOPyP_dvzfwINgbxTIPzw/edit?utm_content=DAFPfr48sw8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
