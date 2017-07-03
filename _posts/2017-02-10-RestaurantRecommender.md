---
layout: post
title:  "Restaurant Recommender"
date:   2017-02-10 21:04:11 -0500
categories: jekyll update
---

# Introduction

In this project, I extracted restaurant ratings and reviews from [Foursquare](https://foursquare.com/) and used distance (one of the main ideas behind recommender systems) to generate recommendations for restaurants in one city that have similar reviews to restaurants in another city.  This post is the abridged version, but check out [my github post](https://github.com/JeffMacaluso/Blog/blob/master/Restaurant%20Recommender.ipynb) for all of the code if you are curious or want to use it.

### Motivation

I grew up in Austin, Texas, and moved to Minneapolis, Minnesota for my wife's work a few years ago. My wife and I are people who love food, and loved the food culture in Austin. After our move, we wanted to find new restaurants to replace our favorites from back home.  However, most decently rated places in Minneapolis we went to just didn't quite live up to our Austin expectations.  These restaurants usually came at the recommendations of locals, acquaintances, or from Google ratings, but the food was often bland and overpriced.  It took us one deep dive into the actual reviews to figure it out.  

In order to better illustrate our problem, below are recent reviews from three restaurants that left us disappointed.  On the left is a pizza shop, the middle is a typical American restaurant, and the right is a Vietnamese restaurant:

<img src="https://github.com/JeffMacaluso/JeffMacaluso.github.io/blob/master/_posts/RestaurantRecommender_files/Reviews.png?raw=true">

I highlighted the main points to stand out against the small font. Service, atmosphere, and apparently eggrolls were the most common and unifying factors. You see very little discussion on the quality of the actual food,  and you can even see an instance where a reviewer rates the pizza place as 5/5 even after saying that it is expensive. I began to notice a disconnect in how I evaluate restaurants versus how the people of Minneapolis evaluate restaurants. If you have previously worked with recommender systems, you already know where I'm going with this. If not, here is a primer:

### Recommender Systems Overview

Before getting into the overview of recommender systems, I wanted to point out that I won't actually be building a legitimate recommender system in this notebook.  There are some [great](https://turi.com/learn/userguide/recommender/introduction.html) [packages](https://github.com/lyst/lightfm) for doing so, but I'm going to stick with one of the main ideas behind recommender systems.  This is for two reasons:

**1)** Classes have started back up, and my available free time for side projects like this one is almost non-existant.

**2)** My gastronomic adventures don't need the added benefits that a recommender system provides over what I'll be doing.

Let's get back into it.

In the world of recommender systems, there are three broad types:

- **[Collaborative Filtering (user-user)](https://en.wikipedia.org/wiki/Collaborative_filtering)**: This is the most prevalent type of recommender systems that uses "wisdom of the crowd" for popularity among peers. This option is particularly popular because you don't need to know a lot about the item itself, you only need the ratings submitted by reviewers. The two primary restrictions are that it makes the assumption that peoples' tastes do not change over time, and new items run into the "[cold start problem](https://en.wikipedia.org/wiki/Cold_start)".  This is when either a new item has not yet received any ratings and fails to appear on recommendation lists, or a new user has not reviewed anything so we don't know what their tastes are.
    - **E.x.:** People who like item **X** also like item **Y**
        - This is how Spotify selects songs for your recommended play list.  Specifically, it will take songs from other play lists that contain songs you recently liked.
    
    
- **[Content-Based (item-item)](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)**: This method recommends items based off of their similarity to other items.  This requires reliable information about the items themselves, which makes it difficult to implement in a lot of cases.  Additionally, recommendations generated from this will option likely not deviate very far from the item being compared to, but there are tricks available to account for this.
    - **E.x.:** Item **X** is similar to item **Y**
        - This is how Pandora selects songs for your stations.  Specifically, it assigns each song a list of characteristics (assigned through the [Music Genome Project](http://www.pandora.com/corporate/mgp.shtml)), and selects songs with similar characteristics as those that you liked.
    

- **[Hybrid](https://en.wikipedia.org/wiki/Recommender_system#Hybrid_recommender_systems)**: You probably guessed it - this is a combination of the above two types.  The idea here is use what you have if you have it. [Here](http://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2005%20-%20Hybrid%20recommendation%20approaches.pdf) are a few designs related to this that are worth looking into.


Those are the three main types, but there is one additional type that you may find if you are diving a little deeper into the subject material:


- **[Knowledge-Based](https://en.wikipedia.org/wiki/Knowledge-based_recommender_system)**: This is is the most rare type mainly because it requires explicit domain knowledge. It is often used for products that have a low number of available ratings, such as high luxury goods like hypercars. We won't delve any further into this type, but I recommend reading more about it if you're interested in the concept.


### Methodology

Let's return to our problem. The previous way of selecting restaurants at the recommendation of locals and acquaintances (collaborative filtering) wasn't always successful, so we are going to use the idea behind content-based recommender systems to evaluate our options.  However, we don't have a lot of content about the restaurants available, so we are going to primarily use the reviews people left for them.  More specifically, we are going to determine similarity between restaurants based off of the similarity of the reviews that people have written for them.  

We're going to use cosine similarity since it's generally accepted as producing better results in item-to-item filtering:

<img hspace="40" src="http://mathurl.com/ybv72jfa.png">

Before calculating this, we need to perform a couple of pre-processing steps on our reviews in order to make the data  usable for our cosine similarity calculation.  These will be common NLP (**n**atural **l**anguage **p**rocessing) techniques that you should be familiar with if you have worked with text before.  These are the steps I took, but I am open to feedback and improvement if you have recommendations on other methods that may yield better results.

**1) Normalizing**: This step converts our words into lower case so that when we map to our feature space, we don't end up with redundant features for the same words.  For example:

    "Central Texas barbecue is the best smoked and the only barbecue that matters"

becomes


    "central texas barbecue is the best smoked and the only barbecue that matters"


**2) Tokenizing**: This step breaks up a sentence into individual words, essentially turning our reviews into [bags of words](https://en.wikipedia.org/wiki/Bag-of-words_model), which makes it easier to perform other operations.  Though we are going to perform many other preprocessing operations, this is more or less the beginning of mapping our reviews into the feature space.  For example:

    Ex. 'Central Texas barbecue is the best smoked and the only barbecue that matters'

becomes


    ['Central', 'Texas', 'barbecue', 'is', 'the', 'best', 'smoked', 'and', 'the', 'only', 'barbecue', 'that', 'matters']


**3) Removing Stopwords and Punctuation**: This step removes unnecessary words and punctuation often used in language that computers don't need such as *as*, *the*, *and*, and *of*.  For example:

    Ex. ['central', 'texas', 'barbecue', 'is', 'the', 'best', 'smoked', 'and', 'the', 'only', 'barbecue', 'that', 'matters']

becomes


    ['central', 'texas', 'barbecue', 'best', 'smoked', 'only', 'barbecue', 'matters']


**4) Lemmatizing (Stemming)**: Lemmatizing (which is very similar to stemming) removes variations at the end of a word to revert words to their root word.  For example:

    Ex. ['central', 'texas', 'barbecue', 'best', 'smoked', 'only', 'barbecue', 'matters']

becomes


    ['central', 'texas', 'barbecue', 'best', 'smoke', 'only', 'barbecue', 'matter']


**5) Term Frequency-Inverse Document Frequency (TF-IDF)**: This technique determines how important a word is to a document (which is a review in this case) within a corpus (the collection documents, or all reviews).  This doesn't necessarily help establish context within our reviews themselves (for example, 'this Pad Kee Mao is bad ass' is technically a good thing, which wouldn't be accounted for unless we did [n-grams](https://en.wikipedia.org/wiki/N-gram) (which will give my laptop a much more difficult time)), but it does help with establishing the importance of the word.

<img hspace="40" src="http://mathurl.com/ya3r3wkx.png">

<img hspace="40" src="http://mathurl.com/yazeemgy.png">

<img hspace="40" src="http://mathurl.com/y82lttkz.png">

On a side note, sarcasm, slang, misspellings, emoticons, and context are common problems in NLP, but we will be ignoring these due to time limitations.  

### Assumptions

It's always important to state your assumptions in any analysis because a violation of them will often impact the reliability of the results.  My assumptions in this case are as follows:

- The reviews are indicative of the characteristics of the restaurant.
- The language used in the reviews does not directly impact the rating a user gives.
     - E.g. Reviews contain a description of their experience, but ratings are the result of the user applying weights to specific things they value.
         - Ex. "The food was great, but the service was terrible." would be a 2/10 for one user, but a 7/10 for users like myself.
- The restaurants did not undergo significant changes in the time frame for the reviews being pulled.
- Sarcasm, slang, misspellings, and other common NLP problems will not have a significant impact on our results.

---
# Restaurant Recommender


If you're still with us after all of that, let's get started!

---


We'll be using standard libraries for this project (pandas, nltk, and scikit-learn), but one additional thing we need for this project are credentials to access the Foursquare API.  I'm not keen on sharing mine, but you can get your own by [signing up](https://developer.foursquare.com/).  

## The Data

Foursquare works similarly to Yelp where users will review restaurants.  They can either leave a rating (1-10), or write a review for the restaurant.  The reviews are what we're interested in here since I established above that the rating has less meaning due to the way people rate restaurants differently between the two cities.

The [documentation](https://developer.foursquare.com/docs/) was fortunately fairly robust, and you can read about the specific API calls I used in [my github code](https://github.com/JeffMacaluso/Blog/blob/master/Restaurant%20Recommender.ipynb).  I had to perform a few calls in order to get all of the data I needed, but the end result is a data frame with one row for each of the ~1,100 restaurants and a column (comments) that contains all of the reviews:

<div style="overflow-x:auto;">
<style>
table {
        margin-left: auto;
        margin-right: auto;
        border: none;
        border-collapse: collapse;
        border-spacing: 0;
        color: @rendered_html_border_color;
        font-size: 12px;
        table-layout: fixed;
        max-width:800px;
        overflow-x:scroll;
    }
    thead {
        border-bottom: 1px solid @rendered_html_border_color;
        vertical-align: bottom;
    }
    tr, th, td {
        text-align: right;
        vertical-align: middle;
        padding: 0.5em 0.5em;
        line-height: normal;
        white-space: normal;
        max-width: none;
        border: none;
    }
    th {
        font-weight: bold;
    }
    tbody tr:nth-child(odd) {
        background: #f5f5f5;
    }
    tbody tr:hover {
        background: rgba(66, 165, 245, 0.2);
    }
    * + table {margin-top: 1em;}

    p {text-align: left;}
* + p {margin-top: 1em;}

td, th {
    text-align: center;
    padding: 8px;
}
</style>

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>category</th>
      <th>shortCategory</th>
      <th>checkinsCount</th>
      <th>city</th>
      <th>state</th>
      <th>location</th>
      <th>commentsCount</th>
      <th>usersCount</th>
      <th>priceTier</th>
      <th>numRatings</th>
      <th>rating</th>
      <th>comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4e17b348b0fb8567c665ddaf</td>
      <td>Souper Salad</td>
      <td>Salad Place</td>
      <td>Salad</td>
      <td>1769</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>17</td>
      <td>683</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>6.9</td>
      <td>{Healthy fresh salad plus baked potatoes, sele...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4aceefb7f964a52013d220e3</td>
      <td>Aster's Ethiopian Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1463</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>34</td>
      <td>1018</td>
      <td>2.0</td>
      <td>93.0</td>
      <td>8.0</td>
      <td>{The lunch buffet is wonderful; everything is ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4b591015f964a520c17a28e3</td>
      <td>Taste Of Ethiopia</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1047</td>
      <td>Pflugerville</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>31</td>
      <td>672</td>
      <td>2.0</td>
      <td>88.0</td>
      <td>8.3</td>
      <td>{Wonderful! Spicy lovers: Kitfo, was awesome! ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4ead97ba4690615f26a8adfe</td>
      <td>Wasota African Cuisine</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>195</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>12</td>
      <td>140</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>6.2</td>
      <td>{Obsessed with this place. One little nugget (...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4c7efeba2042b1f76cd1c1ad</td>
      <td>Cazamance</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>500</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>11</td>
      <td>435</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>{West African fusion reigns at this darling tr...</td>
    </tr>
  </tbody>
</table>
</div>

I excluded fast food restaurants and chains from my API call since I'm not interested in them, but a few were included due to having a different category assigned to them.  For example, most of the restaurants under the "coffee" category are Starbucks.

Let's look at a few charts for exploratory analysis to get a better idea of our data, starting with the number of reviews per restaurant:

<img src="https://raw.githubusercontent.com/JeffMacaluso/JeffMacaluso.github.io/master/_posts/RestaurantRecommender_files/Restaurant%20Recommender_12_0.png">


One thing to note is that I’m currently limited to 30 comments per restaurant ID since I’m using a free developer key, so this will impact the quality of the analysis to a degree.


Next, let's look at the number of restaurants per city:


<img src="https://raw.githubusercontent.com/JeffMacaluso/JeffMacaluso.github.io/master/_posts/RestaurantRecommender_files/Restaurant%20Recommender_13_0.png">


These appear to be even, so let's look at the breakdown of restaurant categories between the two cities:


<img src="https://raw.githubusercontent.com/JeffMacaluso/JeffMacaluso.github.io/master/_posts/RestaurantRecommender_files/Restaurant%20Recommender_14_0.png">


To summarize this chart:

**Austin:**
- Significantly more BBQ, tacos, food trucks, donuts, juice bars, and Cajun restaurants (but I could have told you this)
- Seemingly more diversity in the smaller categories

**Minneapolis:**
- American is king
- Significantly more bars, bakeries, middle eastern, cafés, tea rooms, German, and breweries


Lastly, here's an example of the comments for one random restaurant to give a better idea of the text that we're dealing with:

```
'{Make sure you check your steaks, they don\'t cook it the way you ask for them. The Outlaw Ribeye is excellent when cooked to order!

Kyle is the best waiter here he should be a trainer or head waiter an train everyone at his level of customer service. I will wait to be seated when he\'s working.

Alex is a great waitress very polite and very conscious about refills and service.  I give her 4.5 stars. Not quite a 5 yet.

Great place awesome staff....but alas....their is a sign banning Cody G. Sangria is awesome!!!

They make good food but if you are taking to go it takes awhile. They say 15 minutes but it\'s like 25-30.  Prepare for that.Lunch specials are great & service is always good.

If you don\'t like lemon juice in your water, make sure to ask for it without.The burgers here are awesome! Freshly ground sirloin! Yummy!Try the Wild West Shrimp. Plenty to share. Yum!!!Customer service here is at 110%. Best service of any longhorns that I have been to across Texas.I was enjoying my salad when I bit into something odd.  It turned out to be a shard of clear hard plastic. I called the manager over. He replied, "glad you didn\'t swallow that". Not even an apology.I ordered a pork chop dinner and they brought me ribs. Talk about disorganized, plus I had to wait an hour for the pork chops.This isn\'t Ruth\'s Chris; understanding that is key to having a good time.The Mula drink it\'s excellentI can\'t believe they don\'t have Dr Pepper here; come on, this Texas, it\'s a requirement!!!The broccoli cheese soup was good!The passion/pineapple vodka is yummy!!!!Excelente lugarBuen servicio.This place is trash, save ur money ! Go to Texas road house.... YeaaaahhhhhhhCody G. must live here.Casey spends a lot of time here.TERRIBLE!!! Mediocre chain food and RUDE SERVICE!!! NEVER GOING BACK!!!Expensive.}'
```

<div style="background-color:rgba(0, 0, 0, 0.0470588); text-align:center; vertical-align: middle; padding:40px 0;">
'{Make sure you check your steaks, they don\'t cook it the way you ask for them. The Outlaw Ribeye is excellent when cooked to order!Kyle is the best waiter here he should be a trainer or head waiter an train everyone at his level of customer service. I will wait to be seated when he\'s working.Alex is a great waitress very polite and very conscious about refills and service.  I give her 4.5 stars. Not quite a 5 yet.Great place awesome staff....but alas....their is a sign banning Cody G. Sangria is awesome!!!They make good food but if you are taking to go it takes awhile. They say 15 minutes but it\'s like 25-30.  Prepare for that.Lunch specials are great and service is always good.If you don\'t like lemon juice in your water, make sure to ask for it without.The burgers here are awesome! Freshly ground sirloin! Yummy!Try the Wild West Shrimp. Plenty to share. Yum!!!Customer service here is at 110%. Best service of any longhorns that I have been to across Texas.I was enjoying my salad when I bit into something odd.  It turned out to be a shard of clear hard plastic. I called the manager over. He replied, "glad you didn\'t swallow that". Not even an apology.I ordered a pork chop dinner and they brought me ribs. Talk about disorganized, plus I had to wait an hour for the pork chops.This isn\'t Ruth\'s Chris; understanding that is key to having a good time.The Mula drink it\'s excellentI can\'t believe they don\'t have Dr Pepper here; come on, this Texas, it\'s a requirement!!!The broccoli cheese soup was good!The passion/pineapple vodka is yummy!!!!Excelente lugarBuen servicio.This place is trash, save ur money ! Go to Texas road house.... YeaaaahhhhhhhCody G. must live here.Casey spends a lot of time here.TERRIBLE!!! Mediocre chain food and RUDE SERVICE!!! NEVER GOING BACK!!!Expensive.}'
</div>

## Data Processing

When working with language, we have to process the text into something that a computer can handle more easily.  Our end result will be a large number of numerical features for each restaurant that we can use to calculate the cosine similarity.

The steps here are:
1. Normalizing
2. Tokenizing
3. Removing stopwords
4. Lemmatizing (Stemming)
5. Term Frequency-Inverse Document Frequency (TF-IDF)

I'll explain a little more on what these are and why we are doing them below in case you aren't familiar with them.

### 1) Normalizing

This section uses regex scripts that makes cases every word lower cased, removes punctuation, and removes digits.

For example:

**Before:**

    "ThIs Is HoW mIdDlE sChOoLeRs TaLkEd 2 EaCh OtHeR oN AIM!!!!"

**After:**

    "this is how middle schoolers talked each other on aim"

The benefit in this is that it vastly reduces our feature space.  Our pre-processed example would have created an additional ~10 features from someone who doesn't know how to type like a regular human being.


```python
%%time

# Converting all words to lower case and removing punctuation
df['comments'] = [re.sub(r'\d+\S*', '',
                  row.lower().replace('.', ' ').replace('_', '').replace('/', ''))
                  for row in df['comments']]

df['comments'] = [re.sub(r'(?:^| )\w(?:$| )', '', row)
                  for row in df['comments']]

# Removing numbers
df['comments'] = [re.sub(r'\d+', '', row) for row in df['comments']]

df['comments'].head()
```




    0    {healthy fresh salad plus baked potatoes, sele...
    1    {the lunch buffet is wonderful; everything is ...
    2    {wonderful! spicy lovers: kitfo, was awesome! ...
    3    {obsessed with this place  one little nugget (...
    4    {west african fusion reigns at this darling tr...
    Name: comments, dtype: object

    Wall time: 971 ms



### 2) Tokenizing

Tokenizing a sentence is a way to map our words into a feature space.  This is achieved by treating every word as an individual object.

**Before:**

    'central texas barbecue is the best smoked and the only barbecue that matters'

**After:**

    ['central', 'texas', 'barbecue', 'is', 'the', 'best', 'smoked', 'and', 'the', 'only', 'barbecue', 'that', 'matters']



```python
%%time

# Tokenizing comments and putting them into a new column
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')  # by blank space
df['tokens'] = df['comments'].apply(tokenizer.tokenize)

df['tokens'].head()
```




    0    [healthy, fresh, salad, plus, baked, potatoes,...
    1    [the, lunch, buffet, is, wonderful, everything...
    2    [wonderful, spicy, lovers, kitfo, was, awesome...
    3    [obsessed, with, this, place, one, little, nug...
    4    [west, african, fusion, reigns, at, this, darl...
    Name: tokens, dtype: object

    Wall time: 718ms

### 3) Removing Stopwords  & Punctuation

Stopwords are unnecessary words like *as*, *the*, *and*, and *of* that aren't very useful for our purposes.  Since they don't have any intrinsic value, removing them reduces our feature space which will speed up our computations.

**Before:**

    ['central', 'texas', 'barbecue', 'is', 'the', 'best', 'smoked', 'and', 'the', 'only', 'barbecue', 'that', 'matters']

**After:**

    ['central', 'texas', 'barbecue', 'best', 'smoked', 'only', 'barbecue', 'matters']

This does take a bit longer to run at ~6 minutes


```python
%%time

filtered_words = []
for row in df['tokens']:
    filtered_words.append([
        word.lower() for word in row
        if word.lower() not in nltk.corpus.stopwords.words('english')
    ])

df['tokens'] = filtered_words
```

    Wall time: 4min 59s
    

### 4) Lemmatizing (Stemming)

Stemming removes variations at the end of a word to revert words to their root in order to reduce our overall feature space (e.x. running $\rightarrow$ run).  This has the possibility to adversely impact our performance when the root word is different (e.x. university $\rightarrow$ universe), but the net positives typically outweigh the net negatives.

**Before:**

    ['central', 'texas', 'barbecue', 'best', 'smoked', 'only', 'barbecue', 'matters']

**After:**

    ['central', 'texas', 'barbecue', 'best', 'smoke', 'only', 'barbecue', 'matter']

One very important thing to note here is that we're actually doing something called **[Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)**, which is similar to [stemming](https://en.wikipedia.org/wiki/Stemming), but is a little different. Both seek to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form, but they go about it in different ways.  In order to illustrate the difference, here's a dictionary entry:

<img src="http://college.cengage.com/english/raimes/keys_writers/5e/assets/students/images/annotated_dictionary_html_5c444ce8.png">

Lemmatization seeks to get the *lemma*, or the base dictionary form of the word.  In our example above, that would be "graduate".  It does this by using vocabulary and a morphological analysis of the words, rather than just chopping off the variations (the "verb forms" in the example above) like a traditional stemmer would.

The advantage of lemmatization here is that we don't run into issues like our other example of *university* $\rightarrow$ *universe* that can happen in conventional stemmers.  It is also relatively quick on this data set!

The disadvantage is that it is not able to infer if the word is a noun/verb/adjective/etc., so we have to specify which type it is.  Since we're looking at, well, everything, we're going to lemmatize for nouns, verbs, and adjectives.

[Here](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) is an excerpt from the Stanford book *Introduction to Information Retrieval* if you wanted to read more stemming and lemmatization.


```python
%%time

# Setting the Lemmatization object
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()

# Looping through the words and appending the lemmatized version to a list
stemmed_words = []
for row in df['tokens']:
    stemmed_words.append([
        # Verbs
        lmtzr.lemmatize(  
            # Adjectives
            lmtzr.lemmatize(  
                # Nouns
                lmtzr.lemmatize(word.lower()), 'a'), 'v')
        for word in row
        if word.lower() not in nltk.corpus.stopwords.words('english')])

# Adding the list as a column in the data frame
df['tokens'] = stemmed_words
```

    Wall time: 4min 52s
    

Let's take a look at how many unique words we now have and a few of the examples:


```python
# Appends all words to a list in order to find the unique words
allWords = []
for row in stemmed_words:
    for word in row:
        allWords.append(str(word))
            
uniqueWords = np.unique(allWords)

print('Number of unique words:', len(uniqueWords), '\n')
print('Previewing sample of unique words:\n', uniqueWords[1234:1244])
```

    Number of unique words: 36592 
    
    Previewing sample of unique words:
     ['andhitachino' 'andhome' 'andhomemade' 'andhopping' 'andhot' 'andhouse'
     'andhuge' 'andiamo' 'andimmediatly' 'andis']
    

We can see a few of the challenges from slang or typos that I mentioned in the beginning.  These will pose problems for what we're doing, but we'll just have to assume that the vast majority of words are spelled correctly.

Before doing the TF-IDF transformation, we need to make sure that we have spaces in between each word in the comments:


```python
stemmed_sentences = []

# Spacing out the words in the reviews for each restaurant
for row in df['tokens']:
    stemmed_string = ''
    for word in row:
        stemmed_string = stemmed_string + ' ' + word
    stemmed_sentences.append(stemmed_string)
    
df['tokens'] = stemmed_sentences
stemmed_sentences[np.random.randint(len(stemmed_sentences))]
```




    ' line ridiculous download app order get bowl min whilst person take place line wait wait hangry townsuper nice staff give extra chicken cost waswink speedy service even onsaturday order fajita sure let know mean pepper bean skimpy portion compare location especially compare portion employee give front pay customer try almost chipotle area one make best steak service really shitty everyone else say third row line least min getbowl andbunch chip best chip dip ever taste use app order advance get comical long line location staff oblivious tell cashier reguerilla getfree bag chip slow chipolte ever seriously come rush anywhere chance go late end withoutdinner get salad dress tangy place crazy busy matter go staff slow expect wait putlime wedge coke good linesridiculous worth wait chicken burrito bowl bestgetbowl healthy great burrito chipsits nice charge guacamolesteak soft taco withside guacamole mexican cuisineorder ahead online great staff line way behind fillingfax order good customer servicei desire earn back mayor ship quality food service takenserious turn since mgr ben sexe leave slow chipotle staff twin citiesthis place go hill lot staff fast mayor unhappyborrito bowlgreat place disfunctional officially bad chipotle go lyndale good service slow chipotle land prepare wait line always terrible management seem carekicks qdoba buttway toooooo slow tonight outta chip guaq areno brainerdamn soooo slow food good service suck'



### 5) Term Frequency-Inverse Document Frequency (TF-IDF)

This determines how important a word is to a document (which is a review in this case) within a corpus (the collection documents). It is a number resulting from the following formula: 

<img hspace="40" src="http://mathurl.com/ya3r3wkx.png">

<img hspace="40" src="http://mathurl.com/yazeemgy.png">

<img hspace="40" src="http://mathurl.com/y82lttkz.png">

Scikit-learn has an [excellent function](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) that is able to transform our processed text into a TF-IDF matrix very quickly.  We'll convert it back to a data frame, and join it to our original data frame by the indexes.


```python
%%time

# Creating the sklearn object
tfidf = sktext.TfidfVectorizer(smooth_idf=False)

# Transforming our 'tokens' column into a TF-IDF matrix and then a data frame
tfidf_df = pd.DataFrame(tfidf.fit_transform(df['tokens']).toarray(), 
                        columns=tfidf.get_feature_names())
```

    Wall time: 1.47 s
    


```python
print(tfidf_df.shape)
tfidf_df.head()
```

    (1335, 36571)
    




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aa</th>
      <th>aaa</th>
      <th>aaaaaaa</th>
      <th>aaaaaaamazing</th>
      <th>aaaaaamaaaaaaazing</th>
      <th>aaaaalll</th>
      <th>aaaaand</th>
      <th>aaaammaazzing</th>
      <th>aaallllllll</th>
      <th>aaarrrggghhh</th>
      <th>...</th>
      <th>zucca</th>
      <th>zucchini</th>
      <th>zuccini</th>
      <th>zuccotto</th>
      <th>zuchini</th>
      <th>zuke</th>
      <th>zuppa</th>
      <th>zur</th>
      <th>zushi</th>
      <th>zzzzzzzzzzzzoops</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36571 columns</p>
</div>



Since we transformed *all* of the words, we have a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix).  We don't care about things like typos or words specific to one particular restaurant, so we're going to remove columns that don't have a lot of contents.


```python
# Removing sparse columns
tfidf_df = tfidf_df[tfidf_df.columns[tfidf_df.sum() > 2.5]]

# Removing any remaining digits
tfidf_df = tfidf_df.filter(regex=r'^((?!\d).)*$')

print(tfidf_df.shape)
tfidf_df.head()
```

    (1335, 976)
    




<div style="overflow-x:auto;">

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>absolute</th>
      <th>absolutely</th>
      <th>across</th>
      <th>actually</th>
      <th>add</th>
      <th>afternoon</th>
      <th>ahead</th>
      <th>ahi</th>
      <th>al</th>
      <th>ale</th>
      <th>...</th>
      <th>wow</th>
      <th>wrap</th>
      <th>wrong</th>
      <th>www</th>
      <th>year</th>
      <th>yes</th>
      <th>yet</th>
      <th>york</th>
      <th>yum</th>
      <th>yummy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.074605</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.034778</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.024101</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023301</td>
      <td>0.023143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.031268</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0376</td>
      <td>0.000000</td>
      <td>0.026175</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025135</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.060402</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.065236</td>
      <td>0.085197</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 976 columns</p>
</div>



This drastically reduced the dimensions of our data set, and we now have something usable to calculate similarity.


```python
# Storing the original data frame before the merge in case any changes are needed
df_orig = df.copy()

# Renaming columns that conflict with column names in tfidfCore
df.rename(columns={'name': 'Name', 
                   'city': 'City', 
                   'location': 'Location'}, inplace=True)

# Merging the data frames by index
df = pd.merge(df, tfidf_df, how='inner', left_index=True, right_index=True)

df.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
      <th>category</th>
      <th>shortCategory</th>
      <th>checkinsCount</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>commentsCount</th>
      <th>usersCount</th>
      <th>...</th>
      <th>wow</th>
      <th>wrap</th>
      <th>wrong</th>
      <th>www</th>
      <th>year</th>
      <th>yes</th>
      <th>yet</th>
      <th>york</th>
      <th>yum</th>
      <th>yummy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4e17b348b0fb8567c665ddaf</td>
      <td>Souper Salad</td>
      <td>Salad Place</td>
      <td>Salad</td>
      <td>1769</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>17</td>
      <td>683</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.074605</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4aceefb7f964a52013d220e3</td>
      <td>Aster's Ethiopian Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1463</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>34</td>
      <td>1018</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.024101</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023301</td>
      <td>0.023143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4b591015f964a520c17a28e3</td>
      <td>Taste Of Ethiopia</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1047</td>
      <td>Pflugerville</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>31</td>
      <td>672</td>
      <td>...</td>
      <td>0.0376</td>
      <td>0.000000</td>
      <td>0.026175</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025135</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4ead97ba4690615f26a8adfe</td>
      <td>Wasota African Cuisine</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>195</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>12</td>
      <td>140</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4c7efeba2042b1f76cd1c1ad</td>
      <td>Cazamance</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>500</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>11</td>
      <td>435</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0.060402</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.065236</td>
      <td>0.085197</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 991 columns</p>
</div>



Lastly, we're going to add additional features for the category.  This just puts a heavier weight on those with the same type, so for example a Mexican restaurant will be more likely to have Mexican restaurants show up as most similar instead of Brazilian restaurants.


```python
# Creates dummy variables out of the restaurant category
df = pd.concat([df, pd.get_dummies(df['shortCategory'])], axis=1)

df.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
      <th>category</th>
      <th>shortCategory</th>
      <th>checkinsCount</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>commentsCount</th>
      <th>usersCount</th>
      <th>...</th>
      <th>Tapas</th>
      <th>Tea Room</th>
      <th>Tex-Mex</th>
      <th>Thai</th>
      <th>Theme Restaurant</th>
      <th>Turkish</th>
      <th>Vegetarian / Vegan</th>
      <th>Vietnamese</th>
      <th>Wine Bar</th>
      <th>Yogurt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4e17b348b0fb8567c665ddaf</td>
      <td>Souper Salad</td>
      <td>Salad Place</td>
      <td>Salad</td>
      <td>1769</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>17</td>
      <td>683</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4aceefb7f964a52013d220e3</td>
      <td>Aster's Ethiopian Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1463</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>34</td>
      <td>1018</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4b591015f964a520c17a28e3</td>
      <td>Taste Of Ethiopia</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1047</td>
      <td>Pflugerville</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>31</td>
      <td>672</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4ead97ba4690615f26a8adfe</td>
      <td>Wasota African Cuisine</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>195</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>12</td>
      <td>140</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4c7efeba2042b1f76cd1c1ad</td>
      <td>Cazamance</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>500</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>11</td>
      <td>435</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1090 columns</p>
</div>



Because we introduced an additional type of feature, we'll have to check it's weight in comparison to the TF-IDF features:


```python
# Summary stats of TF-IDF
print('Max:', np.max(tfidf_df.max()), '\n',
      'Mean:', np.mean(tfidf_df.mean()), '\n',
      'Standard Deviation:', np.std(tfidf_df.std()))
```

    Max: 0.889350911658 
     Mean: 0.005518928565159508 
     Standard Deviation: 0.010229821082730768
    

The dummy variables for the restaurant type are quite a bit higher than the average word, but I'm comfortable with this since I think it has a benefit.

# "Recommender System"

As a reminder, we are not using a conventional recommender system.  Instead, we are using recommender system theory by calculating the cosine distance between comments in order to find restaurants with the most similar comments.  

### Loading in personal ratings

In order to recommend restaurants with this approach, we have to identify the restaurants to which we want to find the most similarities. I took the data frame and assigned my own ratings to some of my favorites.


```python
# Loading in self-ratings for restaurants in the data set
selfRatings = pd.read_csv('selfRatings.csv', usecols=[0, 4])
selfRatings.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43c968a2f964a5209c2d1fe3</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>574481f8498e2cd16a0911a6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4cb5e045e262b60c46cb6ae0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49be75ccf964a520ad541fe3</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4d8d295fc1b1721e798b1246</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging into df to add the column 'selfRating'
df = pd.merge(df, selfRatings)

df.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Name</th>
      <th>category</th>
      <th>shortCategory</th>
      <th>checkinsCount</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>commentsCount</th>
      <th>usersCount</th>
      <th>...</th>
      <th>Tea Room</th>
      <th>Tex-Mex</th>
      <th>Thai</th>
      <th>Theme Restaurant</th>
      <th>Turkish</th>
      <th>Vegetarian / Vegan</th>
      <th>Vietnamese</th>
      <th>Wine Bar</th>
      <th>Yogurt</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4e17b348b0fb8567c665ddaf</td>
      <td>Souper Salad</td>
      <td>Salad Place</td>
      <td>Salad</td>
      <td>1769</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>17</td>
      <td>683</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4aceefb7f964a52013d220e3</td>
      <td>Aster's Ethiopian Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1463</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>34</td>
      <td>1018</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4b591015f964a520c17a28e3</td>
      <td>Taste Of Ethiopia</td>
      <td>Ethiopian Restaurant</td>
      <td>Ethiopian</td>
      <td>1047</td>
      <td>Pflugerville</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>31</td>
      <td>672</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4ead97ba4690615f26a8adfe</td>
      <td>Wasota African Cuisine</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>195</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>12</td>
      <td>140</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4c7efeba2042b1f76cd1c1ad</td>
      <td>Cazamance</td>
      <td>African Restaurant</td>
      <td>African</td>
      <td>500</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>11</td>
      <td>435</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1091 columns</p>
</div>



### Additional features & min-max scaling 

We're going to include a few additional features from the original data set to capture information that the comments may not have.  Specifically:

- **Popularity:** checkinsCount, commentsCount, usersCount, numRatings
- **Price:** priceTier

We're also going to scale these down so they don't carry a huge advantage over everything else.  I'm going to scale the popularity attributes to be between 0 and 0.5, and the price attribute to be between 0 and 1.  I'll do this by first min-max scaling everything (to put it between 0 and 1), and then dividing the popularity features in half.


```python
# Removing everything that won't be used in the similarity calculation
df_item = df.drop(['id', 'category', 'Name', 'shortCategory', 'City', 'tokens',
                   'comments', 'state', 'Location', 'selfRating', 'rating'],
                  axis=1)

# Copying into a separate data frame to be normalized
df_item_norm = df_item.copy()

columns_to_scale = ['checkinsCount', 'commentsCount',
                    'usersCount', 'priceTier', 'numRatings']

# Split
df_item_split = df_item[columns_to_scale]
df_item_norm.drop(columns_to_scale, axis=1, inplace=True)

# Apply
df_item_split = pd.DataFrame(MinMaxScaler().fit_transform(df_item_split),
                             columns=df_item_split.columns)
df_item_split_half = df_item_split.drop('priceTier', axis=1)
df_item_split_half = df_item_split_half / 2
df_item_split_half['priceTier'] = df_item_split['priceTier']

# Combine
df_item_norm = df_item_norm.merge(df_item_split,
                                  left_index=True, right_index=True)

df_item_norm.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>absolute</th>
      <th>absolutely</th>
      <th>across</th>
      <th>actually</th>
      <th>add</th>
      <th>afternoon</th>
      <th>ahead</th>
      <th>ahi</th>
      <th>al</th>
      <th>ale</th>
      <th>...</th>
      <th>Turkish</th>
      <th>Vegetarian / Vegan</th>
      <th>Vietnamese</th>
      <th>Wine Bar</th>
      <th>Yogurt</th>
      <th>checkinsCount</th>
      <th>commentsCount</th>
      <th>usersCount</th>
      <th>priceTier</th>
      <th>numRatings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.030682</td>
      <td>0.017544</td>
      <td>0.020429</td>
      <td>0.000000</td>
      <td>0.017494</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.034778</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.024844</td>
      <td>0.060150</td>
      <td>0.031648</td>
      <td>0.333333</td>
      <td>0.046840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.031268</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.016906</td>
      <td>0.052632</td>
      <td>0.020060</td>
      <td>0.333333</td>
      <td>0.044018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000649</td>
      <td>0.005013</td>
      <td>0.002244</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.006468</td>
      <td>0.002506</td>
      <td>0.012123</td>
      <td>0.000000</td>
      <td>0.002822</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1080 columns</p>
</div>



### Calculating cosine similarities

Here's the moment that we've spent all of this time getting to: the similarity.  

This section calculates the cosine similarity and puts it into a matrix with the pairwise similarity:

|      | 0    | 1    | ...  | n    |
|------|------|------|------|------|
|   0  | 1.00 | 0.03 | ...  | 0.15 |
|   1  | 0.31 | 1.00 | ...  | 0.89 |
| ...  | ...  | ...  | ...  | ...  |
| n    | 0.05 | 0.13 | ...  | 1.00 |

As a reminder, we're using cosine similarity because it's generally accepted as producing better results in item-to-item filtering. For all you math folk, here's the formula again:

<img hspace="40" src="http://mathurl.com/ybv72jfa.png">


```python
# Calculating cosine similarity
df_item_norm_sparse = sparse.csr_matrix(df_item_norm)
similarities = cosine_similarity(df_item_norm_sparse)

# Putting into a data frame
dfCos = pd.DataFrame(similarities)

dfCos.head()
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1113</th>
      <th>1114</th>
      <th>1115</th>
      <th>1116</th>
      <th>1117</th>
      <th>1118</th>
      <th>1119</th>
      <th>1120</th>
      <th>1121</th>
      <th>1122</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.056252</td>
      <td>0.067231</td>
      <td>0.026230</td>
      <td>0.024996</td>
      <td>0.059179</td>
      <td>0.051503</td>
      <td>0.067922</td>
      <td>0.079887</td>
      <td>0.073811</td>
      <td>...</td>
      <td>0.019677</td>
      <td>0.051769</td>
      <td>0.066944</td>
      <td>0.047500</td>
      <td>0.055441</td>
      <td>0.059609</td>
      <td>0.032087</td>
      <td>0.045675</td>
      <td>0.064043</td>
      <td>0.047003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.056252</td>
      <td>1.000000</td>
      <td>0.881640</td>
      <td>0.112950</td>
      <td>0.036702</td>
      <td>0.164534</td>
      <td>0.195639</td>
      <td>0.144018</td>
      <td>0.158320</td>
      <td>0.150945</td>
      <td>...</td>
      <td>0.083064</td>
      <td>0.125948</td>
      <td>0.144452</td>
      <td>0.089293</td>
      <td>0.128221</td>
      <td>0.078654</td>
      <td>0.113436</td>
      <td>0.129943</td>
      <td>0.146670</td>
      <td>0.128873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.067231</td>
      <td>0.881640</td>
      <td>1.000000</td>
      <td>0.118254</td>
      <td>0.045115</td>
      <td>0.139096</td>
      <td>0.179490</td>
      <td>0.132280</td>
      <td>0.153244</td>
      <td>0.145920</td>
      <td>...</td>
      <td>0.094744</td>
      <td>0.120079</td>
      <td>0.145357</td>
      <td>0.083324</td>
      <td>0.126703</td>
      <td>0.082119</td>
      <td>0.102128</td>
      <td>0.122537</td>
      <td>0.151989</td>
      <td>0.132252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.026230</td>
      <td>0.112950</td>
      <td>0.118254</td>
      <td>1.000000</td>
      <td>0.767280</td>
      <td>0.078028</td>
      <td>0.151436</td>
      <td>0.090713</td>
      <td>0.086704</td>
      <td>0.102522</td>
      <td>...</td>
      <td>0.094147</td>
      <td>0.071328</td>
      <td>0.103130</td>
      <td>0.080002</td>
      <td>0.100153</td>
      <td>0.049730</td>
      <td>0.093961</td>
      <td>0.111084</td>
      <td>0.109474</td>
      <td>0.096562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.024996</td>
      <td>0.036702</td>
      <td>0.045115</td>
      <td>0.767280</td>
      <td>1.000000</td>
      <td>0.032617</td>
      <td>0.030823</td>
      <td>0.030906</td>
      <td>0.041741</td>
      <td>0.031845</td>
      <td>...</td>
      <td>0.006193</td>
      <td>0.026284</td>
      <td>0.038349</td>
      <td>0.042255</td>
      <td>0.046679</td>
      <td>0.038977</td>
      <td>0.011920</td>
      <td>0.027661</td>
      <td>0.044796</td>
      <td>0.021876</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1123 columns</p>
</div>



These are the some of the restaurants I rated very highly, and I'm pulling these up so we can use the index number in order to compare it to the others in our data set:


```python
# Filtering to those from my list with the highest ratings
topRated = df[df['selfRating'] >= 8].drop_duplicates('Name')

# Preparing for display
topRated[['Name', 'category', 'Location', 'selfRating']].sort_values(
    'selfRating', ascending=False)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>category</th>
      <th>Location</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Jack Allen's Kitchen</td>
      <td>American Restaurant</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Black's Barbecue</td>
      <td>BBQ Joint</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>447</th>
      <td>Cabo Bob's</td>
      <td>Burrito Place</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>420</th>
      <td>Tacodeli</td>
      <td>Taco Place</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Round Rock Donuts</td>
      <td>Donut Shop</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Jack Allens on Capital of TX</td>
      <td>American Restaurant</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Blue Dahlia Bistro</td>
      <td>Café</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>967</th>
      <td>Chimborazo</td>
      <td>Latin American Restaurant</td>
      <td>Minneapolis, MN</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Black's BBQ, The Original</td>
      <td>BBQ Joint</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>66</th>
      <td>The Salt Lick</td>
      <td>BBQ Joint</td>
      <td>Austin, TX</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Torchy's Tacos</td>
      <td>Taco Place</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Clay Pit Contemporary Indian Cuisine</td>
      <td>Indian Restaurant</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>637</th>
      <td>Brasa Premium Rotisserie</td>
      <td>BBQ Joint</td>
      <td>Minneapolis, MN</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>576</th>
      <td>Afro Deli</td>
      <td>African Restaurant</td>
      <td>Minneapolis, MN</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>Trudy's Texas Star</td>
      <td>Mexican Restaurant</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>426</th>
      <td>Trudy's North Star</td>
      <td>Mexican Restaurant</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Chuy's</td>
      <td>Mexican Restaurant</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>360</th>
      <td>Mandola's Italian Market</td>
      <td>Italian Restaurant</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>La Barbecue Cuisine Texicana</td>
      <td>BBQ Joint</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Franklin Barbecue</td>
      <td>BBQ Joint</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Mighty Fine Burgers</td>
      <td>Burger Joint</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>151</th>
      <td>Hopdoddy Burger Bar</td>
      <td>Burger Joint</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>171</th>
      <td>P. Terry's Burger Stand</td>
      <td>Burger Joint</td>
      <td>Austin, TX</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Juan in a Million</td>
      <td>Mexican Restaurant</td>
      <td>Austin, TX</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Mozart's Coffee</td>
      <td>Coffee Shop</td>
      <td>Austin, TX</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Uchi</td>
      <td>Japanese Restaurant</td>
      <td>Austin, TX</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>558</th>
      <td>The Coffee Bean &amp; Tea Leaf</td>
      <td>Coffee Shop</td>
      <td>Austin, TX</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Texas Honey Ham Company</td>
      <td>Sandwich Place</td>
      <td>Austin, TX</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>832</th>
      <td>Kramarczuk's East European Deli</td>
      <td>Eastern European Restaurant</td>
      <td>Minneapolis, MN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



In order to speed things up, we'll make a function that formats the cosine similarity data frame and retrieves the top n most similar restaurants for the given restaurant:


```python
def retrieve_recommendations(restaurant_index, num_recommendations=5):
    """
    Retrieves the most similar restaurants for the index of a given restaurant 
    
    Outputs a data frame showing similarity, name, location, category, and rating
    """
    # Formatting the cosine similarity data frame for merging
    similarity = pd.melt(dfCos[dfCos.index == restaurant_index])
    similarity.columns = (['restIndex', 'cosineSimilarity'])

    # Merging the cosine similarity data frame to the original data frame
    similarity = similarity.merge(
        df[['Name', 'City', 'state', 'Location',
            'category', 'rating', 'selfRating']],
        left_on=similarity['restIndex'],
        right_index=True)
    similarity.drop(['restIndex'], axis=1, inplace=True)

    # Ensuring that retrieved recommendations are for Minneapolis
    similarity = similarity[(similarity['Location'] == 'Minneapolis, MN') | (
        similarity.index == restaurant_index)]

    # Sorting by similarity
    similarity = similarity.sort_values(
        'cosineSimilarity', ascending=False)[:num_recommendations + 1]
    
    return similarity
```

Alright, let's test it out!

### Barbecue

Let's start with the [Salt Lick](http://saltlickbbq.com/).  This is a popular central Texas barbecue place featured on [various food shows](https://www.youtube.com/watch?v=vLnsXechOWc).  They are well-known for their open smoke pit:

<img style="width: 700px;" src="http://cdn.loc.gov/service/pnp/highsm/28500/28571v.jpg">

In case you're not familiar with [central Texas barbecue](https://en.wikipedia.org/wiki/Barbecue_in_Texas), it primarily features smoked meats (especially brisket) with white bread, onions, pickles, potato salad, beans and cornbread on the side.  Sweet tea is usually the drink of choice if you're not drinking a Shiner or a Lonestar.


```python
# Salt Lick
retrieve_recommendations(66)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>1.000000</td>
      <td>The Salt Lick</td>
      <td>Driftwood</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>BBQ Joint</td>
      <td>9.5</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>637</th>
      <td>0.696079</td>
      <td>Brasa Premium Rotisserie</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>BBQ Joint</td>
      <td>9.3</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>646</th>
      <td>0.604501</td>
      <td>Famous Dave's</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>BBQ Joint</td>
      <td>7.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>0.590742</td>
      <td>Psycho Suzi's Motor Lounge &amp; Tiki Garden</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Theme Restaurant</td>
      <td>8.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>654</th>
      <td>0.572947</td>
      <td>Rack Shack BBQ</td>
      <td>Burnsville</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>BBQ Joint</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>838</th>
      <td>0.567213</td>
      <td>Brit's Pub &amp; Eating Establishment</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>English Restaurant</td>
      <td>8.8</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Surprisingly, our top recommendation is one of my favorite restaurants I've found in Minneapolis - Brasa!  They're actually a [Creole](https://en.wikipedia.org/wiki/Louisiana_Creole_cuisine) restaurant, but they have a lot of smoked meats, beans, and corn bread, and they're probably the only restaurant I've found so far that lists sweet tea on the menu:

<img src="https://getbento.imgix.net/accounts/0430ef0cdee9c5699c71ec5ebdb2541c/media/images/press_thumbnail54270.jpeg?w=600&fit=max&auto=compress,format&h=600">

Funny enough, Brasa was also in [Man vs Food](https://www.youtube.com/watch?v=gZmGAi5DKE4) with Andrew Zimmerman as a guest.

Famous Dave's is a Midwestern barbecue chain that focuses more on ribs, which isn't generally considered a Texan specialty.  Psycho Suzi's (a theme restaurant that serves pizza and cocktails) and Brit's Pub (an English pub with a lawn bowling field) don't seem very similar, but their cosine similarity scores reflect that.

### Donuts

Moving on, let's find some donuts.  Before maple-bacon-cereal-whatever donuts become the craze (thanks for nothing, Portland), my home town was also famous for [Round Rock Donuts](http://roundrockdonuts.com/), a simple and delicious no-nonsense donut shop.  And yes, Man vs. Food also did a segment here.

<img src="http://www.fronteraridge.com/wp-content/uploads/2016/10/8092563177_48b5681eb7.jpg">


```python
# Round Rock Donuts
retrieve_recommendations(222)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>222</th>
      <td>1.000000</td>
      <td>Summer Moon Wood-Fired Coffee</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>Coffee Shop</td>
      <td>8.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>749</th>
      <td>0.865335</td>
      <td>Spyhouse Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>8.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>742</th>
      <td>0.857070</td>
      <td>Five Watt Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>8.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>745</th>
      <td>0.848061</td>
      <td>Spyhouse Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>8.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>747</th>
      <td>0.843742</td>
      <td>Dunn Bros Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>7.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>740</th>
      <td>0.834707</td>
      <td>Spyhouse Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>9.1</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



Sadly, a lot of the most similar places our results returned were places I've tried and didn't like.  For some reason, the donuts at most places up here are usually cold, cake-based, and covered in kitschy stuff like bacon.  However, Granny donuts looks like it could be promising, as does Bogart's:

<img style="width: 400px;" src="https://s3-media3.fl.yelpcdn.com/bphoto/VVpoCxOqusvCGpr5zAW-ZQ/o.jpg">

### Tacos

This is another Austin specialty that likely won't give promising results, but let's try it anyway.

[Tacodeli](http://www.tacodeli.com/) is my personal favorite taco place in Austin (yes, it's better than [Torchy's](http://torchystacos.com/)), and they're a little different than the traditional taco (corn tortilla, meat, onions, cilantro, and lime that you might find at traditional Mexican taquerias).  They're typically on flour tortillas, and diversify their flavor profiles and toppings:

<img style="width: 600px;" src="https://cdn.vox-cdn.com/thumbor/TiugzPrR2vKWCPitTTV1oo9a1iE=/114x0:1934x1365/1200x900/filters:focal(114x0:1934x1365)/cdn.vox-cdn.com/uploads/chorus_image/image/53151105/16112919_10154883686894449_6136333774221690944_o.0.0.jpg">


```python
# Tacodeli
retrieve_recommendations(420)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>420</th>
      <td>1.000000</td>
      <td>Tacodeli</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>Taco Place</td>
      <td>9.2</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>986</th>
      <td>0.869268</td>
      <td>Rusty Taco</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Taco Place</td>
      <td>8.1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>849</th>
      <td>0.826003</td>
      <td>Taco Taxi</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Taco Place</td>
      <td>8.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>0.270148</td>
      <td>Psycho Suzi's Motor Lounge &amp; Tiki Garden</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Theme Restaurant</td>
      <td>8.5</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>579</th>
      <td>0.259052</td>
      <td>Hell's Kitchen</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>American Restaurant</td>
      <td>8.9</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>838</th>
      <td>0.256581</td>
      <td>Brit's Pub &amp; Eating Establishment</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>English Restaurant</td>
      <td>8.8</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It looks like there's a pretty sharp drop-off in cosine similarity after our second recommendation (which makes sense when you look at the ratio of taco places in Austin vs. Minneapolis from when we pulled our data), so I'm going to discard the bottom three.  I'm surprised again that Psycho Suzi's and Brit's Pub made a second appearance since neither of them serve tacos, but I won't into that too much since their cosine similarity is really low.

I have tried Rusty Taco, and it does seem a lot like Tacodeli. They even sell breakfast tacos, which is a very Texan thing that can be rare in the rest of the country. The primary difference is in the diversity and freshness of ingredients, and subsequently for me, taste:

<img src="http://heavytable.com/wp-content/uploads/2011/05/rusty-taco_6-x-tacos.jpg">

Taco Taxi looks like it could be promising, but they appear to be more of a traditional taqueria (delicious but dissimilar). To be fair, taquerias have had the best tacos I've found up here (though most of them aren't included in this list because they were outside of the search range).

### Burritos

I'm not actually going to run our similarity function for this part because the burrito place back home actually disappeared from our data pulling query in between me originally running this and finally having time to annotate everything and write this post. However, I wanted to include it because it was one of my other field tests.

[Cabo Bob's](http://cabobobs.com/) is my favorite burrito place back home, and they made it to the semi-finals in the [538 best burrito in America search](https://fivethirtyeight.com/burrito/#brackets-view) losing to the overall champion.  To anyone not familiar with non-chain burrito restaurants, they are similar to Chipotle, but are usually higher quality.

<img style="width: 800px;" src="http://5a41cb68e4f4b5f1172e-bf5f2072334b80bdf7b0bc4cd64c7593.r11.cf2.rackcdn.com/16c3f0f16a61fb6aeb081d8c0dd99467-b41587790ca364353f5a4b46d585058b.jpg">

[El Burrito Mercado](http://elburritomercado.com/) returned as highly similar, so we tried it.  It's tucked in the back of a mercado, and has both a sit-down section as well as a lunch line similar to Cabo Bob's.  We decided to go for the sit-down section since we had come from the opposite side of the metropolitan area, so the experience was a little different.  My burrito was more of a traditional Mexican burrito (as opposed to Tex-Mex), but it was still pretty darn good.

<img src="https://media-cdn.tripadvisor.com/media/photo-s/06/a6/13/44/el-burrito-mercado.jpg">

### Indian

Next up is the the [Clay Pit](https://www.claypit.com/), a contemporary Indian restaurant in Austin.  They focus mostly on curry dishes with naan, though some of my friends from grad school can tell you that India has way more cuisine diversity than curry dishes.

<img src="http://www.funjunkie.com/wp-content/themes/GeoPlaces/timthumb.php?src=http://www.funjunkie.com/wp-content/uploads/2013/06/l-23.jpg&w=600&h=345&zc=1&q=70">


```python
# Clay Pit
retrieve_recommendations(338, 8)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>338</th>
      <td>1.000000</td>
      <td>Clay Pit Contemporary Indian Cuisine</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>Indian Restaurant</td>
      <td>8.9</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>904</th>
      <td>0.874870</td>
      <td>India Palace</td>
      <td>Saint Paul</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>909</th>
      <td>0.853726</td>
      <td>Darbar India Grill</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>905</th>
      <td>0.848635</td>
      <td>Dancing Ganesha</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>916</th>
      <td>0.847020</td>
      <td>Gandhi Mahal</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>917</th>
      <td>0.845153</td>
      <td>Best of India Indian Restaurant</td>
      <td>Saint Louis Park</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>906</th>
      <td>0.837536</td>
      <td>Gorkha Palace</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>912</th>
      <td>0.834911</td>
      <td>India House</td>
      <td>Saint Paul</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>910</th>
      <td>0.821674</td>
      <td>Copper Pot Indian Grill</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Indian Restaurant</td>
      <td>7.3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This was actually the first place I did a field test on.  When I originally looked this up, we ended up trying [Gorkha Palace](http://gorkhapalace.com/) since it was the closest one to our house with the best reviews.  It has a more expanded offering including Nepali and Tibetan food (though I wasn't complaining because I love [momos](https://en.wikipedia.org/wiki/Momo_(food)).  It was delicious, and was very similar to the Clay Pit.  We'll be going back.

<img style="width: 300px;" src="https://s3-media4.fl.yelpcdn.com/bphoto/gueLSLa3LY6LGjYVrGdZkw/o.jpg">

### French/Bistro

One of our favorite places back home is [Blue Dahlia Bistro](http://www.bluedahliabistro.com/), a European-style bistro specializing in French fusion.  They use a lot of fresh and simple ingredients, and it's a great place for a date night due to its cozy interior and decorated back patio.

<img src="https://ferociousfoodiedotcom.files.wordpress.com/2016/03/blue-dahlia21.jpg">


```python
# Blue Dahlia
retrieve_recommendations(124)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>124</th>
      <td>1.000000</td>
      <td>Blue Dahlia Bistro</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>Café</td>
      <td>9.1</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>584</th>
      <td>0.827561</td>
      <td>Wilde Roast Cafe</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Café</td>
      <td>8.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>762</th>
      <td>0.785928</td>
      <td>Jensen's Cafe</td>
      <td>Burnsville</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Café</td>
      <td>8.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>741</th>
      <td>0.781564</td>
      <td>Cafe Latte</td>
      <td>Saint Paul</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Café</td>
      <td>9.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>759</th>
      <td>0.777269</td>
      <td>Peoples Organic</td>
      <td>Edina</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Café</td>
      <td>8.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>686</th>
      <td>0.772962</td>
      <td>Black Dog, Lowertown</td>
      <td>Saint Paul</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Café</td>
      <td>8.6</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



I think our heavier category weighting is hurting us here since Blue Dahlia is classified as a café.  Most of the recommendations focus on American food (remember, American food is king in Minneapolis), but I'm guessing the [Wilde Roast Cafe](http://wildecafe.com/) was listed as the most similar restaurant due to the similarly cozy interior and various espresso drinks they offer.  I've been to the Wilde Roast before beginning this project, and I can tell you that the food is completely different.

<img style="width: 350px;" src="https://igx.4sqi.net/img/general/600x600/2736_Sy4oGIGsTbzfD5ykDTnPCUDgISnx9xQnDlCpqu6bGyU.jpg">


### Coffee

Speaking of coffee, let's wrap this up with coffee recommendations.  I still have a lot of places to find matches for, but since I did this project as a poor graduate student, most of them would be "that looks promising, but I haven't tried it yet".  

Sadly, a lot of my favorite coffee shops from in and around Austin didn't show up since Starbucks took up most of the space when searching for coffee places (remember, we're limited to 50 results per category).  We ended up with [Mozart's](http://www.mozartscoffee.com/):

<img src="http://www.mozartscoffee.com/images/photo9_5.jpg">

and the [Coffee Bean & Tea Leaf](https://www.coffeebean.com/):

<img src="http://dining.ucr.edu/images/foodshot-coffeebean.jpg">

I ran the results for Mozart's and didn't get anything too similar back.  To be fair, there aren't any coffee shops on a river up here, and I'm sure most comments for Mozart's are about the view.  

Let's go with The Coffee Bean & Tea Leaf instead. It's actually a small chain out of California that is, in my opinion, tastier than Starbucks.


```python
# Coffee Bean & Tea Leaf
retrieve_recommendations(558)
```




<div style="overflow-x:auto;">
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cosineSimilarity</th>
      <th>Name</th>
      <th>City</th>
      <th>state</th>
      <th>Location</th>
      <th>category</th>
      <th>rating</th>
      <th>selfRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>558</th>
      <td>1.000000</td>
      <td>The Coffee Bean &amp; Tea Leaf</td>
      <td>Austin</td>
      <td>TX</td>
      <td>Austin, TX</td>
      <td>Coffee Shop</td>
      <td>8.1</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>742</th>
      <td>0.811790</td>
      <td>Five Watt Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>8.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>747</th>
      <td>0.805479</td>
      <td>Dunn Bros Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>7.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>757</th>
      <td>0.797197</td>
      <td>Caribou Coffee</td>
      <td>Savage</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>7.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>760</th>
      <td>0.791427</td>
      <td>Dunn Bros Coffee</td>
      <td>Minneapolis</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>7.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>778</th>
      <td>0.790609</td>
      <td>Starbucks</td>
      <td>Bloomington</td>
      <td>MN</td>
      <td>Minneapolis, MN</td>
      <td>Coffee Shop</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



These are perfectly logical results.  [Caribou](https://www.cariboucoffee.com/) is a chain popular in the Midwest (I also describe it as 'like Starbucks but better' to friends back home), and [Dunn Bros](https://dunnbrothers.com/) is similar, but specific to Minnesota.

This chart from [an article on FlowingData](http://flowingdata.com/2014/03/18/coffee-place-geography/) helps describe why I think these results make so much sense:

<img src="http://i1.wp.com/flowingdata.com/wp-content/uploads/2014/03/coffee-breakdowns-final.png?w=954">

As for my verdict on Caribou, I actually like it better than the Coffee Bean and Tea Leaf.  In fact, I actually have a gift card for them in my wallet right now.  There also used to be a location for the Coffee Bean and Tea Leaf up here, but they closed it down shortly after I moved here (just like all of the Minnesotan [Schlotzsky's](https://www.schlotzskys.com/) locations...I'm still mad about that).

# Summary

Like any other tool, this method isn't perfect for every application, but it can work if we use it effectively. While there is room for improvement, I am pleased with how it has been working for me so far.

I'm going to continue to use this when we can't decide on a restaurant and feeling homesick, and will likely update this post in the future with either more places I try out or when I move to a new city in the future. In the meantime, I hope you enjoyed reading, and feel free to use my code ([here is the github link](https://github.com/JeffMacaluso/Blog/blob/master/Restaurant%20Recommender.ipynb)) to try it out for your own purposes.

Happy eating!
