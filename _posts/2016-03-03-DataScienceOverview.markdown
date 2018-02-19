---
layout: single
classes: wide
title:  "Data Science Overview"
date:   2016-03-03 20:04:11 -0500
categories: jekyll update
---


This was something I threw together for a FAQ/sidebar link on the [data science subreddit](www.reddit.com/r/datascience) to help with common questions about the field.  Since this field is constantly growing and changing, the information below may not be fully up to date.  Regardless, I hope it can help!

# Overview:
If you're reading this, you’ve probably heard a little about data science, big data, or machine learning, and are interested in learning more.  Maybe you're thinking about becoming a data scientist, working somewhere within the field of data science, or are just curious about the hype.  This guide will hopefully give you enough details and information to get you started, help you figure out if you would like to continue learning more, or give you a broader understanding of the field.
 
### What they do
Data scientists do a lot of things.  Because it is an amalgamation of multiple skillsets and industries, there is a lot of diversity in what a data scientist actually does.  It might be predicting the future, such as future revenue, who is at risk of turnover, or working on a trading algorithm to purchase beneficial stocks before the market rate adjusts.  It could also be describing current situations, like the classic example of discovering that [males who buy diapers are more likely to buy beer](http://www.theregister.co.uk/2006/08/15/beer_diapers/).  There are countless other examples, but it boils down to analyzing data to discover insights.
 
### Industries
Data science is diverse enough to apply to a variety of industries.  It is most prevalent in the tech industry, but has expanded into the financial industry, retail, manufacturing, non-profits, and more.

### Salaries
Due to the high demand and low supply of data science positions, as well as their propensity to include senior positions the average salary looks attractive. O’Reilly Media does a comprehensive salary survey every year ([here](http://www.oreilly.com/data/free/files/2015-data-science-salary-survey.pdf) is the 2015 survey) which breaks down a lot of areas, including geographical area and technology used.  Since a lot of positions are in the Bay Area (which includes a high cost of living), the average salary is can be skewed, so it’s best to check Glassdoor for salary by area and company.  For instance, data scientist positions in the UK will command roughly half of those on the west coast in the USA.

In addition to having a generous salary, data scientists are usually higher on the charts of job satisfaction, which is one reason why it is often called “the sexiest job of the decade”.  This can be due to a mix of autonomy, novel and creative problem solving, and the opportunity to make an impact. [Here](https://www.glassdoor.com/Best-Jobs-in-America-LST_KQ0,20.htm) is Glassdoor’s 2016 best jobs report showing data scientist at the top.


### Areas
The largest concentration of data scientist positions are in the Bay Area of California, though there are several positions in larger cities.  New York City, Washington D.C., Seattle, Austin, and Chicago have large amounts of data scientist positions available, but this is also rapidly changing.  These areas also have different industry focuses (finance in NYC, tech in San Francisco, government in Washington D.C., etc.), but these are generalizations.
Similar roles

There are similar supporting roles for those that are interested in the world of data, but do not want to become a data scientist.  Data engineers have more of a technical focus, and are generally more concerned with acquiring data, structuring it, storing it, and sometimes preparing it for data scientists.  Data analyst positions are often preliminary positions to the role of a data scientist. Their roles can include lower-level tasks, and usually don’t require an advanced degree.  Database Administrators (DBAs) are similar to data engineers, with a focus on maintaining a SQL data warehouse.

### Skills
Data scientists are in high demand due to the diversity and flexibility of their skillset.  The [traditional venn diagram](http://i.stack.imgur.com/OQqi5.png) describes data scientists as having in depth knowledge in statistics, computer science, and domain expertise.  It is now argued that communication is another skill that should be added to their primary functions, or replace the "domain expertise" function.
 
#### Statistics
Some argue that a data scientist is just an applied statistician, which gives an idea of the breadth and depth of statistics used in the role.  Data scientists must have a good understanding of the two branches of statistics: frequentist and Bayesian.  They must also have a good grasp on a variety of other statistical concepts, including tests, distributions, values, biases, modelling (especially under/overfitting), and so on.  Several books have been written that include the necessary statistical knowledge a data scientist must have, which is one of many reasons why a degree in statistics is one of the most recommended degrees for the field.

#### Computer Science
Due to the heavy amount of required programming and work with complex systems, computer science skills are crucial.  A data scientist must be able to work with multiple systems to collect, clean, analyze, and distribute their work.  They must be able to optimize their code to reduce computational cost, create interactive reports for the end user, and be able to put their code into production, among many other things.
 
#### Machine Learning
Machine learning is a large part of data science, and those not familiar with it are often confused about the concept of machine learning.  Machine learning is a subset of artificial intelligence that is, in layman's terms, feeding a computer data and instructing it to figure it out.  There are two divisions of this:
 
- Supervised: When you split your data between a "training" set (usually 70% of the data), and a "testing" set (the remaining %), feed the computer the training set to "tell it the answer", let it figure out how to get there, and test it with the test set.  The two subtypes of this are regression (predicting a number), and classification (choosing what group to put it in).
 
- Unsupervised: When you give all of the data to the machine, and let it figure it all out on its own.  The most common form of this is known as clustering.  This is where the computer figures out the statistical relationship between various entities in a detail in which humans are often not able to calculate on their own.
 
#### "Big Data"
To both the general public and a lot of non-technical articles, data science is often synonymous with "big data".  To set the record straight, some data scientists work with "big data".  However, the majority do not.  This boils down to the definition of what constitutes "big data" - some think of it as gigabytes, others as terabytes, but the wide consensus among data scientists is petabytes.  Generally, if it is something that a requires processing on either a distributed computing network (using a network of computers to process the data) or on servers, it is more likely to be considered "big data".
 
#### Communication
"You can create the best model in the world, but it will be useless if nobody understands or uses it" is a common adage in the field.  In order to get desired results, business users will have to understand your insights.  Since most business users will not understand or care about the technical jargon, this means explaining things in a clear and concise way that is understandable and useable by the business users.  As a data scientist, you must read your audience. The majority of business users are from another world , and you have to be able to speak their language.
 
#### Domain Expertise
It is difficult to be fully effective if you are unfamiliar with the data you are working with and the numbers seem arbitrary.  This may not be as crucial if you have good communication skills and are able to get the desired information from a subject matter expert, but it is still crucial to have a good understanding of your data.
 
## Education
### Graduate School
While there are data scientists out there with only an undergraduate degree, the vast majority have graduate degrees.  This is because a "data scientist" is often considered a senior position, and thus requires copious amounts of experience and/or education.  According to the O’Reilly 2014 Data Scientist Survey, roughly 65% of data scientists held a master’s degree or above.

### Ph.D.

One question that often pops up is if it is worth it to get a PhD. to break into the field.  Those with a PhD. often have an easier time finding a job and a slightly higher salary, but it is not often recommended unless you were already interested in pursuing one.  For example, the 2015 O'Reilly Data Scientist Salary Survey shows the salary increase from an MS to a PhD. as less than the salary benefit of being proficient with Apache Spark.  Couple this with the ratio of 9:4 MS to PhD. data scientists, and it paints a clearer picture.
 
#### Which Major (Data Science, Statistics, Computer Science, etc.)
One hotly debated topic in the field is what type of graduate degree to pursue.  Traditional data science teams include people from a wide variety of quantitative backgrounds, which brings in diverse and creative methods to solve problems instead of a standard textbook approach.  There are a lot of data science and business or predictive analytics programs popping up, and a lot of data scientists can be very critical of these, often urging aspiring data scientists to choose a statistics, computer science, or heavily quantitative program instead.  The main reason for this being that some of these new programs are capitalizing on the "new hot field of data science", charging more than they would for other graduate programs, and providing vapid course material.  However, a substantial amount of data scientists agree that these programs are more beneficial than a stats or computer science program - it just depends on the substance of the program itself. Once again, this is a highly debated topic, and we see instances of recruiters that refuse to hire individuals from a data science/analytics program , and recruiters that [refuse](https://www.reddit.com/r/datascience/comments/49z0zq/view_on_masters_in_data_science_vs_stats/d0w1wdf) to hire those from a statistics program. 

### Bootcamps
There are a lot of data science boot camps springing up, and their value is often a point of contention.  Some see them as cash grabs with no real payout (akin to predatory for-profit universities), and it has helped others gain the necessary skills to transition into the field.  The general perception of these programs is that they are only productive if you have an advanced degree and/or experience in another field, are looking to make a career switch to data science, and will not go into debt or give up a well paying job for them.

### MOOCs
There are a variety of quality Massively Open Online Courses, or MOOCs, which provide very valuable information to both aspiring and current data scientists.  These can help you "get your feet wet" and determine if this is the right field for you before investing in a graduate degree.  Websites like Coursera, Udemy, Udacity, and others host a variety of courses for free that are worth taking a look at.  Most notably is Andrew Ng's [Machine Learning course](https://www.coursera.org/learn/machine-learning).  While this uses Matlab or Octave, it shows the math behind common machine learning algorithms without going too into depth.  Another honorable mention is the [John Hopkins data science series](https://www.coursera.org/specializations/jhu-data-science). 

### Certifications
Many places offer certifications in different technologies, and there is a lot of debate over if these are worth the investment.  The general consensus seems to be that it is usually only worth it if your employer is paying for it, and that a fair amount of recruiters don't take them too seriously.
 
 
## Programs/Software
Like most things in the realm of data science, the best tools to use are often debated.  It is easy to get wrapped up in which language is better or why you should use one tool instead of another, but the most important thing is to focus on understanding the ideas and theories behind the work, instead of the tools to execute those theories.  The tools you use will undoubtedly change over your career, but the ideas behind them will most likely remain consistent.  
 
### Open Source Languages (R, Python, and Julia)
Most positions use open source languages like R or Python heavily.  This is partially because companies do not have to pay heavy licensing fees, and also because anyone can install them on their computer, so it is easier to find people with experience in them.  Additionally, anybody can develop a package for them, so there is a lot of added functionality available.

#### R:
- **Overview:** R is a statistical programming language that has seen a [large surge in popularity within the past few years](http://a7.typepad.com/6a0105360ba1c6970c01b8d1773127970c-pi).  Microsoft recently acquired the primary R developer (Revolution R), and has begun incorporating R in the SQL Server 2016 suite.
    
- **Strengths:** Natively handles many statistical functions, a purpose built IDE for scientific computing and data analysis (RStudio), easy to set up, and includes more statistical packages and available examples than any other language (greater depth).
    
- **Weaknesses:** Often described as "a statistical language written by statisticians", so the syntax is considered unintuitive or awkward and the speeds are slower.
    
- **Misc.:** Recommended for those with a math background.  If you are interested in R, try the [swirl package](http://swirlstats.com/students.html) to learn R within R.
  
#### Python:
- **Overview:** Python is a general purpose, high level, interpreted, dynamic programming language.  That means it does a lot of things.  Thanks to the availability of packages like pandas, numpy, scikit-learn, and matplotlib, it also does scientific computing.  Like R, it has seen a surge in popularity recently, but it is difficult to quantify the exact amount for scientific and statistical computing because it is also used for other purposes.
 
- **Strengths:** More versatile/wider breadth, more intuitive syntax, easy to learn, easier to data munge/clean, and faster than R.
 
- **Weaknesses:** Must import packages which results in more code, more difficult to set up, IDEs are not as robust as RStudio, and less statistical packages than R.
    
- **Misc.:** Recommended for those with any prior programming experience due to the syntax.  There are a variety of free and paid resources to learn Python, including [DataCamp](https://www.datacamp.com/courses/intro-to-python-for-data-science), [Learning Python for Data Analysis and Visualization](https://www.udemy.com/learning-python-for-data-analysis-and-visualization/), and many others.
 
#### Julia
- **Overview:** Julia is a newer open source language being developed by MIT with the goal of having the functionality of R, the syntax of Python, and the speed of C.
    
- **Strengths:** [Faster](http://julialang.org/benchmarks/) than the other two  (though [this blog post](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Python_Meets_Julia_Micro_Performance?lang=en) from a Python developer argues that Python is faster than those benchmarks), designed for scientific and statistical computing. 
    
- **Weaknesses:** Extremely young compared to the other two languages, which means far fewer packages, and much less likely to be used in corporate environments.
    
- **Misc.:** Since it's still such a young language, it might be worth keeping an eye on, but choose Python or R as your primary language.  If you still want to learn Julia, then try recreating projects you have done in Python or R in Julia.
 
### Poprietary (SAS, Matlab, RapidMiner, etc.)
Many companies traditionally used proprietary software, though this has been changing in recent years due to licensing costs.  These tend to be more common at larger companies that have employed statisticians or quantitative analysts in the past.  These can be difficult to learn on your own, but you can often get educational copies with a .edu email address.
 
### Databases
Structured Query Language, or SQL, is a mandatory proficiency for data scientists.  This is because you will not be able to analyze data if you cannot extract it to begin with.  There are a wide variety of free resources for learning SQL online, and while there are a few different "flavors" of SQL (Microsoft's T-SQL, IBM's DB2 SQL, etc.), they are all very similar.  Data scientists should also understand basic database concepts; fact/dim tables, star/snowflake schema, primary/secondary/foreign keys, levels of normalization, etc. 
 
### "Big Data" Framework
A lot of programs, applications, and frameworks for big data (Spark, Hadoop, MapReduce, Hive, Pig, etc.) are fortunately open source.  Proficiency in these programs can generally command higher salaries, so most see it as worthwhile to spend time researching the concepts behind them.  Unless working at a startup or extremely small company, data engineers are generally those that are in charge of maintaining these programs, and the data scientist must be able to extract the data they need from them. 
 
### Spreadsheets
There are a few positions out there with the title of "data scientist" that deal solely with Excel, and this upsets a lot of people.  Spreadsheets are useful under certain circumstances, but should not be the main tool for a data scientist.  The primary reason for this is that, unlike code, they are not easily reproducible.  Additionally, they do not have the capability to do anything beyond extremely basic statistical functions.  For a more in depth discussion on this topic, listen to [this episode](https://soundcloud.com/nssd-podcast/episode-9-spreadsheet-drama) of the "Not So Standard Deviations" podcast featuring one of the creators of the Data Science Specialization course on Coursera. 
 

## Where to go from here
### Kaggle
www.Kaggle.com is a great website to test your skills in real world problems against others in machine learning competitions.  There are basic free competitions, and competitions that offer cash prizes.  These can be great to put in your portfolio or on your resume, but realize that the data received here is much cleaner than what you would receive in the real world.  In addition, blackbox algorithms (such as random forests, neural networks, etc.) are favored since interpretability does not matter.

### Projects & Portfolio
Pick a project that you can get data for, and get going.  Some hiring managers prefer to see this over kaggle competitions because it shows a personal interest in the field, and provides more experience in dealing with messy data.  

After you’re done, create a Github account at www.Github.com and post your code to begin your coding portfolio.  Hiring managers are skeptical of candidates who do not have a Github account or coding portfolio since they provide visibility to your technical prowess.

### MOOCs
Here’s a short collection of MOOCs mentioned above.  Available MOOCs are growing every day, but these should give a good foundation:
- [Andrew Ng’s introduction to Machine Learning](https://www.coursera.org/learn/machine-learning) - focuses on the math behind machine learning algorithms, and uses MATLAB or Octave instead of Python or R


- [The John Hopkins University Data Science courses](https://www.coursera.org/specializations/jhu-data-science) - covers a variety of topics, and focuses in R.

- [Python for Data Analysis and Visualization](https://www.udemy.com/learning-python-for-data-analysis-and-visualization/) - covers data manipulation, visualization, machine learning, statistics, and applies them to datasets.


### Blogs
Here’s a short list of blogs and websites to keep up with news:
 
- [Flowing Data](http://flowingdata.com/): Beautiful data visualization over a variety of topics
- [Simply Statistics](http://simplystatistics.org/): Three biostatistics professors talk about statistical topics 
- [KDNuggets](http://www.kdnuggets.com/): An amalgamation of data science news stories and articles 
- [DataTau](http://www.datatau.com/): Described as a “Hacker News” for data science 
- [R-Bloggers](http://www.r-bloggers.com/): The primary R blog 
- [Data Elixer](http://dataelixir.com/): A newsletter including articles of data in the real world.

### Podcasts
Here's a short list of a few data science podcasts and what they do:
 
- [Linear Digressions](http://lineardigressions.com/): A data scientist and a software engineer explain and discuss a different topic each 10-20 minute episode. 
- [Partially Derivative](http://www.partiallyderivative.com/): Three data scientists talk about different news stories involving data while drinking beer or interview specialists in the field. 
- [Talking Machines](http://www.thetalkingmachines.com/): An MIT professor of machine learning and his cohost discuss different topics in machine learning or interview specialists.  It is a little more technical, and more focused towards machine learning than data science.
- [Data Skeptic](http://dataskeptic.com/): A data scientists interviews experts in the field or explains concepts in layman's terms to his wife.
- [Data Stories](http://datastori.es/): Specialists in the field discuss data topics, and are primarily focused on data visualization. 
- [O'Reilly Data Show](http://radar.oreilly.com/tag/oreilly-data-show-podcast): The chief data scientist of O’Reilly Media interviews experts in the field.
- [Not So Standard Deviations](https://soundcloud.com/nssd-podcast): A biostatistics professor in academia and a data scientist in the industry discuss a variety of topics.
