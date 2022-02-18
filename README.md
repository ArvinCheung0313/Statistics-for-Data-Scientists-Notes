## Statistics for Data Scientists Notes
This repo includes essential notes I read from "Practical Statistics for Data Scientists - 50 Essential Concepts Using R and Python".

:surfer:Currently updating...:surfing_man: 2020-02-17

To assess the book, see the repo or [here](https://github.com/ArvinCheung0313/Statistics-for-Data-Scientists-Notes/blob/main/PracticalStatisticsforDataScientists50EssentialConceptsUsingRandPythonbyPeterBruceAndrewBrucePeterGedeck.pdf)

## Contents
* [1 Exploratory Data Analysis](#1-exploratory-data-analysis)
* [2 Data and Sampling](#2-data-and-sampling)
  * [2.1 Basic Key Terms](#21-basic-key-terms)
  * [2.2 When Does Size Matter?](#22-when-does-size-matter)
  * [2.3 Selection Bias](#23-selection-bias)
  * [2.4 Central Limit Theorem](#24-central-limit-theorem)
  * [2.5 Law of Large Numbers (LLN)](#25-law-of-large-numbers-lln)
  * [2.6 The Bootstrap](#26-the-bootstrap)
* [3 Common Distributions](#3-common-distributions)
  * [3.1 Normal Distribution](#31-normal-distribution)
  * [3.2 Binomial Distribution](#32-binomial-distribution)
  * [3.3 Geometric Distribution](#33-geometric-distribution)
  * [3.4 Poisson Distribution](#34-poisson-distribution)
  * [3.5 Exponential Distribution](#35-exponential-distribution)
* [4 Statistical Experiments and Significance Testing](#4-statistical-experiments-and-significance-testing)
  * [4.1 A/B Testing](#41-ab-testing)
  * [4.2 Statistical Significance and p-Values](#42-statistical-significance-and-p-values)
  * [4.3 *t*-Tests](#43-t-tests)
  * [4.4 *z*-Tests](#44-z-tests)
  * [4.5 ANOVA](#45-anova)
  * [4.6 Chi-Square Test](#46-chi-square-test)
  * [4.7 Power and Sample Size](#47-power-and-sample-size)
  * [4.8 Simpson's Paradox](#48-simpsons-paradox)
* [5 Classification](#5-classification)
  * [5.1 Naive Bayes](#51-naive-bayes)
  * [5.2 Confusion Matrix](#52-confusion-matrix)
  * [5.3 Bias-variance Trade Off](#53-bias-variance-trade-off)
* [6 Probability](#6-probability)
  * [6.1 Conditional Probability](#61-conditional-probability)
  * [6.2 Bayesian Formula](#62-bayesian-formula)

## 1 Exploratory Data Analysis


## 2 Data and Sampling

### 2.1 Basic Key Terms

#### 2.1.1 Sample
A subset from a larger data set.

#### 2.1.2 Population
The larger data set or idea of a data set.

#### 2.1.3 N(n)
The size of the population(sample).

#### 2.1.4 Random Sampling
Drawing elements into a sample at random. A very powerful and basic sampling method.

#### 2.1.5 Stratified Sampling
Dividing the population into strata and randomly sampling from each strata. For example, randomly sampling from differnt age groups, geographic area.

#### 2.1.6 Stratum (pl., strata)
A homogeneous subgroup of a population with common characteristics.

#### 2.1.7 Simple Random Sample
The sample that results from random sampling without stratifying the population.

#### 2.1.8 Bias
Systematics error. An error can be minimized but cannot be fully eliminated in an experiment. It occurs when measurements or observaions are systematically in error because they are not representative of the full population.

#### 2.1.9 Sample Bias
A sample that misrepresents the population.
  
### 2.2 When Does Size Matter?
Sometimes smaller amounts of data is better. Time and effort spent on random sampling not only reduces bias but also allows greater attention to data exploration and data quality. It might be prohibitively expensive to track down missing values or evaluate outliers (which may contain useful information) in millions of records, but doing to in a sample of several thousand records may be feasible and efficient.

**When are massive amounts of data necessary?**  
> One classic scenario for the value of big data is when the data is not only **big** but also **sparse** as well. Only when such enormous quantities of data are accumulated can effective search results be returned for most queries when considering the search queries reveived by Google.

### 2.3 Selection Bias
Bias resulting from the way in which observations are selected and the samples cannot fully represent the whole population. For example, when analyzing which web design is more attractive to customers, we cannot decide which customer view which version of web, and such a self-selection lead to imbalance groups of samples.

### 2.4 Central Limit Theorem
In many situations, when independent random variables are summed up, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed. 

The Central Limit Theorem assumes as follow:
* **Randomization Condition**: The data must be sampled randomly.
* **Independence Assumption**: The sample values must be independent of each other. This means that the occurrence of one event has no influence on the next event. Usually, if we know that people or items were selected randomly we can assume that the independence assumption is met.
* **10% Condition**: When the sample is drawn without replacement (usually the case), the sample size, n, should be no more than 10% of the population.
* **Sample Size Assumption**: The sample size must be sufficiently large. Although the Central Limit Theorem tells us that we can use a Normal model to think about the behavior of sample means when the sample size is large enough, it does not tell us how large that should be. If the population is very skewed, you will need a pretty large sample size to use the CLT, however if the population is unimodal and symmetric, even small samples are acceptable. So think about your sample size in terms of what you know about the population and decide whether the sample is large enough. In general a sample size of 30 is considered sufficient if the sample is unimodal (and meets the 10% condition).

### 2.5 Law of Large Numbers (LLN)
In statistics, the theorem that, as the number of identically distributed, randomly generated variables increases, their sample mean (average) approaches their theoretical mean. In other words, we can replace probability with frequency approximation; we can replace the overall mean with sample mean approximation.

### 2.6 The Bootstrap
One easy and effective way to estimate the sampling distribution of a model parameters, is to draw additional samples, with replacement, from the sample itself and recalculate the model for each resample. This procedure is called **_Bootstrap_**.

Such a procedure does not involve any assumptions about the data or the sample statistic being normally distributed. For a sample of *n*, steps are sa follow:
* Draw a sample value, record it, and then replace it.
* Repeat *n* times.
* Record the mean of the *n* resampled values.
* Repeat steps 1-3 *M* times. (*M* could be 100, 1000, etc.)
* Use the *M* results to:
  * Calculate their standard deviation (this estimates sample mean standard error).
  * Produce a histogram or boxplot.
  * Find a confidence interval.

The bootstrap is a powerful tool for assessing the variability of a sample statistic. It can be applied in a wide variety of circumstances without extensive study of mathematical approximations of sampling distributions. It also allows us to estimate sampling distributions for statistics where no mathematical approximation has been developed. When applied to predictive models, aggregating multiple boostrap sample predictions (bagging) outperforms the use of a single model.

## 3 Common Distributions

### 3.1 Normal Distribution
#### 3.1.1 Bell Curve
Normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve.

In a normal distribution, 68% of the data lies within one standard deviation of the mean, and 95% lies within two standard deviations.

*Figure 2-1. Normal Curve*  
<img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154372563-825e72d0-7274-4528-a54b-94c93cfa2ddd.png">

#### 3.1.2 Standard Normal Distribution
A normal distribution with mean = 0 and standard deviation = 1. We can transform data through *normalization* or *standardization*. The transformed value is called *z-score*, and the normal distribution is sometimes called *z-distribution*.
  
We can use a *QQ-Plot* to see how close a sample is to a specified distribution, say normal distribution. If the points roughly fall on the diagonal line, then the sample distribution can be regarded close to normal.
  
*Figure 2-2. QQ-Plot of a sample of 100 values drawn from a standard normal distribution*
<img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154375710-a5ca65fa-eab3-4756-8df5-22d75482f59c.png">

### 3.2 Binomial Distribution

#### 3.2.1 AKA Bernoulli Distribution
The binomial distribution is the frequency distribution of the number of successes(x) in a given number of trials(*n*) with specified probability(*p*) of success in each trial. There is a family of binomial distributions, depending on the values of *n* and *p*. The binomial distribution would answer a question like:
> If the probability of a click converting to a sale is 0.02, what is the probability of observing 0 sales in 200 clicks?

#### 3.2.2 Characteritics
* **mean**: *n***p*
  Which is known as expected value.
* **variance**: *n***p*(1-*p*)

### 3.3 Geometric Distribution

#### 3.3.1 Definition
Geometric distribution is a type of discrete probability distribution that represents the probability of the number of successive failures before a success is obtained in a Bernoulli trial. A Bernoulli trial is an experiment that can have only two possible outcomes, ie., success or failure. In other words, in a geometric distribution, a Bernoulli trial is repeated until a success is obtained and then stopped. A Geometric distribution has the following assumptions:
* Each trials are independent to each other
* Each trails are Bernoulli distribution
* The probability of success for each trial, namely *p*, is identical

#### 3.3.2 Characteritics
* **mean**: 1/*p*
  Which is known as expected value.
* **variance**: (1-*p*)/*p*^2

### 3.4 Poisson Distribution

#### 3.4.1 Definition
A Poisson distribution is a discrete probability distribution that is used to show how many times an event is likely to occur over a specified period. In other words, it is a count distribution. Poisson distributions are often used to understand independent events that occur at a constant rate within a given interval of time. It is useful when addressing queuing questions such as “How much capacity do we need to be 95% sure of fully processing the internet traffic that arrives on a server in any five- second period?”

#### 3.4.2 Characteritics
* **Lambda**
The key parameter in a Poisson distribution is λ, or lambda. This is the mean number of events that occurs in a specified interval of time or space. The variance for a Pois‐ son distribution is also λ.
* **mean**: λ
* **variance**: λ

### 3.5 Exponential Distribution
#### 3.5.1 Definition
A Exponential distribution is a continuous distribution which is used to measure the time or distance from an event to the next event to occur. For example, the expected length of time to detect a flaw in an assemble line, or the time for a next earthquake to happen.

#### 3.5.2 Characteritics
* **mean**: 1/λ
* **variance**: 1/λ^2

## 4 Statistical Experiments and Significance Testing

### 4.1 A/B Testing


### 4.2 Statistical Significance and p-Values

#### 4.2.1 P-value
Given a chance model that embodies the null hypothesis, the p-value is the probability of obtaining results as unusual or extreme as the observed results.

Note: P-value does not provide a good measure of evidence regarding a model or hypothesis, it does not measure the size of an effect or the importance of a result.

#### 4.2.2 Alpha
Also known as **_Significance Level_**, is the probability threshold of “unusualness” that chance results must surpass for actual outcomes to be deemed statistically significant. Typical alpha(or significance) levels are 5% and 1%.

#### 4.2.3 Type I Error
Mistakenly concluding an effect is real (when it is due to chance), also known as the false positive. In other words, a type I error occurs when a person rejects a **true** null hypothesis. The probability of making a type I error is represented by your alpha level (α).
  
*Figure 4-1. Type I Error*  
<img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154390478-ec2e8d95-566d-4bb2-bb1e-220d7822424f.png">

#### 4.2.4 Type II Error
Mistakenly concluding an effect is due to chance (when it is real), also known as false negative. In other words, a type II error occurs when a person failed to reject a **false** null hypothesis. The probability of making a type II error is called Beta (β), and this is related to the power of the statistical test (power = 1- β). We can understand them by using a confusion matrix.
  
*Figure 4-2. Type II Error*  
<img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154390424-f74f44db-1931-4b7f-be82-1f332271a7ad.png">
  
#### 4.2.5 Confidence Level
The percentage of confidence intervals, constructed in the same way from the same population, that are expected to contain the statistic of interest. Given a 90% confidence interval, it covers the central 90% of the bootstrap sampling distribution of a sample statistic. More generally, an 90% confidence interval should contain similar sample estimates 90% of the time on average in similar sampling procedure.

*Figure 4-4. Boostrap confidence interval for the annual income of loan applications, based on a sample of 20*  
<img width="450" alt="Figure 2-9" src="https://user-images.githubusercontent.com/91806768/154370693-2f438d49-1b91-444b-a4f9-1acfac822edf.png">

#### 4.2.6 Confidence Interval
The confidence interval consists of the upper and lower bounds of the estimate you expect to find at a given level of confidence.

#### 4.2.7 Practical Significance
Practical significance is the level of change that you would expect to see from a business standpoint for the change to be valuable. What is considered practically significant can vary by field. In medicine, one would expect a 5,10 or 15% improvement for the result to be considered practically significant. At Google, for example, a 1-2% improvement in click through probability is practically significant.

The statistical significance bar is often lower than the practical significance bar, so that if the outcome is practically significance, it is also statistically significant.
  
### 4.3 *t*-Tests
A type of inferential statistic used to determine if there is a significant difference between the means of two groups, which may be related in certain features.
* **Formula**  
  <img width="139" alt="image" src="https://user-images.githubusercontent.com/91806768/154591478-acc706c9-c805-4879-9e0c-5fdde64237e3.png">  
  Where:
  * m = mean
  * μ = theoretical value
  * sd = standard deviation
  * n = The sample size (the number of paired differences)

* **Assumptions**  
  * The scale of measurement applied to the data collected follows a continuous or ordinal scale, such as the scores for an IQ test.
  * Of a simple random sample, the data is collected from a representative, randomly selected portion of the total population.
  * The data should result in a normal distribution, bell-shaped distribution curve when plotted.
  * Homogeneous of variance exists when the standard deviations of samples are approximately equal.

### 4.4 *z*-Tests
A statistical test to determine whether two population means are different when the variances are known and the sample size is large. It's also a hypothesis test in which the z-statistic follows a normal distribution.

* **Formula**  
  <img width="139" alt="image" src="https://user-images.githubusercontent.com/91806768/154591580-310e0bda-ee9a-409f-b2b1-fda6c57ca814.png">  
  Where:
  * X = sample average
  * μ0 = mean
  * sd = standard deviation

* **Differences Between *z*-Tests And *t*-Tests**
  * **Sample Size**  
    z-tests are closely related to t-tests, but t-tests are best performed when an experiment has a small sample size, less than 30.
  * **Known Sd**  
    Also, t-tests assume the standard deviation is unknown, while z-tests assume it is known. If the standard deviation of the population is unknown, but the sample size is greater than or equal to 30, then the assumption of the sample variance equaling the population variance is made while using the z-test.

* **z-Score**  
  A z-score, or z-statistic, is a number representing how many standard deviations above or below the mean population the score derived from a z-test is. It describes a value's relationship to the mean of a group of values. 0 indicates that the data point's score is identical to the mean score. It could be positive or negative.

### 4.5 ANOVA


### 4.6 Chi-Square Test


### 4.7 Power and Sample Size


### 4.8 Simpson's Paradox


## 5 Classification
### 5.1 Naive Bayes


### 5.2 Confusion Matrix


### 5.3 Bias-variance Trade Off


## 6 Probability

### 6.1 Conditional Probability


### 6.2 Bayesian Formula 
