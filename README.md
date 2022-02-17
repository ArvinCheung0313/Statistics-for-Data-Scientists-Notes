## Statistics for Data Scientists Notes
This repo includes essential notes I read from "Practical Statistics for Data Scientists - 50 Essential Concepts Using R and Python".

Currently updating...

To assess the book, see the repo or [here](https://github.com/ArvinCheung0313/Statistics-for-Data-Scientists-Notes/blob/main/PracticalStatisticsforDataScientists50EssentialConceptsUsingRandPythonbyPeterBruceAndrewBrucePeterGedeck.pdf)

## Contents
* [1 Exploratory Data Analysis](#1-exploratory-data-analysis)
* [2 Data and Sampling](#2-data-and-sampling)
  * [2.1 Basic Key Terms](#21-basic-key-terms)
  * [2.2 When Does Size Matter?](#22-when-does-size-matter)
  * [2.3 Selection Bias](#23-selection-bias)
  * [2.4 Central Limit Theorem](#24-central-limit-theorem)
  * [2.5 The Bootstrap](#25-the-bootstrap)
  * [2.6 Confidence Intervals](#26-confidence-intervals)
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
  * [4.4 ANOVA](#44-anova)
  * [4.5 Chi-Square Test](#45-chi-square-test)
  * [4.6 Power and Sample Size](#46-power-and-sample-size)

## 1 Exploratory Data Analysis


## 2 Data and Sampling

### 2.1 Basic Key Terms
* **_Sample_**  
  A subset from a larger data set.
* **_Population_**  
  The larger data set or idea of a data set.
* **_N(n)_**  
  The size of the population(sample).
* **_Random Sampling_**  
  Drawing elements into a sample at random. A very powerful and basic sampling method.
* **_Stratified Sampling_**  
  Dividing the population into strata and randomly sampling from each strata. For example, randomly sampling from differnt age groups, geographic area.
* **_Stratum (pl., strata)_**  
  A homogeneous subgroup of a population with common characteristics.
* **_Simple Random Sample_**  
  The sample that results from random sampling without stratifying the population.
* **_Bias_**  
  Systematics error. An error can be minimized but cannot be fully eliminated in an experiment. It occurs when measurements or observaions are systematically in error because they are not representative of the full population.
* **_Sample Bias_**  
  A sample that misrepresents the population.
  
### 2.2 When Does Size Matter?
Sometimes smaller amounts of data is better. Time and effort spent on random sampling not only reduces bias but also allows greater attention to data exploration and data quality. It might be prohibitively expensive to track down missing values or evaluate outliers (which may contain useful information) in millions of records, but doing to in a sample of several thousand records may be feasible and efficient.

**When are massive amounts of data necessary?**
One classic scenario for the value of big data is when the data is not only **big** but also **sparse** as well. Only when such enormous quantities of data are accumulated can effective search results be returned for most queries when considering the search queries reveived by Google.

### 2.3 Selection Bias
Bias resulting from the way in which observations are selected and the samples cannot fully represent the whole population. For example, when analyzing which web design is more attractive to customers, we cannot decide which customer view which version of web, and such a self-selection lead to imbalance groups of samples.

### 2.4 Central Limit Theorem
In many situations, when independent random variables are summed up, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed. 

The Central Limit Theorem assumes as follow:
* **Randomization Condition**: The data must be sampled randomly.
* **Independence Assumption**: The sample values must be independent of each other. This means that the occurrence of one event has no influence on the next event. Usually, if we know that people or items were selected randomly we can assume that the independence assumption is met.
* **10% Condition**: When the sample is drawn without replacement (usually the case), the sample size, n, should be no more than 10% of the population.
* **Sample Size Assumption**: The sample size must be sufficiently large. Although the Central Limit Theorem tells us that we can use a Normal model to think about the behavior of sample means when the sample size is large enough, it does not tell us how large that should be. If the population is very skewed, you will need a pretty large sample size to use the CLT, however if the population is unimodal and symmetric, even small samples are acceptable. So think about your sample size in terms of what you know about the population and decide whether the sample is large enough. In general a sample size of 30 is considered sufficient if the sample is unimodal (and meets the 10% condition).

### 2.5 The Bootstrap
One easy and effective way to estimate the sampling distribution of a model parameters, is to draw additional samples, with replacement, from the sample itself and recalculate the model for each resample. This procedure is called **_Bootstrap_**.

Such a procedure does not involve any assumptions about the data or the sample statistic being normally distributed. For a sample of *n*, steps are sa follow:
1. Draw a sample value, record it, and then replace it.
2. Repeat *n* times.
3. Record the mean of the *n* resampled values.
4. Repeat steps 1-3 *M* times. (*M* could be 100, 1000, etc.)
5. Use the *M* results to:
   * Calculate their standard deviation (this estimates sample mean standard error).
   * Produce a histogram or boxplot.
   * Find a confidence interval.

The bootstrap is a powerful tool for assessing the variability of a sample statistic. It can be applied in a wide variety of circumstances without extensive study of mathematical approximations of sampling distributions. It also allows us to estimate sampling distributions for statistics where no mathematical approximation has been developed. When applied to predictive models, aggregating multiple boostrap sample predictions (bagging) outperforms the use of a single model.

### 2.6 Confidence Intervals
* **_Confidence Level_**  
  The percentage of confidence intervals, constructed in the same way from the same population, that are expected to contain the statistic of interest. Given a 90% confidence interval, it covers the central 90% of the bootstrap sampling distribution of a sample statistic. More generally, an 90% confidence interval should contain similar sample estimates 90% of the time on average in similar sampling procedure.

  *Figure 2-1. Boostrap confidence interval for the annual income of loan applications, based on a sample of 20*
  <img width="450" alt="Figure 2-9" src="https://user-images.githubusercontent.com/91806768/154370693-2f438d49-1b91-444b-a4f9-1acfac822edf.png">

* **_Interval Endpoints_**  
  The top and bottom of the confidence interval. The confidence interval consists of the upper and lower bounds of the estimate you expect to find at a given level of confidence.

## 3 Common Distributions

### 3.1 Normal Distribution
* **_Bell Curve_**
  Normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve.

  In a normal distribution, 68% of the data lies within one standard deviation of the mean, and 95% lies within two standard deviations.

  *Figure 2-2. Normal Curve*  
  <img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154372563-825e72d0-7274-4528-a54b-94c93cfa2ddd.png">

* **_Standard Normal Distribution_**
  A normal distribution with mean = 0 and standard deviation = 1. We can transform data through *normalization* or *standardization*. The transformed value is called *z-score*, and the normal distribution is sometimes called *z-distribution*.
  
  We can use a *QQ-Plot* to see how close a sample is to a specified distribution, say normal distribution. If the points roughly fall on the diagonal line, then the sample distribution can be regarded close to normal.
  
  *Figure 2-3. QQ-Plot of a sample of 100 values drawn from a standard normal distribution*
  <img width="450" alt="image" src="https://user-images.githubusercontent.com/91806768/154375710-a5ca65fa-eab3-4756-8df5-22d75482f59c.png">

### 3.2 Binomial Distribution
* **_AKA Bernoulli Distribution_**  
  The binomial distribution is the frequency distribution of the number of successes(x) in a given number of trials(*n*) with specified probability(*p*) of success in each trial. There is a family of binomial distributions, depending on the values of *n* and *p*. The binomial distribution would answer a question like:
  > If he probability of a click converting to a sale is 0.02, what is the probability of observing 0 sales in 200 clicks?

* **_Characterics_**
  * **mean**: *n* * *p*
    Which is known as expected value.
  * **variance**: *n* * *p*(1-*p*)

### 3.3 Geometric Distribution


### 3.4 Poisson Distribution


### 3.5 Exponential Distribution










## 4 Statistical Experiments and Significance Testing

### 4.1 A/B Testing


### 4.2 Statistical Significance and p-Values


### 4.3 *t*-Tests


### 4.4 ANOVA


### 4.5 Chi-Square Test


### 4.6 Power and Sample Size


