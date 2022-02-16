# Statistics-for-Data-Scientists-Notes
This repo includes notes I read from "Practical Statistics for Data Scientists - 50 Essential Concepts Using R and Python".
To assess the book, see the repo or [here]<PracticalStatisticsforDataScientists50EssentialConceptsUsingRandPythonbyPeterBruceAndrewBrucePeterGedeck.pdf>

# Contents
* [1 Exploratory Data Analysis](#1-exploratory-data-analysis)
* [2 Data and Sampling Distributions](#2-data-and-sampling-distributions)
  * [Key Terms](#key-terms)
  * [When Does Size Matter?](#when-does-size-matter)
  * [Selection Bias](#selection-bias)
  * [Central Limit Theorem](#central-limit-theorem)
  * [The Bootstrap](#the-bootstrap)

## 1 Exploratory Data Analysis


## 2 Data and Sampling Distributions
### Key Terms
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
  
### When Does Size Matter?
Sometimes smaller amounts of data is better. Time and effort spent on random sampling not only reduces bias but also allows greater attention to data exploration and data quality. It might be prohibitively expensive to track down missing values or evaluate outliers (which may contain useful information) in millions of records, but doing to in a sample of several thousand records may be feasible and efficient.

**When are massive amounts of data necessary?**
One classic scenario for the value of big data is when the data is not only **big** but also **sparse** as well. Only when such enormous quantities of data are accumulated can effective search results be returned for most queries when considering the search queries reveived by Google.

### Selection Bias
Bias resulting from the way in which observations are selected and the samples cannot fully represent the whole population. For example, when analyzing which web design is more attractive to customers, we cannot decide which customer view which version of web, and such a self-selection lead to imbalance groups of samples.

### Central Limit Theorem
In many situations, when independent random variables are summed up, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed. 

### The Bootstrap
One easy and effective way to estimate the sampling distribution of a model parameters, is to draw additional samples, with replacement, from the sample itself and recalculate the model for each resample. This procedure is called **_Bootstrap_**.

Such a procedure does not involve any assumptions about the data or the sample statistic being normally distributed. For a sample of *n*, steps are sa follow:
1. Draw a sample value, record it, and then replace it.
2. Repeat *n* times.
3. Record the mean of the *n* resampled values.
4. Repeat steps 1-3 M times. (M could be 100, 1000, etc.)
5. Use the M results to:
   * Calculate their standard deviation (this estimates sample mean standard error).
   * Produce a histogram or boxplot.
   * Find a confidence interval.

The bootstrap is a powerful tool for assessing the variability of a sample statistic. It can be applied in a wide variety of circumstances without extensive study of mathematical approximations of sampling distributions. It also allows us to estimate sampling distributions for statistics where no mathematical approximation has been developed. When applied to predictive models, aggregating multiple boostrap sample predictions (bagging) outperforms the use of a single model.

### Confidence Intervals
* **_Confidence Level_**  
  The percentage of confidence intervals, constructed in the same way from the same population, that are expected to contain the statistic of interest. Given a 90% confidence interval, it covers the central 90% of the bootstrap sampling distribution of a sample statistic. More generally, an 90% confidence interval should contain similar sample estimates 90% of the time on average in similar sampling procedure.

  *Figure. Boostrap confidence interval for the annual income of loan applications, based on a sample of 20*
<img width="400" alt="Figure 2-9" src="https://user-images.githubusercontent.com/91806768/154370693-2f438d49-1b91-444b-a4f9-1acfac822edf.png">

* **_Interval Endpoints_**  
  The top and bottom of the confidence interval.

### Normal Distribution
