# Nonparametric Statistics Review

## Concepts of nonparametric statistics

**When to use?**

Data doesn't meet certain assumptions, especially the normality assumption.

Data distribution is peculiar, uncertain or no prior knowledge about it.

Many outliers may affect analysis results.

Small sample size.

**What is nonparametric statistics?**

Gibbons (1985) : If one of following criteria is satisfied, then it is called nonparametric: 

1. The data are count data of number of observations in each category
   (or cross-category) (e.g., numbers of successes in treating 20 cancer patients with each of k = 6 therapies ) 

2. The data are nominal scale data  (e.g., hair color)
3. The data are ordinal scale data  (e.g. pain scale 1 to 10)
4. The inference does not concern a parameter  (e.g., whether a set of numbers ~ U(0,1))
5. The assumptions are general rather than specific  (the assumption of a continuous population distribution)

Remark: 

The key of nonparametric concepts is "<u>Distribution Free</u>", which refers to <u>free population distribution</u> rather than <u>statistics distribution</u>! Statistics distribution is always required for statistical tests. 

**Pros of nonparametric methods relative to parametric methods.**

Robust!!!

## Fundamentals

### Permutation Test

**How to choose a permutation statistics?**

1. *Key* : Increase/decrease <u>monotonically</u> with respect to difference between H0 and H1 grows larger.
2. Not necessary to be pivotal statistics.
3. Linear transformation of test statistics does not change test results (p-value) and power since it is fully dependent about rank of statistics. For instance, if we standardize the statistics, old results still hold.
4. If so, then what will change test results? H0!

**Assumption**:

1. i.i.d. :  Stronger assumption. Independence is not always easy to satisfy.

2. Exchangeablity : Weaker assumption : $X_1, ..., X_n \sim same distributed with \sim X_{a1} ,..., X_{an}$

   Remark: iid $\Rightarrow$ exchangeability, exchangeability $\\nRightarrow$ iid 
   $$
   \text{e.g.} (x,y) \sim N(\mu_x,\mu_y,\sigma_x^2,\sigma_y^2,\rho) \Rightarrow pdf ：f(x,y)=f(y,x)\text{ but x depends on y}
   $$

3. 

**Procedure:**

Example: Whether $x_i \in P$  and $y_i \in Q$ has correlation?

![image-20220104161005507](C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104161005507.png)

1. H0 determines how to shuffle:

   Fix P and shuffle Q (or reversely)

2. Shuffles produce 20! samples. Here we regard permutation sample's histogram as real population distribution under H0.

3. Construct statistics: monotonic function of random variables

   <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104161439286.png" alt="image-20220104161439286" style="zoom: 50%;" />

4. *Optional: draw a random sample of size 50,000 out of 20!

5.  Compute P-value: 

   Compute observed $r_{obs}$ for the original sample.

   Count frequence of samples whose $r_{xy}$ is larger than $r_{obs}$ / Random sample size

   P-value = frequency of $r_{xy}  \geqq r_{obs}$

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104162743157.png" alt="image-20220104162743157" style="zoom:33%;" />

**Exact test**

Permutation test is an exact test which does <u>no asymptotic approximation</u>, thus gives exact significance level since we already know the actual population distribution under H0.

Note that randomization test is a little bit difference from permutation test: randomization refers to <u>drawing a small sample out of all permutation possibilities</u>, which leads to inexact significance level. 

- [ ] **Size distortion** w.r.t t test in linear regression



### Binomial Test

**Target problems** (Note that binomial refers to statistics distribution rather than population distribution)

Whether some population proportion is p?

**Assumption**

1. i.i.d.
2. Probability of given outcome is the same for all n samples.

**Procedure** (e.g.)
$$
H_0 : p = p_0 \leftrightarrow H_1 : p> p_0
$$
<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104171717997.png" alt="image-20220104171717997" style="zoom:20%;" />

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104171749105.png" alt="image-20220104171749105" style="zoom:20%;" />

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104171852473.png" alt="image-20220104171852473" style="zoom:20%;" />

R code:

```R
binom.test(x,n,p=0.5,alternative=c("two.sided","less","greater"),conf.level=0.95)
```

**Pros and Cons compared to z-test**

|               | Pros                                                         | Cons                                                         |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Binomial Test | 1. Small sample size <br/>2. Robust<br/>3. No symmetric or normal assumption<br/>4. Any proportion unequal to 0.5! | 1. Computational burden <br/>2. Only 1-sided test<br/>3. Less Powerful:<br/>Discrete p-values causing actual rejection rate below nominal significance level |
| Z Test        | 1. Computational convenience<br/>2. 2-sided<br/>3. Easy to estimate Power<br/>4. Powerful | 1. Population needs to distributed on $R$ <br/>2. Normal assumption or Large sample size <br/>3. Only caters to p=0.5 tests |



### Order Statistics and Ranks

Extract information we needed of a sample : Order. Discard magnitude info.

**Order statistics**

*<u>Definition:</u>* Let 𝑋1, ⋯ , 𝑋n be an independent sample from a population with absolutely continuous c.d.f. 𝐹 and p.d.f. 𝑓.
The continuity of 𝐹 implies that 𝑃 (𝑋i = 𝑋j) = 0, when 𝑖 ≠ 𝑗 and the same could be ordered with strict inequalities.
$$
𝑋_{(1)} < 𝑋_{(2)} < ⋯ < 𝑋_{(n)} \quad X_{(i)} \text{  is called the i-th order statistic}
$$



 *<u>Commonly used order statistics:</u>* 
$$
X_{(1)},X_{(n)}, \text{ sample median} =  \left\{ \begin{array}{rcl} 
X_{[(n+1)/2]} & \mbox{for odd n}
\\ [X_{[n/2]} + X_{[n/2]+1}] & \mbox{for even n} \\
\end{array}\right.
$$
Distribution

### Empirical Distribution Function and Quantile Function

### Efficiency of Nonparametric Procedures

