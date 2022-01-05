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
*<u>Distribution:</u>* (mariginal)

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104193601044.png" alt="image-20220104193601044" style="zoom:40%;" />

**Rank , Rank-order statistics & Rank statistics** 

<u>*Definition:*</u> 
$$
X_1,..,X_n \mbox{ is a random sample from continous F with order statistics: } X_{(1)},...,X_{(n)}
$$

1. Rank: $X_i$ has rank $R_i$ among $X_1, ⋯ , X_n$ if $𝑿𝒊 = 𝑿(𝑹𝒊)$ assuming the $R_i$ th order statistic is
   uniquely defined (i.e., ties are not possible, or $X_(i) \neq X_(j)$ for all $i \neq j$)  
2. Rank-order statistics : The ith rank-order statistic $r(X_i)$ is $R_i$, the rank of the ith observation $X_i$ in the original unordered sample.
3. Rank statistics : A statistic (such as 𝑉(𝑹)) that is a function of X1, ⋯ , 𝑋n only through the rank vector 𝑹 is called a rank statistics.

*<u>R code</u>*

```R
x = c(8,11,17,25,6)
rank(x) # returns a vector with each value's rank
# 2 3 4 5 1
order(x) # returns the indices that would put the initial vector x in order
# 5 1 2 3 4
x[order(x)]
# 6 8 11 17 25
```

*<u>Properties</u>*

1. Theorem: The rank vector is distributed uniformly over all permutation rank vectors.

   Let ℛ be the set of all permutations of the integers (1, 2, . . . , 𝑛) and 𝑹 = (𝑅1, 𝑅2, ⋯ , 𝑅n) be the vector of ranks. Then 𝑹 is uniformly distributed over ℛ. That is, 𝑃(𝑹 = 𝒓) = 1/𝑛! for each permutation 𝒓.

2. Corollary: 𝑟(𝑋i) is a discrete random variable. For a random sample from a continuous population, it follows the <u>discrete uniform distribution</u>, i.e., 𝑃 [ 𝑟(𝑋i) =j ] = 1/𝑛 for 𝑗 = 1, 2, . . . , 𝑛  

**Distribution-free**

The distribution of a statistics does not depend on sample's joint distribution at all but only depends on the sample observation itself. 

Formal def: Let 𝑌1, ⋯ , 𝑌n be random variables with joint distribution function D, where D is a member of some collection ℱ of possible joint distributions. The statistic 𝑇(𝑌1, ⋯ , 𝑌n) based on 𝑌1, ⋯ , 𝑌n is distribution-free over ℱ if the distribution of T is the same for every joint distribution in ℱ.

### Empirical Distribution Function and Quantile Function

**EDF**

*<u>Definition:</u>* Let $X_1,...,X_n$ be i.i.d. with CDF F(x). The empirical distribution function EDF is defined as 
$$
F_n(x) = \frac{1}{n} \Sigma_{i=1}^n 1(X_i \leq x)
$$

$$
nF_n(x) = \Sigma_{i=1}^n 1 (X_i \leq x)\sim B(n, F(x))
$$

*<u>Properties:</u>* 

1. The EDF is a cumulative distribution function

2. It is an unbiased estimator of the true CDF 
   $$
   E(F_n(x)) =F(x)
   $$

3. It is consistent and asymptotically normal
   $$
   F_n(x) \rightarrow p \rightarrow F(x)
   $$

4. EDF is the nonparametric MLE for CDF
   $$
   F_n(x) \mbox{ maximize } L(F|\bold{x}) = \Pi_{i=1}^n P_F(x_i)
   $$

5. $$
   Var(F_n(x)) = \frac{1}{n}F(x)(1-F(x))
   $$

6. Glivenko-Canteli Theorem:
   $$
   sup_x|F_n(x) - F(x)| \rightarrow a.s. \rightarrow 0
   $$

7. 

*<u>R code:</u>* 

```R
ecdf(x)
```

**KS distance**
$$
D_n = D_k(F_n,F) = sup_x|F_n(x) - F(x)| = max_i(max(|F(x_{(i)})-\frac{i-1}{n}|,|F(x_{(i)}) - \frac{i}{n}|))
$$
*<u>Distribution-free</u>*: KS distance is distribution free:

Proof:

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104210559547.png" alt="image-20220104210559547" style="zoom:40%;" />

*<u>Distribution:</u>*

Theorem: If F is any continuous distribution, then for every d > 0, 
$$
\mathop{lim}\limits_{n\rightarrow \infty} P(D_n \leq \frac{d}{\sqrt{n}}) = 1-2\sum_{i=1}^{\infty}(-1)^{i-1}e^{-2i^2d^2}
$$
Limiting distribution: (n >= 80)
$$
P(\sqrt{n}D_k(F_n,F_0)\leq x) \rightarrow H(x)=\mathop{sup}\limits_{t}|B(F(t))|
$$
We calculate P-value by above formula.

*<u>DKW Inequality</u>*:
$$
P(D_k(F_n,F_0)>\epsilon) \leq 2exp(-2n\epsilon^2)
$$
*<u>KS test & R code:</u>*

Test whether the sample follows pre-assumed distribution F(x).

```R
ks.test(x,pexp)
```

**Confidence Intervals & Confidence Band**:

*<u>CI:</u>* Point-wise
$$
P(F(x)\in C(x)) \geq 1-\alpha \mbox{, for  fixed }  x
$$
*<u>CB:</u>* Simultaneous
$$
P(F(x)\in C(x))\geq1-\alpha, \mbox{ for  } \forall x
$$
CB envelopes all CIs, so that CB is wider than all CIs.

**Empirical Quantile Function**
$$
F^{-1}_n(p) = inf\{y:F_n(y)\geq p\}
$$
CDF F is right continuous and quantile is left continuous.

### Efficiency of Nonparametric Procedures

Efficiency is a measure of quality of an estimator, of an experimental design, or of a hypothesis testing procedure.

We may compare tests by considering the <u>relative sample sizes</u>(n) necessary to achieve the <u>same power</u>(1-$\beta$) at the <u>same level</u>($\alpha$) against the same alternative( Under same H0 ).

Key: Fixed $\alpha \& \beta$

**Relative efficiency**: $\frac{n_1}{n_2}$

Relative efficiency is a function of alpha and beta. $f(\alpha, \beta)$

To achieve same power at same significance level under same H0, the smaller sample size means the larger efficiency.

**Asymptotic Relative Efficiency (Pitman ARE)**

<u>*Definition*</u>: Asymptotic relative efficiency is the limit of the relative efficiencies as the sample size grows.

<u>*Five regularity conditions:*</u>

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104214500626.png" alt="image-20220104214500626" style="zoom:50%;" />

<u>*Theorem:*</u> If $T_n^{(1)} \& T_n^{(2)}$ are two tests satisfying above 5 regularities, the ARE of them is :
$$
ARE(T_n^{(1)},T_n^{(2)}) = \mathop{lim}\limits_{n\rightarrow \infty}frac{e(T_n^{(1)})}{e(T_n^{(2)})} = \mathop{lim}\limits_{n\rightarrow \infty}[\frac{dE(T_n^{(1)})/d\theta|\theta = \theta_0}{dE(T_n^{(1)})/d\theta|\theta = \theta_0}]^2\frac{\sigma^2(T_n^{(2)})|\theta = \theta_0}{\sigma^2(T_n^{(1)})|\theta = \theta_0}
$$
Key: Know the distribution of statistics of two tests $T_n^{(1)} \& T_n^{(2)}$ and give their E($T_n^{(1)}$) form, then do limitation.

*<u>Example:</u>*
$$
ARE(\mbox{sign test}, \mbox{t test}) = \frac{2}{\pi}
$$
T test:

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104215428239.png" alt="image-20220104215428239" style="zoom:40%;" />

Sign test:

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220104215733761.png" alt="image-20220104215733761" style="zoom:55%;" />

## Location Inference for Single Samples (and Paired-Samples)

**Assumption:** 

*<u>Continuous r.v.</u>* X is a continuous r.v. with unique unknown median $\theta$

*<u>Discrete r.v.</u>* equal chance of X can be greater or less than median. 

Remark: Discrete r.v.'s median is different from 50% quantile.
$$
M = \frac{x_1+x_2}{2}
$$

$$
x_1 \mbox{ is the largest that satisfies } P(X \leq x_1) \leq 0.5\quad; x_2 \mbox{ is the smallest }P(X\leq x_2) \geq 0.5
$$

### Hypothesis Test Methods for Medians

#### 1. Sign test

**Assumption:**

Independent but not necessary from the same distribution.

No need for symmetry, normal or other assumptions about rank and magnitude, we only care about signs here.

**Essence:**

Binomial test with H0: $\theta = 0.5$

**Procedure:**

1. statistics k = min{count of +, count of -}
2. binom.test(k,n) (H0 refers to $\theta$ = 0.5)

#### 2. Wilcoxon test

**Assumption:**

i.i.d. + Symmetry

Sign and rank are needed, while magnitude is still neglected.

**Statistics: **sum of ranks

If 𝐻0 is true, we would expect that the distribution of positives and negatives to be <u>distributed at random among the ranks</u>.
$$
W  = S_+ \mbox{ or }S_- \mbox{ or } |S_+-S_-|
$$
Multiple choices of W, as long as it changes monotonically so the results are all the same.

**W's Distribution:**

1. <u>*Exact Distribution*</u>:

$$
P(W = w) = \frac{c(w)}{2^n}I_{w\in[0,\frac{n(n+1)}{2}]}
$$

$$
c(w) = \mbox{the possible allocations so that }W = S_+ = \sum_{x(R_i)>0}R_i = w \Leftarrow \mbox{use methods of permutation and combination}
$$

​		Exact distribution table (only affordable for small sample size) n=3:

​		<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105001016556.png" alt="image-20220105001016556" style="zoom:33%;" />

​		

2. *<u>Asymptotic Distribution:</u>* (CLT)

$$
W' =\frac{W-\frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}= \frac{\sum_{x(R_i)>0}R_i-\frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}} \sim N(0,1)
$$

*<u>R code:</u>*

```R
wilcox.test(data,paried = False, alternative = "two.sided","Exact")
```

​		<u>*Intractable Problems for Normality Approximation*</u>

1. Continuity Correction:

   ​	When using continuous distribution approximate discrete ones.

   ​	Replace P(X = n) with P(n-0.5<Y<n+0.5) where X is discrete and Y  is continuous.

2. Ties:

   <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105074335906.png" alt="image-20220105074335906" style="zoom:33%;" />

   When with ties, the pdf is heavily multimodal and discrete property is more evident, so that continuity correction is more important.

   When there are ties in the data, the variance of normal approximation would be decrease.

   How to deal with ties? -- Just ignore the ties, it does not change distribution of statistics *W* that much.

   Don't add random noise to ties! This will heavily affect the rank thus sabotage the results. 

### Confidence Intervals for Medians

! Caution: The CI is constructed for Median $\theta$, not statistics W or S+ !!!

#### 1. Sign test

**Percentile Method**

*<u>Quantiles are distributed according to the binomial distribution:</u>*

For a sample $(X_1,...,X_n)$, its order statistics are $(Y_1,...Y_n)$, denote $Y_0  \triangleq -\infty, Y_{n+1}  \triangleq + \infty$, for the p quantile $X_p$
$$
P(Y_k < X_p < Y_{k+1}) = \left( \begin{array}{c} n \\ k \end{array} \right)p^k(1-p)^{n-k}
$$
Exact :<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105080910105.png" alt="image-20220105080910105" style="zoom:40%;" />

<u>*Normal Approximation for Percentile Method*</u>

Condition: $np\geq 5 \& n(1-p) \geq 5$ (Because support of binomial distribution is positive but normal's support is ***R***)

$\mu = np, \sigma = \sqrt{np(1-p)}$

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105081423427.png" alt="image-20220105081423427" style="zoom:35%;" />

As shown above, the CI of normal approximation is :
$$
CI = [\mu \pm Z_{\alpha/2}\sigma]
$$
So the corresponding CI in original sample is :
$$
[\lceil \mu - Z_{\alpha/2}\sigma\ \rceil ,\lfloor\mu+Z_{\alpha/2}\sigma\rfloor+1]
$$
(left ceiling and right floor)

**Trial and Error**

![image-20220105091502746](C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105091502746.png)

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105093428416.png" alt="image-20220105093428416" style="zoom:40%;" />

```R
x <- c(-2, 4, 8, 25, -5, 16, 3, 1, 12, 17, 20, 9)
x <- sort(x)
n <- length(x)
conf <- 1 - 2 * pbinom(0:(n/2), n, .5)
names(conf) <- 1:(n/2 + 1)
conf
# 1 2 3 4 5 6 7
# 0.9995117 0.9936523 0.9614258 0.8540039 0.6123047 0.2255859 -0.2255859
```

CI is the third candidate interval: [3rd, 10th] --> [1,17] in x

#### 2. Wilcoxon test

**Trial and Error**
$$
W' =\frac{W-\frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}} \sim N(0,1)
$$

$$
P(W \leq w_{\alpha/2}) = \alpha/2
$$

Here, $w_{\alpha/2}$ is the critical value for level of $\alpha$, two-sided test.

Trial and error long the real axis to find two bounds $\theta_1 < \theta <\theta_2$ and calculate corresponding wilcoxon statistics $W_1(X_i-\theta_1),  W_2(X_i-\theta_2)$, respectively.

s.t. statistics $W_1, W_2$ are slightly larger than $w_{\alpha/2}$, which means $W_1 > w_{\alpha/2}, W_2>w_{\alpha/2}$ and can't reject H0 of $\theta_1,\theta_2$

Remark: For sign test, $\theta$s between two adjacent data points (sorted) have same results because only considering sign. But for Wilcoxon test, above conclusion doesn't hold anymore since rank matters here!

**Walsh Average:**

*<u>Definition:</u>* For random variables $X_1,...,X_n$ and any $ i\leq j \in {1,...,n}$, the average $\frac{X_i+X_j}{2}$ of $X_i, X_j$ is called Walsh average.

More general way than simple trial and error. It is easy to apply and yields unique CI.

*<u>Key: Arrange all two-point pairs as transformed data, then do ordinary sign-test CI on its order statistics</u>*

1. Original observations: $x_1, x_2, ... x_n$

   ---> transformation --->  two-point pairs:

   $\frac{n(n+1)}{2}$ Transformed observations: $\frac{x_i+x_j}{2}$ 

2. New sign test statistics:

$$
S_+ = \sum I_{x_i+x_j-2\theta_0>0}
$$

$$
S_- = \sum I_{x_i+x_j-2\theta_0<0}
$$

3. Arrange transformed observations into their order statistics:
   $$
   W_{(1)},...,W_{(\frac{n(n+1)}{2})}
   $$

4. $100(1-\alpha)$% CI:
   $$
   [W_{(w_{\alpha/2} + 1)}, W_{(\frac{n(n+1)}{2}-w_{\alpha/2})}]
   $$

*<u>R code:</u>*

```R
wilcox.test(data, conf.int = TRUE)
```

Remark:

If assumptions such as symmetry are right, than when increasing sample size, the power of tests will generally improves thus yield shorter confidence interval. If increasing sample size does not yield shorter CI, it might imply that assumptions may be violated and needed to check.

### Point Estimation for Medians

**Pseudomedian** (i.e., Hodges-Lehmann Estimator) Practical!

Median of Walsh average $W_{(1)},...,W_{(\frac{n(n+1)}{2})}$

*<u>Pros:</u>* In the premise of robustness, HL estimator has highest efficiency!

1. A robust and nonparametric estimator; consistent, unbiased, and highly efficient for the most continuous symmetric distributions used in practice.
2. Compared to median: While the median is preferred with nonsymmetric populations, it requires far more observations than the mean to obtain the same level of precision. By contrast, the Hodges–Lehmann estimator has a higher ARE for similar data.

<u>*ARE of median, mean and pseudomedian (HL estimator):*</u>
$$
mean:\bar{X_n} \stackrel{d}\longrightarrow N(\theta, \frac{\sigma^2}{n})
$$

$$
median: Med_n\stackrel{d}\longrightarrow N(\theta, \frac{1}{4[f(\theta)]^2n})
$$

$$
Hodges-Lehmann: HL_n\stackrel{d}\longrightarrow N(\theta, \frac{1}{12[\scriptstyle \int f^2(x)\,dx]^2n})
$$

1. median v.s. mean: ARE(Med, mean) = 0.64
2. HL v.s. mean: ARE(HL, median) = 0.96

### CI of proportion p in Sign Test

*<u>Trial and Error</u>*

1. Clopper-Pearson Interval

   <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105150802639.png" alt="image-20220105150802639" style="zoom: 50%;" />
   $$
   1-\alpha \mbox{ CI: } S_{\leq} \cap S_{\geq} = (Beta_{\alpha/2}(x,n-x+1),Beta_{1-\alpha/2}(x+1,n-x))
   $$
   

2. Wilson Score Interval
   $$
   \frac{X-np}{\sqrt{np(1-p)}} \stackrel{d}\longrightarrow N(0,1) \Rightarrow \frac{\hat p -p}{\sqrt{\frac{p(1-p)}{n}}} \stackrel{d}\longrightarrow N(0,1)
   $$

   $$
   \mbox{CI for normal statistics Z: }p-\hat{p} = \pm z_{\alpha/2}\sqrt{\frac{p(1-p)}{n}}
   $$

   Solve the quadratic equation: $ z=z_{\alpha/2} $
   $$
   p = \frac{\hat{p}+\frac{z^2}{2n}}{1+\frac{z^2}{n}}\pm\frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}
   $$

### Alternative Scores

**Pitman Test** (Fisher-Pitman Test/Raw Data Test)

<u>*Assumption:*</u> Symmetry

*<u>Score:</u>* Signed Deviations

*<u>H0: positive deviations = negative deviations</u>*

<u>*H1: positive deviations >/< negative deviations*</u>

<u>*Procedure:*</u>

1. Do a permutation test for sign based on S+.
2. Under H0, sign of each deviation is equally like.
3. There are $2^n$ possibilities for sign allocation to n absolute values of deviations.

<u>*P-value*</u>

Exact:
$$
P-value = \frac{\#\{\mbox{Allocation of signs less likely to appear under H0}(e.g.,S_+\leq S_{+obs}) \}}{2^n}
$$


*<u>Pitman statistics</u>*: $S_+,S_-,S_d = |S_+-S_-|$

Remark: pitman statistics has one-to-one ordered relationship with t statistics, but rejection region is different thus is not equivellent.

**Inverse Normal Scores** (Van der Waerden Scores)

Converts the ranks to quantiles of the standard normal distribution.

<u>*Pros:*</u> Provides high efficiency when the normality assumptions are in fact satisfied, and the robustness of the non-parametric test when the normality assumptions are not satisfied.

### Comparison: Robustness v.s. Efficiency

Generally Speaking,

<u>*Robustness:*</u> Wilcoxon & Sign > Pitman & t test (Unless real population distribution is normal in which case t test is the best)

<u>*Efficiency:*</u> Compute ARE, and ARE depends on real population distribution. No general conclusion.

## Other Single-sample Inferences

### Goodness-of-Fit tests

*<u>Always remember QQ plot!</u>*

#### Kolmogorov-Smirnov test

Test whether observations are samples from a certain distribution.

*<u>Statistics:</u>*
$$
D_n = D_k(F_n,F) = \mathop{sup}\limits_{x}|F_n(x) - F(x)| = \mathop{max}\limits_{i}(max(|F(x(i)) - \frac{i-1}{n}|,|F(x(i)) - \frac{i}{n}|))
$$
$D_n$ is distribution-free

*<u>R code:</u>*

```R
ks.test(x,pdf,min, max, exact)
```

<u>*Exact Distribution:*</u>

 <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105163443521.png" alt="image-20220105163443521" style="zoom:33%;" />

<u>*Asymptotic Distribution:*</u>

Regarding $F_n(x)$ as a function-valued random variable.
$$
\sqrt n [F_n(x)-F(x)]\stackrel{d}\longrightarrow B(F(x))
$$
Where,
$$
B(t) \triangleq (W_t|W_T = 0), t\in[0,T] \mbox{ is a Gaussian stochastic process called the Brownian bridge}
$$
<u>*Positive KS Distance:*</u> If F is any continuous distribution, then for every d>0,
$$
\mathop{P}\limits_{n\rightarrow \infty}(D_n^+ \leq \frac{d}{\sqrt n}) = 1 - e^{-2d^2}
$$
<u>*Procedure:*</u>

5 numbers: 0.44, 0.81, 0.14, 0.05, 0.93 , test whether they follow U(0,1)

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105164617290.png" alt="image-20220105164617290" style="zoom:50%;" />

#### Normality test

1. For fixed $\mu, \sigma^2$, use Kolmogorov's test

2. If don't have any prior knowledge about $\mu, \sigma^2$, simply want to test whether they follow a normal distribution.'

   **Lilliefors Test** (K-S D test)

   1. Estimate population mean and standard deviation by trivial methods (not MLE! containing prior preference to a certain distribution)
   2. ks.test(standardized sample cdf, N(0,1))
   3. Compute K-S statistics $D_n$
   4. P-value: For p less than 0.1, n between 5 and 100: Dallal-Wilkinson approxiamation: 

$$
exp (−7.01256 d_n^2(𝑛 + 2.78019) + 2.99587𝑑_𝑛 (n+ 2.78019)^{1/2}− .122119 + .974598/\sqrt𝑛 + 1.67997/𝑛)
$$

​								Otherwise: modified statistic
$$
D^* = D_n(\sqrt n - 0.01 + 0.85/\sqrt n)
$$
​	<u>*R code:*</u>

```R
library(nortest)
lillie.test(x)
```

​		**Shapiro-Wilk Test**
$$
Statistics: W = \frac{(\sum_{i=1}^na_ix_{(i)})^2}{\sum_{i=1}^n(x_i-\bar x)^2}
$$
The test is based on the correlation between the data and the corresponding normal scores. If the data are normal, the correlation should be close to 1.

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105170137906.png" alt="image-20220105170137906" style="zoom:33%;" />

```R
shapiro.test(x)
```

#### Pearson chi-squared test

Mainly for discrete distribution.

Statistics:
$$
\frac{\sum_{i=1}^n(Observation - Expectation)^2}{Expectation} \sim \chi_{n-1}^2
$$
For unknown expected distribution parameters: MLE!

### Trend Test

#### Monotonicity

**Cox-Stuart test**

1. Transformation: The second half minus the first half

$$
x_{m+1} - x_1,...,x_{2m} - x_m
$$

When odd number encountering, omit the middle $x_{m+1}$

2. Sign test for above m minus results.

**A specific trend**

Rearrange data order and then run cox-stuart

#### Randomness

**Runs Test** 

statistics: R (number of runs)

<u>*Two categories*</u>  ( n is #{kind A}, m is #{kind B}, N = n+m ):

Exact
$$
\mbox{odd R: } P(R = 2s+1) = \frac{\left( \begin{array}{c} m-1 \\s-1 \end{array} \right)\left( \begin{array}{c} n-1 \\s \end{array} \right)+\left( \begin{array}{c} m-1 \\s \end{array} \right)\left( \begin{array}{c} n-1 \\s \end{array} \right)}{\left( \begin{array}{c} N \\M \end{array} \right)}
$$

$$
\mbox{even R: }P(R=2s) = 2 \frac{\left( \begin{array}{c} m-1 \\s-1 \end{array} \right)\left( \begin{array}{c} n-1 \\s-1 \end{array} \right)}{\left( \begin{array}{c} N \\ m \end{array} \right)}
$$

Asymptotic
$$
E(R) = 1+\frac{2nm}{N}, Var(R) = \frac{2nm(2nm-N)}{N^2(N-1)}
$$

$$
Z = \frac{R-E(R)}{\sqrt{Var(R)}} \stackrel{d}\longrightarrow N(0,1)
$$

*<u>Multi Categories</u>*

Asymptotic: $p_i = \frac{n_i}{N}$
$$
E(R) = N(1-\sum_{i=1}^np_i^2)+1, Var(R) = N[\sum_{i=1}^k(p_i^2-2p_i^3) + (\sum_{i=1}^np_i^2)^2]
$$


## Paired Samples

The only thing to note is that paired samples' difference may not be i.i.d. They are independent but not follows same distribution.

Single sample analysis for matched pairs.

**Location inference:**
$$
D = \mbox{X - Y}
$$
Assumption: D is symmetrically distributed, which leads to two possibilities: 

1. X and Y have same independent distribution
2. X and Y have different but both symmetric independent distribution with the same medians.

**Mcnemar test**

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105181005717.png" alt="image-20220105181005717" style="zoom:30%;" />
$$
S_+ = 9, S_- = 14 \Rightarrow n = 9+14 = 23, p = 9/23 = 0.4049
$$
Binomial test for B(9,23,0.4049), both exact test or approximation are fine.

## Two Independent Samples

### Do they have same centrality?

#### Wilcoxon-Mann-Whitney (WMW)

**Test:**

Null hypothesis: We expect a mix of low, medium and high ranks in each sample.

Alternative hypothesis: We expect lower ranks to dominate in one population and higher ranks in the other.

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105181540430.png" alt="image-20220105181540430" style="zoom:50%;" />

*<u>Procedure:</u>*
$$
H_0:P(X>Y) = 0.5 \leftrightarrow  H_1:P(X>Y) \neq0.5
$$

1. Mix Group A and Group B, and then sort them to get each sample's rank in the whole group. $S_x, S_y$

2. Take each sample size into account.
   $$
   U_x = S_x - n_1(n_1+1)/2
   $$

   $$
   U_y = S_y - n_2 ( n_2+1)/2
   $$

   Note that $U_x + U_y = n_1 n_2$ is fixed.
   $$
   U = min(U_x, U_y)
   $$
   Smaller U suggests more extreme deviations form H0.

3. $$
   T = \frac{U - n_1n_2/2}{\sqrt{n_1n_2(n_1+n_2+1)/12}}\stackrel{d}\longrightarrow N(0,1)
   $$

4. :star: Ties correction:
   $$
   \frac{U - n_1n_2/2}{\sqrt{\frac{n_1n_2}{12}[(n_1+n_2+1) - \frac{\sum(t_i^3-t_i)}{(n_1+n_2)(n_1+n_2+1)}]}}\stackrel{d}\longrightarrow N(0,1)
   $$
   Make up the decreasing variance when encountering many ties.

**Confidence Interval**

Walsh Average

**Point Estimation**

Median of $D_ij$ sequence

**Remark for WMW**

If use WMW to test medians, it only caters to rare cases when two populations only differing in location but not shape or scale. EDA is important here.

#### Kruskal-Waliis test (KW)

More than 2 samples.

*<u>Assumptions:</u>* 

1. The observations in the data set are independent of each other.
2. The observations must be drawn from the population by the process of random sampling.
3. The data with at least ordinally scaled characteristics must be available.
4. The distribution of the population should not be necessarily normal and the variances should not be necessarily equal.

*<u>Counterpart of ANOVA:</u>* 

All the values are combined together "Pool" and ranked into one series. 

Assess the differences against the average ranks in order to determine whether or not they are likely to have come from samples drawn from the same population.

<img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105183946305.png" alt="image-20220105183946305" style="zoom:50%;" />

### Do they have equal variance?

If two samples come from populations differing only in variance, the sample from the population with greater variance will be more spread out.

#### Siegel-Tukey test (centrality difference is known)

1. Align the samples by <u>subtracting the median difference</u> from all values in one sample.

   <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105184536942.png" alt="image-20220105184536942" style="zoom:50%;" />

2. Then arrange the combined samples in order and allocate rank 1 to the smallest observation, rank 2 to the largest, rank 3 to the next largest, ranks 4 and 5 to the next two smallest …  

   <img src="C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105184605758.png" alt="image-20220105184605758" style="zoom:50%;" />

3. WMW test:

   The sum of the ranks for each sample should be small if there were no difference in variance:
   $$
   S_A = 106 \Rightarrow U_A = S_A - \frac{m(m+1)}{2} = 70, U_B = mn - U_A = 26
   $$

### Do they come from same distribution?

#### Runs test

1. Suppose we have 𝑚 observations of the random variable 𝑋~𝐹(𝑥), and 𝑛 observations of the random variable 𝑌~𝐺(𝑦)  

2. H0: F(z) = G(z)

3. Combine the two sets of independent observations into one larger collection of 𝑁 = 𝑚 + 𝑛 observations, then arrange the observations in increasing order of magnitude.

![image-20220105185036972](C:\Users\daisy\AppData\Roaming\Typora\typora-user-images\image-20220105185036972.png)

#### Smirnov test

<u>*Statistics:*</u>
$$
D_{m,n} =\mathop{max}\limits_{x}|S_m(x)-S_n(x)|
$$
<u>*Asymptotic distribution:*</u>
$$
\mathop{lim}\limits_{m,n\stackrel{d}\rightarrow \infty} p(\sqrt{\frac{mn}{m+n}}D_{m,n}^+ \leq d) = 1-e^{-2d^2}
$$
