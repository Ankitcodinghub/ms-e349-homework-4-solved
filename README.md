# ms-e349-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [MS&E349 Homework 4 Solved](https://www.ankitcodinghub.com/product/mse-349-homework-4-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118676&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;MS\u0026amp;E349 Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Please submit one homework assignment per group. The homework solution should be submitted online to Canvas. Please indicate clearly the names of all group members. I prefer that solutions are typed in Latex, but it is also fine to submit scanned copies of handwritten solutions. Include the commented code in an appendix section. Please also submit the executable and commented code.

Question 1 Machine Learning in Finance

In this question, we are going to study different machine learning methods to predict monthly S&amp;P 500 index returns with financial variables. (Welch and Goyal(2007)) examines the predictability of these characteristics on the equity premium. The data is available on Canvas. All methods are implemented in the scikit package in Python, i.e. you just need to run the commands. Our goal is to estimate the conditional mean of the index return. We assume that the conditional mean is a function of parameters zt:

rt+1 = Et[rt+1] + εt+1

where

Et[rt+1] = g∗(zt)

and zt ∈ Rp is a p-dimensional predictor vector.

Sample splitting and tuning: The following machine learning methods rely on a choice of hyperparameters to control model complexity and avoid overfitting. Hyperparameters include, for example, the penalization parameters in LASSO and elastic net, the number of iterated trees in boosting, the number of random trees in a forest, and the depth of the trees. In most cases, there is little theoretical guidance for how to “tune” hyperparameters for optimized out-of-sample performance.

We follow the most common approach in the literature and select tuning parameters adaptively from the data in a validation sample. In particular, we divide our sample into three disjoint time periods that maintain the temporal ordering of the data.

1. The first, or training subsample is used to estimate the model subject to a specific set of tuning parameter values.

2. The second, or validation subsample is used for tuning the hyperparameters.

3. The third or testing subsample is used for neither estimation nor tuning, is truly out-of-sample and thus is used to evaluate a method’s predictive performance.

,

where ˆyi is the predicted yi.

(b) R-squared R2:

,

where the unconditional mean ¯ is only estimated on the training data and used on the training, validation and test data set.

Report the two measures for the training, validation and test data for each of the following methods and briefly interpret your results.

1. Linear Model: The simple linear model imposes that conditional expectations g∗(zt) can be approximated by a linear function of the raw predictor variables and the parameter vector, θ,

g(zt;θ) = zt&gt;θ

We estimate θ by minimizing the standard least squares, or `2, objective function

(1)

There is no hyperparameter in the objective function (1), so you can estimate θ in the training set and report the MSE and R2 for the training, validation and test data. Hint: You can use

sklearn.linear_model.LinearRegression()

More details can be found https://scikit-learn.org/stable/modules/generated/sklearn. linear_model.LinearRegression.html

2. Penalized Linear Model: The penalized linear model also imposes that conditional expectations g∗(zt) can be approximated by a linear function of the raw predictor variables and the vector θ,

g(zt;θ) = zt&gt;θ

However, the penalized methods differ by incorporating a new term in the loss function:

,

Loss Function Penalty

where L(θ) is defined in (1). There are several choices for the penalty function φ(·). We focus on the popular “elastic net” penalty, which takes the form

.

The elastic net involves two non-negative hyperparameters, λ and ρ, and includes two well known regularizers as special cases. The ρ = 0 case corresponds to the LASSO and uses an absolute value, or `1, parameter penalization. LASSO sets coefficients on a subset of covariates to exactly zero. In this sense, the LASSO imposes sparsity on the specification and can thus be thought of as a variable selection method. The ρ = 1 case corresponds to the to ridge regression, which uses an `2 parameter penalization, that draws all coefficient estimates closer to zero but does not impose exact zeros anywhere. In this sense, ridge is a shrinkage method that helps prevent coefficients from becoming unduly large in magnitude. For intermediate values of ρ, the elastic net encourages simple models through both shrinkage and selection.

In this problem,

(a) Let ρ = 0, find the optimal λ from the “cross validation” set and report the MSE and R2 in the training, cross validation and test sets. under the optimal λ

More details can be found https://scikit-learn.org/stable/modules/generated/ sklearn.ensemble.RandomForestRegressor.html Hint: You can use sklearn.linear_model.Lasso() and hyperparamter λ corresponds to the α argument. More details can be found https:

//scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

(b) Let ρ = 1, find the optimal λ and report the MSE and R2 as part 1. Hint: You can use sklearn.linear_model.Ridge() and hyperparameter λ corresponds to the α argument. More details can be found https:

//scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

(c) Find the optimal λ and ρ (0 ≤ ρ ≤ 1) and report the estimation errors as part 1. Hint: You can use sklearn.linear_model.ElasticNet() and hyperparameters λ and ρ corresponds to the α and l1 ratio arguments. More details can be found https://scikit-learn.org/stable/modules/generated/sklearn.linear_ model.ElasticNet.html

3. Principle Component Regression: Principle Component Regression (PCR) consists of a two-step procedure. In the first step, principal components analysis (PCA) combines regressors into a small set of linear combinations that best preserve the covariance structure among the predictors. In the second step, a few leading components are used in standard predictive regression. That is, PCR regularizes the prediction problem by zeroing out coefficients on higher order components.

In particular, we stack rt+1 as a T × 1 vector R, zt as a T × p matrix Z and et+1 as a T × 1 vector E. Then we can write linear regression rt+1 = zt&gt;θ + εt+1 in a matrix form

R = ZΘ + E

The forecasting method for PCR is written as

R = (ZWK)θK + E, (2)

h i where WK = w1 w2 ··· wK . WK are the right singular vectors of matrix Z corresponding to the top K singular values.

In this problem, find the optimal K from the cross validation set and report the MSE and R2 in the training, cross validation and test sets. Hint: you can use

sklearn.decomposition.PCA()

4. Partial Least Squares: The forecasting method for Partial Least Squares (PLS) can also be written as

R = (ZWK)θK + E, (3)

WK is estimated from wj = argmaxCov2(R,Zw), w&gt;w = 1, Cov(Zw,Zwl) = 0, l = 1,2,··· ,j − 1.

w

In this problem, find the optimal K from the cross validation set and report the estimation errors on the training and test sets. Hint: You can use

sklearn.cross_decomposition.PLSRegression()

//www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html

5. Regression Tree: Regression trees are a popular machine learning approach for incorporating multi-way predictor interactions. Unlike linear models, trees are fully nonparametric and possess a logic that departs markedly from traditional regressions. A tree “grow” in a sequence of steps. At each step, a new “branch” sorts the data leftover from the preceding step into bins based on one of the predictor variables. This sequential branching slices the space of predictors into rectangular partitions, and approximates the unknown function g∗(·) with the average value of the outcome variable within each partition.

Formally, the prediction of a tree, T , with K “leaves” (terminal nodes), and depth L, can be written as

K

g(zt;θ,K,L) = Xθ1(zt ∈ Ck(L)),

k=1

where Ck(L) is one of the K partitions of the data. Each partition is a product of up to L indicator functions of the predictors. The constant associated with partition partition k (denoted θk) is defined to be the sample average of outcomes within the partition.

The basic idea to estimate the tree is to myopically optimize forecast error at the start of each branch. At each new level, we choose a sorting variable from the set of predictors and the split value to maximize the discrepancy among average outcomes in each bin. The loss associated with the forecast error for a branch C is often called “impurity” We choose the most popular `2 impurity for each branch of the tree:

,

where |C| denotes the number of observations in set C. Given C, it is clear that the optimal choice of θ: θ = |C1| Pzt∈C rt+1. The procedure is equivalent to finding the branch C that locally minimizes the impurity. Branching halts when the number of leaves or the depth of the tree reach a pre-specified threshold that can be selected adaptively using a validation sample.

In this problem, find the optimal number of leaves K and depth L from the cross validation set and report the MSE and R2 in the training, cross validation and test sets. Hint: You can use

sklearn.tree.DecisionTreeRegressor()

and the hyperparameters L and K correspond to the max depth and max leaf nodes arguments. More details can be found on https://scikit-learn.org/stable/modules/ generated/sklearn.tree.DecisionTreeRegressor.html

The Boosted regression trees starts by fitting a shallow tree (e.g., with depth L = 1). This over-simplified tree is sure to be a weak predictor with large bias in the training sample. Next, a second simple tree (with the same shallow depth L) is used to fit the prediction residuals from the first tree. Forecasts from these two trees are added together to form an ensemble prediction of the outcome, but the forecast component from the second tree is shrunken by a factor ν ∈ (0,1) to help prevent the model from overfitting the residuals. At each new step b, a shallow tree is fitted to the residuals from the model with b − 1 trees, and its residual forecast is added to the total with a shrinkage weight of ν. This is iterated until there are a total of B trees in the ensemble. The final output is therefore an additive model of shallow trees with three tuning parameters (L,ν,B) which we adaptively choose in the validation step.

In this problem, find the optimal (L,ν,B) from the cross validation set and report the MSE and R2 in the training, cross validation and test sets. Hint: You can use

sklearn.ensemble.GradientBoostingRegressor()

and the hyperparameters L, ν and B correspond to the max depth, learning rate and n estimators arguments. More details can be found on https://scikit-learn.org/stable/ modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

7. Random Forests: A random forest is an ensemble method that combines forecasts from many different trees. It is a variation on a more general procedure known as bootstrap aggregation, or “bagging.” The baseline tree bagging procedure draws B different bootstrap samples of the data, fits a separate regression tree to each, then averages their forecasts. Trees for individual bootstrap samples tend to be deep and overfit, making their individual predictions inefficiently variable. Averaging over multiple predictions reduces this variation, thus stabilizing the trees’ predictive performance.

In this problem, find the optimal depth of tree L and number of bootstrap samples B from the cross validation set and report the MSE and R2 in the training, cross validation and test sets.

Hint: You can use

sklearn.ensemble.RandomForestRegressor()

and the hyperparameters L and B correspond to the max depth and n estimators arguments. More details can be found on https://scikit-learn.org/stable/modules/generated/ sklearn.ensemble.RandomForestRegressor.html

b/m Book Value tbl Treasury Bills

AAA AAA-rated Corporate Bond Yields BAA BAA-rated Corporate Bond Yields

lty Long Term Yield ntis Net Equity Expansion

Rfree Risk-free Rate infl Inflation

ltr Long Term Rate of Returns corpr Corporate Bond Returns

svar Stock Variance d/p Dividend Price Ratio

d/y Dividend Lagged Price Ratio e/p Earning to Price Ratio

d/e Dividend Payout Ratio ret Index Return

Table 1: List of financial variables

………

References

Ivo Welch and Amit Goyal. A comprehensive look at the empirical performance of equity premium prediction. The Review of Financial Studies, 21(4):1455–1508, 2007.
