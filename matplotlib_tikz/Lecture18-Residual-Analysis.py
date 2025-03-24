# Converted from: Lecture18-Residual-Analysis.ipynb

# Markdown Cell 1
# <a href="https://www.quantrocket.com"><img alt="QuantRocket logo" src="https://www.quantrocket.com/assets/img/notebook-header-logo.png"></a>
# 
# © Copyright Quantopian Inc.<br>
# © Modifications Copyright QuantRocket LLC<br>
# Licensed under the [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).<br>
# <a href="https://www.quantrocket.com/disclaimer/">Disclaimer</a>

# Markdown Cell 2
# ***
# [Quant Finance Lectures (adapted Quantopian Lectures)](Introduction.ipynb) › Lecture 18 - Residual Analysis
# ***

# Markdown Cell 3
# # Residuals Analysis
# 
# By Chris Fenaroli and Max Margenot 

# Markdown Cell 4
# ## Linear Regression
# 
# Linear regression is one of our most fundamental modeling techniques. We use it to estimate a linear relationship between a set of independent variables $X_i$ and a dependent outcome variable $y$. Our model takes the form of:
# 
# $$ y_i = \beta_{0} 1 + \beta_{i, 1} x_{i, 1} + \dots + \beta_{i, p} x_{i, p} + \epsilon_i = x_i'\beta + \epsilon_i $$
# 
# For $i \in \{1, \dots, n\}$, where $n$ is the number of observations. We write this in vector form as:
# 
# $$ y = X\beta + \epsilon $$
# 
# Where $y$ is a $n \times 1$ vector, $X$ is a $n \times p$ matrix, $\beta$ is a $p \times 1$ vector of coefficients, and $\epsilon$ is a standard normal error term. Typically we call a model with $p = 1$ a simple linear regression and a model with $p > 1$ a multiple linear regression. More background information on regressions can be found in the lectures on [simple linear regression](Lecture12-Linear-Regression.ipynb) and [multiple linear regression](Lecture15-Multiple-Linear-Regression.ipynb).
# 
# Whenever we build a model, there will be gaps between what a model predicts and what is observed in the sample. The differences between these values are known as the residuals of the model and can be used to check for some of the basic assumptions that go into the model. The key assumptions to check for are:
# 
# * **Linear Fit:** The underlying relationship should be linear
# * **Homoscedastic:** The data should have no trend in the variance
# * **Independent and Identically Distributed:** The residuals of the regression should be independent and identically distributed (i.i.d.) and show no signs of serial correlation
# 
# We can use the residuals to help diagnose whether the relationship we have estimated is real or spurious.
# 
# Statistical error is a similar metric associated with regression analysis with one important difference: While residuals quantify the gap between a regression model predictions and the observed sample, statistical error is the difference between a regression model and the unobservable expected value. We use residuals in an attempt to estimate this error.

# Cell 5
# Import libraries
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Markdown Cell 6
# ### Simple Linear Regression
# 
# First we'll define a function that performs linear regression and plots the results.

# Cell 7
def linreg(X,Y):
    # Running the linear regression
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    B0 = model.params[0]
    B1 = model.params[1]
    X = X[:, 1]

    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * B1 + B0
    plt.scatter(X, Y, alpha=1) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=1);  # Add the regression line, colored in red
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    return model, B0, B1

# Markdown Cell 8
# Let's define a toy relationship between $X$ and $Y$ that we can model with a linear regression. Here we define the relationship and construct a model on it, drawing the determined line of best fit with the regression parameters.

# Cell 9
n = 50
X = np.random.randint(0, 100, n)
epsilon = np.random.normal(0, 1, n)

Y = 10 + 0.5 * X + epsilon

linreg(X,Y)[0];
print("Line of best fit: Y = {0} + {1}*X".format(linreg(X, Y)[1], linreg(X, Y)[2]))

# Markdown Cell 10
# This toy example has some generated noise, but all real data will also have noise. This is inherent in sampling from any sort of wild data-generating process. As a result, our line of best fit will never exactly fit the data (which is why it is only "best", not "perfect"). Having a model that fits every single observation that you have is a sure sign of overfitting.
# 
# For all fit models, there will be a difference between what the regression model predicts and what was observed, which is where residuals come in.

# Markdown Cell 11
# ## Residuals
# 
# The definition of a residual is the difference between what is observed in the sample and what is predicted by the regression. For any residual $r_i$, we express this as 
# 
# $$r_i = Y_i - \hat{Y_i}$$
# 
# Where $Y_i$ is the observed $Y$-value and $\hat{Y}_i$ is the predicted Y-value. We plot these differences on the following graph:

# Cell 12
model, B0, B1 = linreg(X,Y)

residuals = model.resid
plt.errorbar(X,Y,xerr=0,yerr=[abs(residuals),0*residuals],linestyle="None",color='Green');

# Markdown Cell 13
# We can pull the residuals directly out of the fit model.

# Cell 14
residuals = model.resid
print(residuals)

# Markdown Cell 15
# ### Diagnosing Residuals
# 
# Many of the assumptions that are necessary to have a valid linear regression model can be checked by identifying patterns in the residuals of that model. We can make a quick visual check by looking at the residual plot of a given model.
# 
# With a residual plot, we look at the predicted values of the model versus the residuals themselves. What we want to see is just a cloud of unrelated points, like so:

# Cell 16
plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');
plt.xlim([1,50]);

# Markdown Cell 17
# What we want is a fairly random distribution of residuals. The points should form no discernible pattern. This would indicate that a plain linear model is likely a good fit. If we see any sort of trend, this might indicate the presence of autocorrelation or heteroscedasticity in the model.

# Markdown Cell 18
# ## Appropriateness of a Linear Model
# 
# By looking for patterns in residual plots we can determine whether a linear model is appropriate in the first place. A plain linear regression would not be appropriate for an underlying relationship of the form:
# 
# $$Y = \beta_0 + \beta_1 X^2$$
# 
# as a linear function would not be able to fully explain the relationship between $X$ and $Y$.
# 
# If the relationship is not a good fit for a linear model, the residual plot will show a distinct pattern. In general, a residual plot of a linear regression on a non-linear relationship will show bias and be asymmetrical with respect to residual = 0 line while a residual plot of a linear regression on a linear relationship will be generally symmetrical over the residual = 0 axis.
# 
# As an example, let's consider a new relationship between the variables $X$ and $Y$ that incorporates a quadratic term.

# Cell 19
n = 50
X = np.random.randint(0, 50, n)
epsilon = np.random.normal(0, 1, n)
Y_nonlinear = 10 - X**1.2 + epsilon

model = sm.OLS(Y_nonlinear, sm.add_constant(X)).fit()
B0, B1 = model.params
residuals = model.resid

print('beta_0:', B0)
print('beta_1:', B1)
plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Markdown Cell 20
# The "inverted-U" shape shown by the residuals is a sign that a non-linear model might be a better fit than a linear one.

# Markdown Cell 21
# ## Heteroscedasticity
# 
# One of the main assumptions behind a linear regression is that the underlying data has a constant variance. If there are some parts of the data with a variance different from another part the data is not appropriate for a linear regression. **Heteroscedasticity** is a term that refers to data with non-constant variance, as opposed to homoscedasticity, when data has constant variance.
# 
# Significant heteroscedasticity invalidates linear regression results by biasing the standard error of the model. As a result, we can't trust the outcomes of significance tests and confidence intervals generated from the model and its parameters.
# 
# To avoid these consequences it is important to use residual plots to check for heteroscedasticity and adjust if necessary.
# 
# As an example of detecting and correcting heteroscedasticity, let's consider yet another relationship between $X$ and $Y$:

# Cell 22
n = 50
X = np.random.randint(0, 100, n)
epsilon = np.random.normal(0, 1, n)
Y_heteroscedastic = 100 + 2*X + epsilon*X

model = sm.OLS(Y_heteroscedastic, sm.add_constant(X)).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Markdown Cell 23
# Heteroscedasticity often manifests as this spread, giving us a tapered cloud in one direction or another. As we move along in the $x$-axis, the magnitudes of the residuals are clearly increasing. A linear regression is unable to explain this varying variability and the regression standard errors will be biased.
# 
# ### Statistical Methods for Detecting Heteroscedasticity
# 
# Generally, we  want to back up qualitative observations on a residual plot with a quantitative method. The residual plot led us to believe that the data might be heteroscedastic. Let's confirm that result with a statistical test.
# 
# A common way to test for the presence of heteroscedasticity is the Breusch-Pagan hypothesis test. It's good to combine the qualitative analysis of a residual plot with the quantitative analysis of at least one hypothesis test. We can add the White test as well, but for now we will use only Breush-Pagan to test our relationship above. A function exists in the `statsmodels` package called `het_breushpagan` that simplifies the computation:

# Cell 24
breusch_pagan_p = smd.het_breuschpagan(model.resid, model.model.exog)[1]
print(breusch_pagan_p)
if breusch_pagan_p > 0.05:
    print("The relationship is not heteroscedastic.")
if breusch_pagan_p < 0.05:
    print("The relationship is heteroscedastic.")

# Markdown Cell 25
# We set our confidence level at $\alpha = 0.05$, so a Breusch-Pagan p-value below $0.05$ tells us that the relationship is heteroscedastic. For more on hypothesis tests and interpreting p-values, refer to the lecture on hypothesis testing. Using a hypothesis test bears the risk of a false positive or a false negative, which is why it can be good to confirm with additional tests if we are skeptical.

# Markdown Cell 26
# ### Adjusting for Heteroscedasticity
# 
# If, after creating a residual plot and conducting tests, you believe you have heteroscedasticity, there are a number of methods you can use to attempt to adjust for it. The three we will focus on are differences analysis, log transformations, and Box-Cox transformations.

# Markdown Cell 27
# #### Differences Analysis
# 
# A differences analysis involves looking at the first-order differences between adjacent values. With this, we are looking at the changes from period to period of an independent variable rather than looking directly at its values. Often, by looking at the differences instead of the raw values, we can remove heteroscedasticity. We correct for it and can use the ensuing model on the differences.

# Cell 28
# Finding first-order differences in Y_heteroscedastic
Y_heteroscedastic_diff = np.diff(Y_heteroscedastic)

# Markdown Cell 29
# Now that we have stored the first-order differences of `Y_heteroscedastic` in `Y_heteroscedastic_diff` let's repeat the regression and residual plot to see if the heteroscedasticity is still present:

# Cell 30
model = sm.OLS(Y_heteroscedastic_diff, sm.add_constant(X[1:])).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Cell 31
breusch_pagan_p = smd.het_breuschpagan(residuals, model.model.exog)[1]
print(breusch_pagan_p)
if breusch_pagan_p > 0.05:
    print("The relationship is not heteroscedastic.")
if breusch_pagan_p < 0.05:
    print("The relationship is heteroscedastic.")

# Markdown Cell 32
# *Note: This new regression was conducted on the differences between data, and therefore the regression output must be back-transformed to reach a prediction in the original scale. Since we regressed the differences, we can add our predicted difference onto the original data to get our estimate:*
# 
# $$\hat{Y_i} = Y_{i-1} + \hat{Y}_{diff}$$

# Markdown Cell 33
# #### Logarithmic Transformation
# 
# Next, we apply a log transformation to the underlying data. A log transformation will bring residuals closer together and ideally remove heteroscedasticity. In many (though not all) cases, a log transformation is sufficient in stabilizing the variance of a relationship.

# Cell 34
# Taking the log of the previous data Y_heteroscedastic and saving it in Y_heteroscedastic_log
Y_heteroscedastic_log = np.log(Y_heteroscedastic)

# Markdown Cell 35
# Now that we have stored the log transformed version of `Y_heteroscedastic` in `Y_heteroscedastic_log` let's repeat the regression and residual plot to see if the heteroscedasticity is still present:

# Cell 36
model = sm.OLS(Y_heteroscedastic_log, sm.add_constant(X)).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Cell 37
# Running and interpreting a Breusch-Pagan test
breusch_pagan_p = smd.het_breuschpagan(residuals, model.model.exog)[1]
print(breusch_pagan_p)
if breusch_pagan_p > 0.05:
    print("The relationship is not heteroscedastic.")
if breusch_pagan_p < 0.05:
    print("The relationship is heteroscedastic.")

# Markdown Cell 38
# *Note: This new regression was conducted on the log of the original data. This means the scale has been altered and the regression estimates will lie on this transformed scale. To bring the estimates back to the original scale, you must back-transform the values using the inverse of the log:*
# 
# $$\hat{Y} = e^{\log(\hat{Y})}$$

# Markdown Cell 39
# #### Box-Cox Transformation
# 
# Finally, we examine the Box-Cox transformation. The Box-Cox transformation is a powerful method that will work on many types of heteroscedastic relationships. The process works by testing all values of $\lambda$ within the range $[-5, 5]$ to see which makes the output of the following equation closest to being normally distributed:
# $$
# Y^{(\lambda)} = \begin{cases}
#     \frac{Y^{\lambda}-1}{\lambda} & : \lambda \neq 0\\ \log{Y} & : \lambda = 0
# \end{cases}
# $$
# 
# The "best" $\lambda$ will be used to transform the series along the above function. Instead of having to do all of this manually, we can simply use the `scipy` function `boxcox`. We use this to adjust $Y$ and hopefully remove heteroscedasticity.
# 
# *Note: The Box-Cox transformation can only be used if all the data is positive* 

# Cell 40
# Finding a power transformation adjusted Y_heteroscedastic
Y_heteroscedastic_box_cox = stats.boxcox(Y_heteroscedastic)[0]

# Markdown Cell 41
# Now that we have stored the power transformed version of `Y_heteroscedastic` in `Y_heteroscedastic_prime` let's repeat the regression and residual plot to see if the heteroscedasticity is still present:

# Cell 42
model = sm.OLS(Y_heteroscedastic_box_cox, sm.add_constant(X)).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Cell 43
# Running and interpreting a Breusch-Pagan test
breusch_pagan_p = smd.het_breuschpagan(residuals, model.model.exog)[1]
print(breusch_pagan_p)
if breusch_pagan_p > 0.05:
    print("The relationship is not heteroscedastic.")
if breusch_pagan_p < 0.05:
    print("The relationship is heteroscedastic.")

# Markdown Cell 44
# *Note: Now that the relationship is not heteroscedastic, a linear regression is appropriate. However, because the data was power transformed, the regression estimates will be on a different scale than the original data. This is why it is important to remember to back-transform results using the inverse of the Box-Cox function:*
# 
# $$\hat{Y} = (Y^{(\lambda)}\lambda + 1)^{1/\lambda}$$

# Markdown Cell 45
# ### GARCH Modeling
# 
# Another approach to dealing with heteroscadasticity is through a GARCH (generalized autoregressive conditional heteroscedasticity) model. More information can be found in the lecture on GARCH modeling.

# Markdown Cell 46
# ## Residuals and Autocorrelation
# 
# Another assumption behind linear regressions is that the residuals are not autocorrelated. A series is autocorrelated when it is correlated with a delayed version of itself. An example of a potentially autocorrelated time series series would be daily high temperatures. Today's temperature gives you information on tomorrow's temperature with reasonable confidence (i.e. if it is 90 °F today, you can be very confident that it will not be below freezing tomorrow). A series of fair die rolls, however, would not be autocorrelated as seeing one roll gives you no information on what the next might be. Each roll is independent of the last.
# 
# In finance, stock prices are usually autocorrelated while stock returns are independent from one day to the next. We represent a time dependency on previous values like so:
# 
# $$Y_i = Y_{i-1} + \epsilon$$
# 
# If the residuals of a model are autocorrelated, you will be able to make predictions about adjacent residuals. In the case of $Y$, we know the data will be autocorrelated because we can make predictions based on adjacent residuals being close to one another.

# Cell 47
n = 50
X = np.linspace(0, n, n)
Y_autocorrelated = np.zeros(n)
Y_autocorrelated[0] = 50
for t in range(1, n):
    Y_autocorrelated[t] = Y_autocorrelated[t-1] + np.random.normal(0, 1) 

# Regressing X and Y_autocorrelated
model = sm.OLS(Y_autocorrelated, sm.add_constant(X)).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Markdown Cell 48
# Autocorrelation in the residuals in this example is not explicitly obvious, so our check is more to make absolutely certain.
# 
# ### Statistical Methods for Detecting Autocorrelation
# 
# As with all statistical properties, we require a statistical test to ultimately decide whether there is autocorrelation in our residuals or not. To this end, we use a Ljung-Box test.
# 
# A Ljung-Box test is used to detect autocorrelation in a time series. The Ljung-Box test examines autocorrelation at all lag intervals below a specified maximum and returns arrays containing the outputs for every tested lag interval.
# 
# Let's use the `acorr_ljungbox` function in `statsmodels` to test for autocorrelation in the residuals of our above model. We use a max lag interval of $10$, and see if any of the lags have significant autocorrelation:

# Cell 49
ljung_box = smd.acorr_ljungbox(residuals, lags=10, return_df=True)
print("Lagrange Multiplier Statistics:", ljung_box.lb_stat.tolist())
print("\nP-values:", ljung_box.lb_pvalue.tolist(), "\n")

if (ljung_box.lb_pvalue < 0.05).any():
    print("The residuals are autocorrelated.")
else:
    print("The residuals are not autocorrelated.")

# Markdown Cell 50
# Because the Ljung-Box test yielded a p-value below $0.05$ for at least one lag interval, we can conclude that the residuals of our model are autocorrelated.

# Markdown Cell 51
# ## Adjusting for Autocorrelation
# 
# We can adjust for autocorrelation in many of the same ways that we adjust for heteroscedasticity. Let's see if a model on the first-order differences of $Y$ has autocorrelated residuals:

# Cell 52
# Finding first-order differences in Y_autocorrelated
Y_autocorrelated_diff = np.diff(Y_autocorrelated)

# Cell 53
model = sm.OLS(Y_autocorrelated_diff, sm.add_constant(X[1:])).fit()
B0, B1 = model.params
residuals = model.resid

plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('Predicted Values');
plt.ylabel('Residuals');

# Running and interpreting a Ljung-Box test
ljung_box = smd.acorr_ljungbox(residuals, lags=10, return_df=True)
print("P-values:", ljung_box.lb_pvalue.tolist(), "\n")

if (ljung_box.lb_pvalue < 0.05).any():
    print("The residuals are autocorrelated.")
else:
    print("The residuals are not autocorrelated.")

# Markdown Cell 54
# *Note: This new regression was conducted on the differences between data, and therefore the regression output must be back-transformed to reach a prediction in the original scale. Since we regressed the differences, we can add our predicted difference onto the original data to get our estimate:*
# 
# $$\hat{Y_i} = Y_{i-1} + \hat{Y_{diff}}$$

# Markdown Cell 55
# We can also perform a log transformation, if we so choose. This process is identical to the one we performed on the heteroscedastic data up above, so we will leave it out this time.

# Markdown Cell 56
# ## Example: Market Beta Calculation
# 
# Let's calculate the market beta between AAPL and SPY using a simple linear regression, and then conduct a residual analysis on the regression to ensure the validity of our results. To regress AAPL and SPY, we will focus on their returns, not their price, and set SPY returns as our independent variable and AAPL returns as our outcome variable. The regression will give us a line of best fit: 
# 
# $$\hat{r_{AAPL}} = \hat{\beta_0} + \hat{\beta_1}r_{SPY}$$
# 
# The slope of the regression line $\hat{\beta_1}$ will represent our market beta, as for every $r$ percent change in the returns of SPY, the predicted returns of AAPL will change by $\hat{\beta_1}$.
# 
# Let's start by conducting the regression the returns of the two assets.

# Cell 57
from quantrocket.master import get_securities
from quantrocket import get_prices

securities = get_securities(symbols=["AAPL", "SPY"], vendors='usstock')

start = '2017-01-01'
end = '2018-01-01'

closes = get_prices("usstock-free-1min", data_frequency="daily", sids=securities.index.tolist(), fields='Close', start_date=start, end_date=end).loc['Close']

sids_to_symbols = securities.Symbol.to_dict()
closes = closes.rename(columns=sids_to_symbols)

asset = closes['AAPL']
benchmark = closes['SPY']

# We have to take the percent changes to get to returns
# Get rid of the first (0th) element because it is NAN
r_a = asset.pct_change()[1:].values
r_b = benchmark.pct_change()[1:].values

# Regressing the benchmark b and asset a
r_b = sm.add_constant(r_b)
model = sm.OLS(r_a, r_b).fit()
r_b = r_b[:, 1]
B0, B1 = model.params

# Plotting the regression
A_hat = (B1*r_b + B0)
plt.scatter(r_b, r_a, alpha=1) # Plot the raw data
plt.plot(r_b, A_hat, 'r', alpha=1);  # Add the regression line, colored in red
plt.xlabel('AAPL Returns')
plt.ylabel('SPY Returns')

# Print our result
print("Estimated AAPL Beta:", B1)

# Calculating the residuals
residuals = model.resid

# Markdown Cell 58
# Our regression yielded an estimated market beta of 1.36; according to the regression, for every 1% in return we see from the SPY, we should see 1.36% from AAPL.
# 
# Now that we have the regression results and residuals, we can conduct our residual analysis. Our first step will be to plot the residuals and look for any red flags:

# Cell 59
plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('AAPL Returns');
plt.ylabel('Residuals');

# Markdown Cell 60
# By simply observing the distribution of residuals, it does not seem as if there are any abnormalities. The distribution is relatively random and no patterns can be observed (the clustering around the origin is a result of the nature of returns to cluster around 0 and is not a red flag). Our qualitative conclusion is that the data is homoscedastic and not autocorrelated and therefore satisfies the assumptions for linear regression.
# 
# ## Breusch-Pagan Heteroscedasticity Test
# 
# Our qualitative assessment of the residual plot is nicely supplemented with a couple statistical tests. Let's begin by testing for heteroscedasticity using a Breusch-Pagan test. Using the `het_breuschpagan` function from the statsmodels package:

# Cell 61
bp_test = smd.het_breuschpagan(residuals, model.model.exog)

print("Lagrange Multiplier Statistic:", bp_test[0])
print("P-value:", bp_test[1])
print("f-value:", bp_test[2])
print("f_p-value:", bp_test[3], "\n")
if bp_test[1] > 0.05:
    print("The relationship is not heteroscedastic.")
if bp_test[1] < 0.05:
    print("The relationship is heteroscedastic.")

# Markdown Cell 62
# Because the P-value is greater than 0.05, we do not have enough evidence to reject the null hypothesis that the relationship is homoscedastic. This result matches up with our qualitative conclusion.

# Markdown Cell 63
# ## Ljung-Box Autocorrelation Test
# 
# Let's also check for autocorrelation quantitatively using a Ljung-Box test. Using the `acorr_ljungbox` function from the statsmodels package and the default maximum lag:

# Cell 64
ljung_box = smd.acorr_ljungbox(r_a, lags=1, return_df=True)
print("P-Values:", ljung_box.lb_pvalue.tolist(), "\n")
if (ljung_box.lb_pvalue < 0.05).any():
    print("The residuals are autocorrelated.")
else:
    print("The residuals are not autocorrelated.")

# Markdown Cell 65
# Because the Ljung-Box test yielded p-values above 0.05 for all lags, we can conclude that the residuals are not autocorrelated. This result matches up with our qualitative conclusion.
# 
# After having visually assessed the residual plot of the regression and then backing it up using statistical tests, we can conclude that the data satisfies the main assumptions and the linear model is valid.

# Markdown Cell 66
# ## References
# * "Analysis of Financial Time Series", by Ruey Tsay

# Markdown Cell 67
# ---
# 
# **Next Lecture:** [Dangers of Overfitting](Lecture19-Dangers-of-Overfitting.ipynb) 
# 
# [Back to Introduction](Introduction.ipynb) 

# Markdown Cell 68
# ---
# 
# *This presentation is for informational purposes only and does not constitute an offer to sell, a solicitation to buy, or a recommendation for any security; nor does it constitute an offer to provide investment advisory or other services by QuantRocket LLC ("QuantRocket"). Nothing contained herein constitutes investment advice or offers any opinion with respect to the suitability of any security, and any views expressed herein should not be taken as advice to buy, sell, or hold any security or as an endorsement of any security or company.  In preparing the information contained herein, the authors have not taken into account the investment needs, objectives, and financial circumstances of any particular investor. Any views expressed and data illustrated herein were prepared based upon information believed to be reliable at the time of publication. QuantRocket makes no guarantees as to their accuracy or completeness. All information is subject to change and may quickly become unreliable for various reasons, including changes in market conditions or economic circumstances.*

