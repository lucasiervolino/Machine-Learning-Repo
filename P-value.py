#p-value statistical test to obtain significance of independent variables on explaining response variable y
import statsmodels.api as sm

est = sm.OLS(y, x)
est2 = est.fit()
print(est2.summary())
