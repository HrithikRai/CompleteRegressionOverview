import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression,HuberRegressor, Ridge,Lasso

# Generate toy data.
rng = np.random.RandomState(0)
X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4.0,
                       bias=100.0)
# Add four strong outliers to the dataset.
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.
X_outliers[2:, :] += X.min() - X.mean() / 4.
y_outliers[:2] += y.min() - y.mean() / 4.
y_outliers[2:] += y.max() + y.mean() / 4.
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, 'b.')
# Fit the huber regressor over a series of epsilon values.
#See how the regresison line gets influenced by the outliers

x = np.linspace(X.min(), X.max(), 7)
huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=1.75)
huber.fit(X, y)
coef_ = huber.coef_ * x + huber.intercept_
plt.plot(x, coef_,'r-', label="huberRegression")

# Fit a ridge regressor to compare it to huber regressor.
ridge = Ridge(fit_intercept=True, alpha=1.0, normalize=True)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, 'g-', label="ridge regression")

#Fit a Lasso Regressor
lasso=Lasso(fit_intercept=True, alpha=0.6, normalize=True)
lasso.fit(X, y)
coef_lasso = lasso.coef_
coef_ = lasso.coef_ * x + lasso.intercept_
plt.plot(x, coef_, 'b-', label="lasso regression")

#Fit a simple linear regressor
linreg=LinearRegression()
linreg.fit(X, y)
coef_ridge = linreg.coef_
coef_ = linreg.coef_ * x + linreg.intercept_
plt.plot(x, coef_, 'm-', label="SimpleLinearRegression")

plt.title("Comparison of HuberRegressor vs Ridge vs Lasso vs SLR")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()

