import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import axes3d
import math

''' x,y need to be the name of the columns in the dataframe '''
def slr(x, y):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # train_x, train_y = get_train_test(train_df)
    # test_x, test_y = get_train_test(test_df)

    xFrame = pd.DataFrame(pre_processed_data, columns=[x])
    yFrame = pd.DataFrame(pre_processed_data, columns=[y])
    """ Perform simple linear regression"""
    reg = LinearRegression().fit(xFrame, yFrame)
    print("For Simple Linear Regression:")
    print("Score: " + str(reg.score(xFrame, yFrame)))
    print("Parameters:")
    print("Gradient: " + str(reg.coef_[0][0]))
    print("Intercept: " + str(reg.intercept_[0]))
    plt.scatter(xFrame, yFrame)
    plt.plot(xFrame, reg.predict(xFrame), color="red")
    plt.show()
    print()

def plr(x, y, maxDegree, TOL):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # train_x, train_y = get_train_test(train_df)
    # test_x, test_y = get_train_test(test_df)

    xFrame = pd.DataFrame(pre_processed_data, columns=[x])
    yFrame = pd.DataFrame(pre_processed_data, columns=[y])
    """ Perform polynomial regression. Determine the best fitting polynomial up to a specified max degree"""
    """ Plots the score function against degree of the polynomial."""
    score = 0
    deg = 1
    degAxis = np.zeros(maxDegree + 1)
    scoreAxis = np.zeros(maxDegree + 1)
    for i in range(0, maxDegree + 1):
        poly_reg = PolynomialFeatures(degree=i)
        X_poly = poly_reg.fit_transform(xFrame)
        poly_reg.fit(X_poly, yFrame)
        lin_reg2 = Lasso(alpha=1.0,max_iter=3000)
        lin_reg2.fit(X_poly, yFrame)
        if lin_reg2.score(X_poly, yFrame) > score:
            score = lin_reg2.score(X_poly, yFrame)
            deg = i
        elif lin_reg2.score(X_poly, yFrame) < score:
            break
        degAxis[i] = i
        scoreAxis[i] = score
        print("Degree used: " + str(i))
        print("Score: " + str(score))
        print("Coefficients: " + str(lin_reg2.coef_[0]))
        print()
        """ Analyse gradients to identify 'elbow' point to avoid further overfitting."""
        if i > 0 and ((scoreAxis[i] - scoreAxis[i - 1]) / (degAxis[i] - degAxis[i - 1])) < TOL:
            print("Exiting algorithm as elbow point has been found.")
            deg = int(degAxis[i - 1])
            print("Proceeding with degree " + str(deg))
            break
    plt.scatter(degAxis, scoreAxis)
    plt.show()

    """ Take the best fitting result & use it for plotting."""
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(xFrame)
    poly_reg.fit(X_poly, yFrame)
    lin_reg2 = LinearRegression().fit(X_poly, yFrame)
    X_grid = np.arange(min(xFrame.to_numpy()), max(xFrame.to_numpy()), (max(xFrame.to_numpy()) - min(xFrame.to_numpy()))/100)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(xFrame, yFrame)
    Y_grid = np.zeros(len(X_grid))
    print()
    print("For Polynomial Regression:")
    print("Degree used: " + str(deg))
    print("Score: " + str(score))
    print("Coefficients: " + str(lin_reg2.coef_[0]))
    print()
    for i in range(len(X_grid)):
        Y_grid[i] = lin_reg2.predict(poly_reg.fit_transform([X_grid[i]]))
    plt.plot(X_grid, Y_grid, color="red")
    plt.show()

def mlr(x,y,z):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    # train_df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')
    # train_x, train_y = get_train_test(train_df)
    # test_x, test_y = get_train_test(test_df)
    xFrame = pd.DataFrame(pre_processed_data, columns=[x])
    yFrame = pd.DataFrame(pre_processed_data, columns=[y])
    zFrame = pd.DataFrame(pre_processed_data, columns=[z])
    df = pd.DataFrame(pre_processed_data, columns=[x,y])

    """ Begin multilinear regression algorithm"""
    lin_reg3 = LinearRegression()
    lin_reg3.fit(df, zFrame)
    print("Using multilinear regression, we get:")
    print("Coefficients: " + str(lin_reg3.coef_))
    print("Intercept: " + str(lin_reg3.intercept_))
    print("Score: " + str(lin_reg3.score(df, zFrame)))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_grid = np.linspace(min(xFrame.to_numpy())[0],max(xFrame.to_numpy())[0],10)
    y_grid = np.linspace(min(yFrame.to_numpy())[0],max(yFrame.to_numpy())[0],10)
    z_grid = np.zeros([10,10])
    for i in range(10):
        for j in range(10):
            z_grid[i][j] = (x_grid[i] * lin_reg3.coef_[0][0] + y_grid[j] * lin_reg3.coef_[0][1] + lin_reg3.intercept_[0])

    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    for i in range((len(xFrame))):
        ax.scatter(xFrame.to_numpy()[i][0], yFrame.to_numpy()[i][0], zFrame.to_numpy()[i][0], s=10)
    ax.plot_surface(x_grid, y_grid, z_grid)
    plt.show()

def mplr(x,y,z, maxDegree, TOL):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    # train_df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')
    # train_x, train_y = get_train_test(train_df)
    # test_x, test_y = get_train_test(test_df)
    xFrame = pd.DataFrame(pre_processed_data, columns=[x])
    yFrame = pd.DataFrame(pre_processed_data, columns=[y])
    zFrame = pd.DataFrame(pre_processed_data, columns=[z])
    df = pd.DataFrame(pre_processed_data, columns=[x,y])

    score = 0
    deg = 1
    degAxis = np.zeros(maxDegree + 1)
    scoreAxis = np.zeros(maxDegree + 1)
    for i in range(0, maxDegree + 1):
        poly_reg = PolynomialFeatures(degree=i)
        x_poly = xFrame ** i
        cross_poly = (xFrame.to_numpy() * yFrame.to_numpy())
        y_poly = xFrame ** i
        lin_reg2 = Lasso(alpha=0.5, max_iter=2000)
        newDf = pd.DataFrame(x_poly)
        newDf['xy'] = cross_poly
        newDf['y'] = y_poly
        lin_reg2.fit(newDf, zFrame)
        print("Degree used: " + str(i))
        print("Score: " + str(score))
        print("Coefficients: " + str(lin_reg2.coef_))
        print()
        if lin_reg2.score(newDf, zFrame) > score:
            score = lin_reg2.score(newDf, zFrame)
            deg = i
        elif lin_reg2.score(newDf, zFrame) < score:
            break
        degAxis[i] = i
        scoreAxis[i] = score
        """ Analyse gradients to identify 'elbow' point to avoid further overfitting."""
        if i > 0 and ((scoreAxis[i] - scoreAxis[i - 1]) / (degAxis[i] - degAxis[i - 1])) < TOL:
            print("Exiting algorithm as elbow point has been found.")
            deg = int(degAxis[i - 1])
            print("Proceeding with degree " + str(deg))
            break
    plt.scatter(degAxis, scoreAxis)
    plt.show()
    """ Take the best fitting result & use it for plotting."""
    poly_reg = PolynomialFeatures(degree=deg)
    x_poly = xFrame ** i
    cross_poly = (xFrame.to_numpy() * yFrame.to_numpy())
    y_poly = xFrame ** i
    lin_reg2 = Lasso(alpha=0.5, max_iter=2000)
    newDf = pd.DataFrame(x_poly)
    newDf['xy'] = cross_poly
    newDf['y'] = y_poly
    lin_reg2 = Lasso(alpha=0.5, max_iter=2000)
    lin_reg2.fit(newDf, zFrame)
    X_grid = np.arange(min(xFrame.to_numpy()), max(xFrame.to_numpy()),
                       (max(xFrame.to_numpy()) - min(xFrame.to_numpy())) / 10)
    X_grid = X_grid.reshape((len(X_grid), 1))
    Y_grid = np.arange(min(yFrame.to_numpy()), max(yFrame.to_numpy()),
                       (max(yFrame.to_numpy() - min(yFrame.to_numpy()))) / 10)
    Y_grid = Y_grid.reshape((len(Y_grid), 1))
    ax = plt.axes(projection='3d')
    Z_grid = np.zeros([len(X_grid), len(Y_grid)])
    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            Z_grid[i][j] = lin_reg2.coef_[0]/1000000 * (Y_grid[i] ** deg) + lin_reg2.coef_[1] * (X_grid[i] * Y_grid[j]) + lin_reg2.coef_[2] * (X_grid[j] ** deg)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    ax.plot_surface(X_grid, Y_grid, Z_grid)
    ax.scatter(xFrame, yFrame, zFrame, s=10)
    print()
    print("For Polynomial Regression:")
    print("Degree used: " + str(deg))
    print("Score: " + str(score))
    print("Coefficients: " + str(lin_reg2.coef_))
    print()
    plt.show()


# slr('AgeGroup', 'AllCause')
# plr('AgeGroup', 'AllCause', 10, 0.02)
# slr('COVID-19 (U071, Underlying Cause of Death)', 'AllCause')
# plr('COVID-19 (U071, Underlying Cause of Death)', 'AllCause', 10, 0.02)
# mlr('AgeGroup', 'COVID-19 (U071, Underlying Cause of Death)', 'AllCause')
mplr('AgeGroup', 'COVID-19 (U071, Underlying Cause of Death)', 'AllCause', 10, 0.001)