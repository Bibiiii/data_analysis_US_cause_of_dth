import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def get_train_test(dataFrame, var1, var2, var3=None):
    if(var3==None):
        return pd.DataFrame(dataFrame, columns=[var1]), pd.DataFrame(dataFrame, columns=[var2])
    else:
        return pd.DataFrame(dataFrame, columns=[var1]), pd.DataFrame(dataFrame, columns=[var2]), pd.DataFrame(dataFrame, columns=[var3])

''' x,y need to be the name of the columns in the dataframe '''
def slr(x, y):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_x, train_y = get_train_test(train_df, x, y)
    test_x, test_y = get_train_test(test_df, x, y)

    xFrame = train_x
    yFrame = train_y
    """ Perform simple linear regression"""
    reg = LinearRegression().fit(xFrame, yFrame)
    print("For Simple Linear Regression:")
    print("Train Score: " + str(reg.score(xFrame, yFrame)))
    print("Parameters:")
    print("Gradient: " + str(reg.coef_[0][0]))
    print("Intercept: " + str(reg.intercept_[0]))
    print("Test score: " + str(reg.score(test_x, test_y)))
    plt.scatter(test_x, test_y, color="red")
    plt.plot(xFrame, reg.predict(xFrame))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("A linear model of " + y + " against " + x)
    plt.show()
    print()

def plr(x, y, maxDegree, TOL=0.02):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_x, train_y = get_train_test(train_df, x, y)
    test_x, test_y = get_train_test(test_df, x, y)

    xFrame = train_x
    yFrame = train_y
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
    plt.title("A plot of fit score against the degree of the polynomial fit")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Score")
    plt.show()

    """ Take the best fitting result & use it for plotting."""
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(xFrame)
    poly_reg.fit(X_poly, yFrame)
    lin_reg2 = LinearRegression().fit(X_poly, yFrame)
    X_grid = np.arange(min(xFrame.to_numpy()), max(xFrame.to_numpy()), (max(xFrame.to_numpy()) - min(xFrame.to_numpy()))/100)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(test_x, test_y, color="red")
    Y_grid = np.zeros(len(X_grid))
    print()
    print("For Polynomial Regression:")
    print("Degree used: " + str(deg))
    print("Train Score: " + str(score))
    print("Coefficients: " + str(lin_reg2.coef_[0]))
    print("Test Score: " + str(lin_reg2.score(poly_reg.fit_transform(test_x), test_y)))
    print()
    for i in range(len(X_grid)):
        Y_grid[i] = lin_reg2.predict(poly_reg.fit_transform([X_grid[i]]))
    plt.plot(X_grid, Y_grid)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("A linear model of " + y + " against " + x + " with degree " + str(deg))
    plt.show()

def mlr(x,y,z):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_x, train_y, train_z = get_train_test(train_df, x, y, z)
    test_x, test_y, test_z = get_train_test(test_df, x, y, z)
    xFrame = train_x
    yFrame = train_y
    zFrame = train_z
    df = pd.DataFrame(xFrame)
    df['y'] = yFrame

    """ Begin multilinear regression algorithm"""
    lin_reg3 = LinearRegression()
    lin_reg3.fit(df, zFrame)
    print("Using multilinear regression, we get:")
    print("Coefficients: " + str(lin_reg3.coef_))
    print("Intercept: " + str(lin_reg3.intercept_))
    print("Train Score: " + str(lin_reg3.score(df, zFrame)))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_grid = np.linspace(min(test_x.to_numpy())[0],max(test_x.to_numpy())[0],10)
    y_grid = np.linspace(min(test_y.to_numpy())[0],max(test_y.to_numpy())[0],10)
    z_grid = np.zeros([10,10])
    for i in range(10):
        for j in range(10):
            z_grid[i][j] = (x_grid[i] * lin_reg3.coef_[0][0] + y_grid[j] * lin_reg3.coef_[0][1] + lin_reg3.intercept_[0])
    finalDf = test_x
    finalDf['y'] = test_y
    print("Test Score: " + str(lin_reg3.score(finalDf, test_z)))
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    for i in range((len(test_x))):
        ax.scatter(test_x.to_numpy()[i][0], test_y.to_numpy()[i][0], test_z.to_numpy()[i][0], color="red", s=10)
    ax.plot_surface(x_grid, y_grid, z_grid)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title("A multilinear regression plot of " + z + " against " + x + " and " + y)
    plt.show()

def mplr(x,y,z, maxDegree, TOL=0.02):
    # Import processed data file
    pre_processed_data = pd.read_csv(r'pre_processed.csv')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_x, train_y, train_z = get_train_test(train_df, x, y, z)
    test_x, test_y, test_z = get_train_test(test_df, x, y, z)
    xFrame = train_x
    yFrame = train_y
    zFrame = train_z
    coeffs = np.zeros([maxDegree,3])
    df = pd.DataFrame(xFrame)
    df['y'] = yFrame
    score = 0
    deg = 1
    degAxis = np.zeros(maxDegree + 1)
    scoreAxis = np.zeros(maxDegree + 1)
    for i in range(1, maxDegree + 1):
        x_poly = xFrame ** i
        mult = xFrame.to_numpy() * yFrame.to_numpy()
        cross_poly = np.zeros(len(mult))
        for j in range(len(mult)):
            cross_poly[j] = mult[j][1]
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
        for j in range(3):
            coeffs[i][j] = lin_reg2.coef_[j]
        if lin_reg2.score(newDf, zFrame) > score:
            score = lin_reg2.score(newDf, zFrame)
            deg = i
        elif lin_reg2.score(newDf, zFrame) < score:
            """ Catches the case where the previous model performed better"""
            print("Higher degree model gives worse result. Exiting loop early and using previous model.")
            print("Current model: " + str(lin_reg2.score(newDf, zFrame)))
            print("Previous model: " + str(score))
            """ Move one step back in the iteration"""
            deg = i - 1
            """ Place the coefficients from the previous model into the current model, thus overwriting the model"""
            for j in range(3):
                lin_reg2.coef_[j] = coeffs[deg][j]
            print("Proceeding with degree " + str(deg))
            break
        """Otherwise, record current iteration and carry on"""
        degAxis[i] = i
        scoreAxis[i] = score
        """ Analyse gradients to identify where the score function begins to converge to avoid overfitting."""
        """ TOL must be chosen as very small. """
        if i > 0 and ((scoreAxis[i] - scoreAxis[i - 1]) / (degAxis[i] - degAxis[i - 1])) < TOL:
            print("Exiting algorithm as convergence point has been found.")
            deg = int(degAxis[i - 1])
            print("Proceeding with degree " + str(deg))
            break
    plt.scatter(degAxis, scoreAxis)
    plt.show()

    """ Take the best fitting result & use it for plotting."""
    X_grid = np.arange(min(test_x.to_numpy()), max(test_x.to_numpy()),
                       (max(test_x.to_numpy()) - min(test_x.to_numpy())) / 10)
    X_grid = X_grid.reshape((len(X_grid), 1))
    Y_grid = np.arange(min(test_y.to_numpy()), max(test_y.to_numpy()),
                       (max(test_y.to_numpy() - min(test_y.to_numpy()))) / 10)
    Y_grid = Y_grid.reshape((len(Y_grid), 1))

    Z_grid = np.zeros([len(X_grid), len(Y_grid)])

    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            Z_grid[i][j] = coeffs[deg][0] * (X_grid[i] ** deg) + coeffs[deg][1] * (X_grid[i] * Y_grid[j]) + coeffs[deg][2] * (Y_grid[j] ** deg)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)
    ax = plt.axes(projection='3d')
    ax.scatter(test_x, test_y, test_z, color='red')
    ax.plot_surface(X_grid, Y_grid, Z_grid)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title("A multivariate polynomial model of " + z + " against " + x + " and " + y + " with degree " + str(deg))
    print()
    print("For Polynomial Regression:")
    print("Degree used: " + str(deg))
    print("Train Score: " + str(score))
    print("Coefficients: " + str(coeffs[deg]))
    testX = test_x.to_numpy()
    testY = test_y.to_numpy()
    for i in range(len(test_x)):
        testX[i] = test_x.to_numpy()[i] ** deg
        testY[i] = test_y.to_numpy()[i] ** deg
    finalDf = pd.DataFrame(test_x)
    cross_poly = np.zeros(len(test_x))
    finalDf['xy'] = cross_poly
    finalDf['y'] = testY
    print("Test Score: " + str(lin_reg2.score(finalDf, test_z)))
    print()
    plt.show()


# slr('AgeGroup', 'AllCause')
# plr('AgeGroup', 'AllCause', 10)
# slr('COVID-19 (U071, Underlying Cause of Death)', 'AllCause')
# plr('COVID-19 (U071, Underlying Cause of Death)', 'AllCause', 10)
# mlr('AgeGroup', 'COVID-19 (U071, Underlying Cause of Death)', 'AllCause')
# mplr('AgeGroup', 'COVID-19 (U071, Underlying Cause of Death)', 'AllCause', 10, 0.001)
mlr('Chronic lower respiratory diseases (J40-J47)', 'COVID-19 (U071, Underlying Cause of Death)', 'AllCause')
# mplr('Chronic lower respiratory diseases (J40-J47)', 'AgeGroup', 'COVID-19 (U071, Underlying Cause of Death)', 10, 0.001)