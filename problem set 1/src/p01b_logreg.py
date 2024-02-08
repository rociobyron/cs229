import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    it = 0
    eps = 1e-5
    err = 1

    m = x_train.shape[0]
    n = x_train.shape[1]
    theta = np.zeros((n, 1))
    H = np.zeros((n, n))

    while (err > eps) and (it < 1000):
        theta_old = theta.copy()
        z = np.dot(np.transpose(theta), np.transpose(x_train)) # 1xm
        g = 1 / (1 + np.exp(-z)) # 1xm

        nl = -1/m * np.dot(np.transpose(x_train), np.transpose(y_train - g)) #nx1

        for i in range(m):
            x = x_train[i, :] #1xn
            H += g[0, i] * np.outer(x,np.transpose(x)) #nxn
        H = 1/m * H

        theta -= np.dot(np.linalg.inv(H), nl)

        it += 1
        err = np.linalg.norm(theta - theta_old, ord=1)
        if it % 10 == 0:
            print("Iteration {}, error = {:.2e}".format(it, err))

    # Plot decision boundary on top of validation set set
    # Decision boundary
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    x_1 = x_eval[:, 1]
    x_2 = x_eval[:, 2]

    x_1bound = np.linspace(np.min(x_1), np.max(x_1), 10)
    x_2bound = (-theta[0] - theta[1]*x_1bound) / theta[2]

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x=x_1, y=x_2, hue=y_eval, 
                    palette=sns.color_palette("Set2"), ax=ax)
    sns.lineplot(x=x_1bound, y=x_2bound, dashes=True, ax=ax)
    plt.show()
    
    # Use np.savetxt to save predictions on eval set to pred_path
    h_theta = 1 / (1 + np.exp(-np.dot(np.transpose(theta), np.transpose(x_eval))))
    y_pred = np.where(h_theta < 0.5, 0, 1)
    np.savetxt(pred_path, np.column_stack((x_1.T, x_2.T, y_eval.T, y_pred[0, :])), fmt= "%.2f", delimiter=",", header="x_1,x_2,y_eval,y_pred")

    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        #print("converged in ", n_steps, " steps")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        # *** END CODE HERE ***


if __name__ == '__main__':
    # Paths relative to workspace folder
    main('./problem set 1/data/ds1_train.csv', 
         './problem set 1/data/ds1_valid.csv', 
         './problem set 1/data/ds1_pred.txt')
