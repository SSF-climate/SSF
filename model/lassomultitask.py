
"""
Multitask Lasso Model

Details: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html


"""
from sklearn.linear_model import MultiTaskLasso



class LassoMultitask(MultiTaskLasso):
    """Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.


    """
    def __init__(self, alpha=1.0, fit_intercept=False, normalize=False, copy_X=True,
                 max_iter=1000, tol=0.0001, warm_start=False, random_state=None,
                 selection='cyclic'):

        """Initalize a Multitask Lasso Model.

        """

        super(LassoMultitask, self).__init__(alpha=alpha, fit_intercept=fit_intercept,
                                             normalize=normalize, copy_X=copy_X,
                                             max_iter=max_iter, tol=tol,
                                             warm_start=warm_start, random_state=random_state,
                                             selection=selection)    




    def fit(self, X, y):
        """Fit MultiTask Lasso Model.
        """

        super(LassoMultitask, self).fit(X, y)
        return self

    def fit_cv(self, train_x, train_y, val_x, val_y):
        """For Hyper-parameter Tuning: Fit MultiTask Lasso Model.
        """

        self.fit(train_x, train_y)
        #pred_y = self.predict(val_x)



        return self.predict(train_x), self.predict(val_x)


    def predict(self, X):
        """Predict using trained Multitask Model.
        """

        return super(LassoMultitask, self).predict(X)
