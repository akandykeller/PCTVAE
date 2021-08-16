import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD


# Source: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.collapse_dims = tuple(range(len(data.shape[:-3])))

            self.mean = data.mean(self.collapse_dims)
            self.std  = data.std(self.collapse_dims)
            self.nobservations = data.shape[0]
            self.chw   = data.shape[-3:]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, *transformations, in_channels, height, width)
        We take the mean over all transformations and observations, so only leaving inchannels,  height and width
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[-3:] != self.chw:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(self.collapse_dims)
            newstd  = data.std(self.collapse_dims)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n



class NormalizePixels(object):
    def __init__(self, mean, std):
        if mean.size() != std.size():
            raise ValueError("Mean and STD should have same size")
        self.mean = mean[(None,)*3] # Extend mean to transform dimensions
        self.std = std[(None,)*3]

    def __call__(self, tensor):
        """
        Assumes Data is already Centered!!!!
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size()[-3:] != self.mean.size()[-3:]:
            raise ValueError("tensor and mean matrix have incompatible shape: {}, {}.".format(str(tensor.size()), str(self.mean.size())))

        tensor = (tensor - self.mean) / (self.std + 1e-5)

        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ 
        return format_string


def zca_whitening_matrix(X, n_components=750):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Singular Value Decomposition. X = U * np.diag(S) * V
    svd = TruncatedSVD(n_components=n_components)
    R = np.dot(X, X.T) / X.T.shape[0]
    svd.fit(R)

    print("ZCA: {}/{} components explain {:.3f} percent of variance".format(n_components, X.shape[0], svd.explained_variance_ratio_.sum()))

    S = svd.singular_values_
    U = svd.components_.T

    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]

    return ZCAMatrix

# def pca_whitening_matrix(X):
#     """
#     Function to compute PCA whitening matrix.
#     INPUT:  X: [M x N] matrix.
#         Rows: Variables
#         Columns: Observations
#     OUTPUT: ZCAMatrix: [M x M] matrix
#     """
#     # Singular Value Decomposition. X = U * np.diag(S) * V
#     pca = PCA(whiten=True)
#     pca.fit_transform(X)

#     S = svd.singular_values_
#     U = svd.components_.T

#     # Whitening constant: prevents division by zero
#     epsilon = 1e-5
#     # ZCA Whitening matrix: U * Lambda * U'
#     ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]

#     return ZCAMatrix


# Source: https://github.com/semi-supervised-paper/semi-supervised-paper-implementation/blob/e39b61ccab/semi_supervised/core/utils/data_util.py#L150
class ZCATransformation(object):
    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Assumes Data is already Centered!!!!
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        flat_tensor = tensor.view(-1, 1)
        if flat_tensor.size(0) != self.transformation_matrix.size(1):
            raise ValueError("tensor and transformation matrix have incompatible shape.")

        transformed_tensor = torch.mm(self.transformation_matrix, flat_tensor)

        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string
