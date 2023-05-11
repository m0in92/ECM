import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class SPKF:
    def __init__(self, xhat, Ny, SigmaX, SigmaW, SigmaV, f_func, h_func):
        """
        Constructor for the SPKF class. Currently, the SPKF class supports Central-Difference Kalman Filter (CDKF).
        :param xhat: x_estimate
        :param Nxa: number of state variables in the augmented state vector.
        :param Ny: number of outputs.
        :param SigmaX: X variance.
        :param SigmaW: W variance
        :param SigmaV: V variance
        """
        self.xhat = xhat
        # in case xhat is an int, convert it to float.
        if isinstance(self.xhat, int):
            self.xhat = float(self.xhat)
        if isinstance(self.xhat, float):
            self.Nx = 1
        elif isinstance(self.xhat, np.ndarray):
            self.Nx = len(xhat)
        else:
            raise TypeError("xhat needs to be an int, float, or Numpy array.")

        self.Ny = Ny

        # the length of SigmaX should be equal to the length of xhat. If SigmaX is an int, convert it to float first.
        self.SigmaX = SigmaX
        if isinstance(self.SigmaX, int):
            self.SigmaX = float(self.SigmaX)
        if isinstance(self.SigmaX, float):
            if self.Nx != 1:
                raise ValueError("The length of SigmaX and xhat does not match.")
        elif isinstance(self.SigmaX, np.ndarray):
            if self.SigmaX.shape[0] != self.Nx:
                raise ValueError(f"SigmaX needs to have {self.Nx} rows.")
            if self.SigmaX.shape[1] != self.Nx:
                raise ValueError(f"SigmaX needs to have {self.Nx} cols.")
        else:
            raise TypeError("SigmaX needs to be an int, float, or Numpy Array")

        self.SigmaW = SigmaW
        self.SigmaV = SigmaV
        self.Nxa = self.Nx + 2
        self.L = self.Nxa # length of the augmented state vector.
        self.p = 2 * self.L # Note that p+1 sigma points are generated.

        if callable(f_func):
            self.f_func = f_func
        else:
            raise TypeError("f_func should be a function object.")
        if callable(h_func):
            self.h_func = h_func
        else:
            raise TypeError("h_func should be a function object.")

        self.yhat = 3.5

    @property
    def aug_state_vector(self):
        """
        Augmented state vector with the state, process noise, and sensor noise. Assume that the process and sensor
        noise have zero mean.
        :return:
        """
        if isinstance(self.xhat, float):
            return np.array([self.xhat, 0, 0]).reshape(-1, 1)
        elif isinstance(self.xhat, np.ndarray):
            noises = np.zeros(self.Nxa - self.Nx)
            return np.append(self.xhat.flatten(), noises).reshape(-1,1)
        elif isinstance(self.xhat, int):
            self.xhat = float(self.xhat)
            return np.array([self.xhat, 0, 0]).reshape(-1, 1)
        else:
            raise TypeError("xhat needs to be a int, float, or Numpy array.")

    @property
    def aug_cov_matrix(self):
        """
        Augmented covariance matrix with the covariances of the state, process noise, and the sensor noise.
        :return:
        """
        if isinstance(self.SigmaX, float):
            return np.diag([self.SigmaX, self.SigmaW, self.SigmaV])
        elif isinstance(self.SigmaX, np.ndarray):
            diag_vector = np.append(np.diag(self.SigmaX), self.SigmaW)
            diag_vector = np.append(diag_vector, self.SigmaV)
            return np.diag(diag_vector)
        elif isinstance(self.SigmaX, int):
            self.SigmaX = float(self.SigmaX)
            return np.diag([self.SigmaX, self.SigmaW, self.SigmaV])
        else:
            raise TypeError("SigmaX needs to be an int, float, or a Numpy array")

    @property
    def sqrt_aug_cov_matrix(self):
        return scipy.linalg.cholesky(self.aug_cov_matrix, lower=True)

    @property
    def gamma(self):
        """
        A Tuning parameter for CFKF. For Guassian distributions, gamma is sqrt(3)
        :return:
        """
        return np.sqrt(3)

    @property
    def h(self):
        """
        A tuning parameter for SPKF. For CDKF, gamma is equal to h.
        :return:
        """
        return self.gamma

    @property
    def alpha_m0(self):
        """
        Constant for CDKF.
        :return:
        """
        return ((self.h ** 2) - self.L) / (self.h ** 2)

    @property
    def alpha_m(self):
        """
        Constant for CDKF.
        :return:
        """
        return 1 / (2 * (self.h**2))

    @property
    def alpha_m_vec(self):
        """
        Row vector of all alpha_m entries.
        :return:
        """
        alpha_m_vec = np.array(np.tile(self.alpha_m, self.p)) # vector of all alpha_m, excluding alpha_c_0
        return np.append(self.alpha_m0, alpha_m_vec).reshape(-1, 1)

    @property
    def alpha_c0(self):
        """
        Constant for CDKF.
        :return:
        """
        return ((self.h ** 2) - self.L) / self.h

    @property
    def alpha_c(self):
        """
        Constant for CDKF.
        :return:
        """
        return 1 / (2 * (self.h ** 2))

    @property
    def alpha_c_vec(self):
        """
        A row vector containing all alpha_c vector.
        :return:
        """
        alpha_c_vec = np.array(np.tile(self.alpha_c, self.p))  # vector of all alpha_c, excluding alpha_c_0
        return np.append(self.alpha_c0, alpha_c_vec).reshape(-1, 1)

    @property
    def sigma_point_matrix(self):
        Sigma_Point_matrix = np.tile(self.aug_state_vector, [1, self.p + 1])
        return Sigma_Point_matrix + self.gamma * np.append(np.zeros(self.aug_state_vector.shape),
                                                           np.append(self.sqrt_aug_cov_matrix,
                                                                     -self.sqrt_aug_cov_matrix, axis=1), axis=1)

    def state_estimate_time_update(self, u):
        """
        This is the step 1a (first step) of the process.
        :return:
        """
        # Pass the input elements of the sigma point into the state function. Then the mean estimate is calculated
        Xx = self.f_func(self.sigma_point_matrix[0:self.Nx, :], u, self.sigma_point_matrix[self.Nx, :])
        return Xx, Xx @ self.alpha_m_vec # outputs are augmented xhat matrix and state estimate vector

    def covariance_prediction(self, Xx, xhat):
        """
        This is the step 1b (second step) of the process. The estimate covariance is estimated.
        :param Xx: Sigma-point matrix containing sigma-points for the xhat.
        :param xhat: state vector estimate as calculated from step 1a.
        :return:
        """
        Xs = Xx - np.tile(xhat, [1, self.p + 1])
        return Xs, Xs @ np.diag(self.alpha_c_vec.flatten()) @ Xs.transpose()

    def output_estimate(self, Xx, u):
        """
        Step 1c (third step), which is the output prediction.
        :param Xx:
        :param u:
        :return:
        """
        Y = self.h_func(Xx, u, self.sigma_point_matrix[self.Nx+1,:])
        return Y, Y @ self.alpha_m_vec

    def estimator_gain_matrix(self, Y, yhat, Xs):
        """
        Step 2a
        :param Y: Output Sigma Point
        :param yhat: output variable vector
        :param Xs: difference betwrrn sigma points and state variable
        :return: SigmaY and gain estimator, Lx
        """
        Ys = Y - np.tile(yhat, [1, self.p + 1])
        SigmaXY = Xs @ np.diag(self.alpha_c_vec.flatten()) @ Ys.transpose()
        SigmaY = Ys @ np.diag(self.alpha_c_vec.flatten()) @ Ys.transpose()
        L = SigmaXY @ np.linalg.inv(SigmaY)
        return SigmaY, L

    def state_update(self, L, xhat, ytrue, yhat):
        return xhat + (L @ (ytrue - yhat)).reshape(-1,1)

    def cov_measurement_update(self, Lx, SigmaX, SigmaY):
        return SigmaX - Lx @ SigmaY @ Lx.transpose()

    def solve(self, u, ytrue):

        if isinstance(ytrue, float):
            Ny_i = 1
        elif isinstance(ytrue, np.ndarray):
            Ny_i = len(ytrue)
        else:
            raise TypeError("ytrue needs to be an float or Numpy array.")

        if self.Ny != Ny_i:
            raise ValueError(f"Length of y ({Ny_i}) should be equal to object's Ny attribute ({self.Ny}).")

        # Step 1a:
        Xx, xhat = self.state_estimate_time_update(u=u)
        # Step 1b:
        Xs, SigmaX = self.covariance_prediction(Xx=Xx, xhat=xhat)
        # Step 1c:
        Y, y_hat = self.output_estimate(Xx=Xx, u=u)
        # Step 2a:
        SigmaY, L = self.estimator_gain_matrix(Y=Y, yhat=y_hat, Xs=Xs)
        # Step 2b
        xhat_update = self.state_update(L=L, xhat=xhat, ytrue=ytrue, yhat=y_hat)
        # Step 2c
        SigmaX = self.cov_measurement_update(L, SigmaX=SigmaX, SigmaY=SigmaY)

        # update class attributes.
        self.xhat = xhat_update
        self.SigmaX = SigmaX
        self.yhat = y_hat

    @staticmethod
    def plot(t_array, measurement_array, sigma_array=None, truth_array=None):
        # Plots
        if truth_array is not None:
            plt.plot(t_array, truth_array, label="Truth")
        plt.plot(t_array, measurement_array, label="SPKF est.")
        if sigma_array is not None:
            plt.plot(t_array, measurement_array + sigma_array, "g:", label="Bounds")
            plt.plot(t_array, measurement_array - sigma_array, "g:")
        plt.xlabel('Iteration')
        plt.ylabel('State')
        plt.legend()
        plt.show()

