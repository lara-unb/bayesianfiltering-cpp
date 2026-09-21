/** @file */
/**********************************************************************************************
***********************************************************************************************
***  File:			BayesianFiltering.hpp
***	 Author:		Geovany A. Borges
***	 Contents:		BayesianFiltering header file.
***********************************************************************************************
***********************************************************************************************
    BayesianFiltering.hpp is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License (LGPL) as published by
    the Free Software Foundation, either version 3 of the License, or any later version.

    BayesianFiltering.hpp is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with BayesianFiltering.hpp.  If not, see <http://www.gnu.org/licenses/>.

	Copyright 2020 Geovany Araujo Borges
**********************************************************************************************/

//******************************************************************
//** Classe BayesianFiltering
//******************************************************************
#include <deque>
#include <stdio.h>

#include <Eigen/Dense>
#include <Eigen/Eigen>

#ifndef BayesianFiltering_H
#define BayesianFiltering_H

/***********************************************************************************************
 *  Auxiliary Macros:
***********************************************************************************************/
/**
 * @brief Print filename followed by line number at command line. Used commonly for debugging allowing to identify the code responsible for errors.
 * @param[in] X The EIGEN::VectorXd variable.
 */
#define PRINT_LINE_DEBUG {printf("\n[%s] line %i",__FILE__,__LINE__); fflush(stdout);}

/**
 * @brief Print pretty matrix X with name which may me different of "X" at command line (printf). Guaranteed print all lines before leaving (fflush).
 * @param[in] X The EIGEN::MatrixXd variable.
 * @param[in] name Name of the variable.
 */
inline void PrintEigenMatrixXd(Eigen::MatrixXd &X, const char* name)
{
	printf("\nMatrix %s (%li x %li):",name,X.rows(),X.cols());
	for (long int i=0; i < X.rows(); ++i){
		printf("\n    ");
		for (long int j=0; j < X.cols(); ++j){
			printf("%s(%li,%li)=%.10f; ",name,i,j,X(i,j));
		}
	}
	fflush(stdout);
}

/**
 * @brief Print pretty matrix X with the same variable name at command line (printf). Guaranteed print all lines before leaving (fflush).
 * @param[in] X The EIGEN::MatrixXd variable.
 */
#define PRINT_EIGEN_MATRIX(X) PrintEigenMatrixXd(X,#X); 

/**
 * @brief Print pretty vector X with name which may me different of "X" at command line (printf). Guaranteed print all lines before leaving (fflush).
 * @param[in] X The EIGEN::VectorXd variable.
 * @param[in] name Name of the variable.
 */
inline void PrintEigenVectorXd(Eigen::VectorXd &X, const char* name)
{
	printf("\nVector %s (%li x %li):",name,X.rows(),X.cols());
	for (long int i=0; i < X.rows(); ++i){
		printf("\n    ");
		for (long int j=0; j < X.cols(); ++j){
			printf("%s(%li,%li)=%.10f; ",name,i,j,X(i,j));
		}
	}
	fflush(stdout);
}
/**
 * @brief Print pretty vector X with the same variable name at command line (printf). Guaranteed print all lines before leaving (fflush).
 * @param[in] X The EIGEN::VectorXd variable.
 */
#define PRINT_EIGEN_VECTOR(X) PrintEigenVectorXd(X,#X);


namespace BayesianFiltering {

	/***********************************************************************************************
	 *  Support types and functions:
	***********************************************************************************************/
	/**
	 * @brief Encapsulated <bool,std::string> pair response of a function, with .first specifying status (success of failure) and .second the reason string in case of failure.
	 */
	typedef std::pair<bool, std::string> Result;

	/**
	 * @brief Format <bool,std::string> pair response with defined status and reason string.
	 * @param[in] success bool indicating status
	 * @param[in] message string message
	 */
	inline Result ResultFormat(bool success, const std::string& message) {
        return std::make_pair(success, message);
    }	

	/**
	 * @brief Format <bool,std::string> pair response with defined status and reason string to be printf-like formated.
	 * @param[in] success bool indicating status
	 * @param[in] format formating string
	 * @param[in] args sequence of variable arguments for message formating
	 */
	template<typename ... Args>
	Result ResultFormat(bool success, const std::string& format, Args ... args )
	{
		Result res; res.first = success;
		int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
		if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
		auto size = static_cast<size_t>( size_s );
		std::unique_ptr<char[]> buf( new char[ size ] );
		std::snprintf( buf.get(), size, format.c_str(), args ... );
		res.second = std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
		return res;
	};


	/**
	 * @brief Generate multivariate \p sample ~ N(\p 0, \p covariance_matrix).
	 * @param[out] sample will be filled with the outcome
	 * @param[in] covariance_matrix as above
	 * @return Result type 
	 */
	Result MultivariateZeroMeanGaussianSampler(Eigen::VectorXd & sample, Eigen::MatrixXd & covariance_matrix)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);

		// Verify dimensions
		if (covariance_matrix.rows() != covariance_matrix.cols()) return ResultFormat(false, "covariance_matrix should be square (%li x %li)",covariance_matrix.rows(),covariance_matrix.cols());

		// Compute Cholesky Decomposition (LLT)
		Eigen::LLT<Eigen::MatrixXd> llt(covariance_matrix);
		Eigen::MatrixXd L = llt.matrixL();

		// Generate independent standard normal vector Z
		Eigen::VectorXd Z = Eigen::VectorXd::NullaryExpr(covariance_matrix.rows(), [&]() { return dist(gen); });

		// Transform to Multivariate Normal
		sample = L * Z;

		return ResultFormat(true,"");
	}

	/**
	 * @brief Generate multivariate \p sample ~ N(\p mean_vector, \p covariance_matrix).
	 * @param[out] sample will be filled with the outcome
	 * @param[in] mean_vector as above
	 * @param[in] covariance_matrix as above
	 * @return Result type 
	 */
	Result MultivariateGaussianSampler(Eigen::VectorXd & sample, Eigen::VectorXd & mean_vector, Eigen::MatrixXd & covariance_matrix)
	{
		// Verify dimensions
		if (mean_vector.rows() != covariance_matrix.rows()) return ResultFormat(false, "mean_vector number of lines (%li) different of covariance_matrix number of lines (%li)",mean_vector.rows(),covariance_matrix.rows());
		// Covariance Matrix will be checked at zero mean Gaussian sampler

		// Generate sample from zero mean sample:
		Result res = MultivariateZeroMeanGaussianSampler(sample, covariance_matrix);
		if (!res.first) return res;

		sample += mean_vector;

		return res;
	}
	
	/**
	 * @brief Generate univariate \p sample ~ N(\p mean, \p variance).
	 * @param[in] mean as above
	 * @param[in] variance as above
	 * @return \p sample
	 */
	double UnivariateGaussianSampler(double mean, double variance)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> dist(0.0, 1.0);

    	return mean + sqrt(variance) * dist(gen);
	}

	/**
	 * @brief Generate univariate \p sample ~ U(\p min_value, \p max_value).
	 * @param[in] min_value as above
	 * @param[in] max_value as above
	 * @return \p sample
	 */
	double UnivariateUniformSampler(double min_value, double max_value)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(0.0, 1.0);

    	return dist(gen);
	}

	/***********************************************************************************************
	 *  Linear model class:
	***********************************************************************************************/
	/** 
	 * @brief Represents the parameters of a discrete time, generic linear state space model:	
	 * 
	 * 		x(k) = A*x(k-1) + B*u(k) + w(k) {Process equation}
	 * 
	 * 		y(k) = C*x(k) + v(k) {Measurement equation}
	 * with \p x(k) being state variables vector, \p u(k) measurement vector and \p y(k) measurement vector. Also, \p w(k) ~ \p N(0,Q) and \p v(k) ~ \p N(0,R) are respectively process and measurement noises.
	 * 
	 * This class contains only model parameters and evaluations of process and measurement equations.
	 * In case \p w(k) and/or \p v(k) are dependent of other noise sources, \p Q and \p R should consider the correct propagation from noise sources covariances. 
	 * 
	 * Initial state is a random variable following \p x(0) ~ \p N(X0,P0)
	 * 
	 * Also dim(x) = \p Nx, dim(u) = \p Nu, dim(y) = \p Ny, dim(w) = \p Nw, dim(v) = \p Nv. 
	 * These dimensions should be first stablished by calling \p SetDimmensions(), which make resizing of system parameters. After that, since parameters are public, they can be set at any time. 
	 * 
	 * Angular measurements in radians need special treatment when computing innovation vector in filters using \p atan2 function. 
	 * In order to do this, \p Y_isangle is used to indicate which components of \p Y are angles, which should be set as \p true by the user when initializing the model.
	 * 
	 * For the same reason, every change in state vector containing angles in some entries should me donne properly.
	 * Thus, \p X_isangle is used to indicate which components of \p x are angles, which should be set as \p true by the user when initializing the model.
	 */
	class StateSpaceLinearModel
	{
	public:
		Eigen::MatrixXd A; //!< Process model transition matrix
		Eigen::MatrixXd B; //!< Process model input matrix
		Eigen::MatrixXd C; //!< Measurement model matrix
		Eigen::MatrixXd Q; //!< Process model noise covariance matrix
		Eigen::MatrixXd R; //!< Measurement model noise covariance matrix
		Eigen::VectorXd X0; //!< Inicial state vector distribution mean vector parameter
		Eigen::MatrixXd P0; //!< Inicial state vector distribution covariance matrix parameter
		int Nx; //!< Number of state variables in X vector
		int Nu; //!< Number of inputs in U vector
		int Ny; //!< Number of measurements in Y vector
		int Nw; //!< Number of process noise in W vector
		int Nv; //!< Number of measurement noise in V vector
		std::vector<bool> X_isangle; //!< Indicates which entries of X vector are angles in rad, for correct computation of correction equation.
		std::vector<bool> Y_isangle; //!< Indicates which entries of Y vector are angles in rad, for correct computation of innovation vector

	/**
	 * @brief Define dimensions of system variables and adjust size of model parameters accordingly.
	 * @param[in] _Nx Number of state variables in X vector
	 * @param[in] _Nu Number of inputs in U vector
	 * @param[in] _Ny Number of measurements in Y vector
	 * @param[in] _Nw Number of process noise in W vector
	 * @param[in] _Nv Number of measurement noise in V vector
	 */
		void SetDimensions(int _Nx, int _Nu, int _Ny, int _Nw, int _Nv)
		{
			Nx = _Nx; Nu =_Nu; Ny = _Ny; Nw = _Nw, Nv = _Nv;			
			A.resize(Nx, Nx);
			B.resize(Nx, Nu);
			C.resize(Ny, Nx);
			Q.resize(Nx, Nx);
			R.resize(Ny, Ny); 
			X0.resize(Nx);
			P0.resize(Nx, Nx);
			X_isangle.resize(Nx);
			std::fill(X_isangle.begin(), X_isangle.end(), false);
			Y_isangle.resize(Ny);
			std::fill(Y_isangle.begin(), Y_isangle.end(), false);
		}

	/**
	 * @brief Compute process model to update state vector \p X (in place).
	 * @param[in,out] X state vector with \p x(k-1) to be updated to \p x(k)
	 * @param[in] U input vector
	 * @param[in] W current sample of process noise
	 */
	Result ProcessModel(Eigen::VectorXd & X, Eigen::VectorXd & U, Eigen::VectorXd & W)
	{
		Eigen::VectorXd Xpos;
		Xpos = A * X + B * U + W; // By using an auxiliary variable, it avoids any risk of copy in place 
		X = Xpos;
		return BayesianFiltering::ResultFormat(true,"");
	}

	/**
	 * @brief Compute system output \p Y from measurement model.
	 * @param[out] Y the computed output
	 * @param[in] X state vector with \p x(k)
	 * @param[in] V current sample of measurement noise
	 */
	Result MeasurementModel(Eigen::VectorXd & Y, Eigen::VectorXd & X, Eigen::VectorXd & V)
	{
		Y = C * X + V;
		return BayesianFiltering::ResultFormat(true,"");
	}

	};

	/***********************************************************************************************
	 *  Nonlinear model base class:
	***********************************************************************************************/
	/** 
	 * @brief Represents the parameters of a discrete time, generic nonlinear state space model:	
	 * 
	 * 		x(k) = f(x(k-1),u(k),w(k)) {Process equation}
	 * 
	 * 		y(k) = h(x(k),v(k)) {Measurement equation}
	 * with \p x(k) being state variables vector, \p u(k) measurement vector and \p y(k) measurement vector. Also, \p w(k) ~ \p N(0,Q) and \p v(k) ~ \p N(0,R) are respectively process and measurement noises.
	 * 
	 * This class contains only model parameters. It is a base virtual class, with computation of process and measurement equations as virtual member functions. 
	 * Its definition should be done in a derived class at user side. 
	 * Partial derivatives of process and measurement models should also be defined at a derived class. Some filters use such derivatives.
	 * 
	 * Initial state is a random variable following \p x(0) ~ \p N(X0,P0)
	 *  
	 * Also dim(x) = \p Nx, dim(u) = \p Nu, dim(y) = \p Ny, dim(w) = \p Nw, dim(v) = \p Nv. 
	 * These dimensions should be first stablished by calling \p SetDimmensions(), which make resizing of system parameters. After that, since parameters are public, they can be set at any time. 
	 * 
	 * Angular measurements in radians need special treatment when computing innovation vector in filters using \p atan2 function. 
	 * In order to do this, \p Y_isangle is used to indicate which components of \p Y are angles, which should be set as \p true by the user when initializing the model.
	 * 
	 * For the same reason, every change in state vector containing angles in some entries should me donne properly. Also, some filters compute covariance matrix from samples, requiring special treatment. 
	 * Thus, \p X_isangle is used to indicate which components of \p x are angles, which should be set as \p true by the user when initializing the model.
	 */
	class StateSpaceNonlinearModelBase
	{
	public:
		Eigen::MatrixXd dfdX; //!< Partial derivative of f(X,U,W) with respect to X
		Eigen::MatrixXd dfdW; //!< Partial derivative of f(X,U,W) with respect to W
		Eigen::MatrixXd dhdX; //!< Partial derivative of h(X,V) with respect to X
		Eigen::MatrixXd dhdV; //!< Partial derivative of h(X,V) with respect to V
		Eigen::MatrixXd Q; //!< Process model noise covariance matrix
		Eigen::MatrixXd R; //!< Measurement model noise covariance matrix
		Eigen::VectorXd X0; //!< Inicial state vector distribution mean vector parameter
		Eigen::MatrixXd P0; //!< Inicial state vector distribution covariance matrix parameter
		int Nx; //!< Number of state variables in X vector
		int Nu; //!< Number of inputs in U vector
		int Ny; //!< Number of measurements in Y vector
		int Nw; //!< Number of process noise in W vector
		int Nv; //!< Number of measurement noise in V vector
		std::vector<bool> X_isangle; //!< Indicates which entries of X vector are angles in rad, for correct computation of correction equation.
		std::vector<bool> Y_isangle; //!< Indicates which entries of Y vector are angles in rad, for correct computation of innovation vector

		 // Always provide a virtual destructor if your class has virtual functions!
		virtual ~StateSpaceNonlinearModelBase() = default; 
		
	/**
	 * @brief Define dimensions of system variables and adjust size of model parameters accordingly.
	 * @param[in] _Nx Number of state variables in X vector
	 * @param[in] _Nu Number of inputs in U vector
	 * @param[in] _Ny Number of measurements in Y vector
	 * @param[in] _Nw Number of process noise in W vector
	 * @param[in] _Nv Number of measurement noise in V vector
	 */
		void SetDimensions(int _Nx, int _Nu, int _Ny, int _Nw, int _Nv)
		{
			Nx = _Nx; Nu =_Nu; Ny = _Ny; Nw = _Nw, Nv = _Nv;			
			dfdX.resize(Nx, Nx);
			dfdW.resize(Nx, Nw);
			dhdX.resize(Ny, Nx);
			dhdV.resize(Ny, Nv);
			Q.resize(Nw, Nw);
			R.resize(Nv, Nv);
			X0.resize(Nx);
			P0.resize(Nx, Nx);
			X_isangle.resize(Nx);
			std::fill(X_isangle.begin(), X_isangle.end(), false);
			Y_isangle.resize(Ny);
			std::fill(Y_isangle.begin(), Y_isangle.end(), false);
		}

	/**
	 * @brief Compute process model to update state vector \p X (in place). Virtual function to be defined in derived class.
	 * @param[in,out] X state vector with \p x(k-1) to be updated to \p x(k)
	 * @param[in] U input vector
	 * @param[in] t time instant scalar
	 * @param[in] W current sample of process noise
	 */
		virtual Result ProcessModel(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W)=0; // Deve ser definido pelo usuario
	/**
	 * @brief Compute system output \p Y from measurement model. Virtual function to be defined in derived class. 
	 * @param[out] Y the computed output
	 * @param[in] X state vector with \p x(k)
	 * @param[in] t time instant scalar
	 * @param[in] V current sample of measurement noise
	 */
		virtual	Result MeasurementModel(Eigen::VectorXd & Y, Eigen::VectorXd & X, double t, Eigen::VectorXd & V)=0; // Deve ser definido pelo usuario
	/**
	 * @brief Update dfdX internal class member, the process model derivative with respect to \p x(k-1). Necessary only in filters using system matrix derivatives, where this function is called from inside the filter class. Virtual function to be defined in derived class.
	 * @param[in] X state vector containing \p x(k-1)
	 * @param[in] U input vector
	 * @param[in] t time instant scalar
	 * @param[in] W current sample of process noise
	 */
		virtual	Result UpdatedfdX(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W)=0; // Deve ser definido pelo usuario
	/**
	 * @brief Update dfdW internal class member, the process model derivative with respect to \p w(k). Necessary only in filters using system matrix derivatives, where this function is called from inside the filter class. Virtual function to be defined in derived class.
	 * @param[in] X state vector containing \p x(k-1)
	 * @param[in] U input vector
	 * @param[in] t time instant scalar
	 * @param[in] W current sample of process noise
	 */
		virtual	Result UpdatedfdW(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W)=0; // Deve ser definido pelo usuario
	/**
	 * @brief Update dhdX internal class member, the measurement model derivative with respect to \p x(k). Necessary only in filters using system matrix derivatives, where this function is called from inside the filter class. Virtual function to be defined in derived class.
	 * @param[in] X state vector containing \p x(k-1)
	 * @param[in] t time instant scalar
	 * @param[in] V current sample of measurement noise
	 */
		virtual	Result UpdatedhdX(Eigen::VectorXd & X, double t, Eigen::VectorXd & V)=0; // Deve ser definido pelo usuario
	/**
	 * @brief Update dhdV internal class member, the measurement model derivative with respect to \p v(k). Necessary only in filters using system matrix derivatives, where this function is called from inside the filter class. Virtual function to be defined in derived class.
	 * @param[in] X state vector containing \p x(k-1)
	 * @param[in] t time instant scalar
	 * @param[in] V current sample of measurement noise
	 */
		virtual	Result UpdatedhdV(Eigen::VectorXd & X, double t, Eigen::VectorXd & V)=0; // Deve ser definido pelo usuario
	};

	/***********************************************************************************************
	 *  Bayesian filter base class:
	***********************************************************************************************/
	/** 
	 * @brief Base class of minimum variables of a Bayesian stochastic filter. All derived classes present filters for linear and nonlinear models.
	 * All filters follow the following loop
	 * 1. Reset
	 * 2. Prediction: compute x(k|k-1) using x(k-1) and input u(k)
	 * 3. Correction, in case of measurement available: compute x(k|k) using x(k|k-1) and measurement vector y(k);
	 * 4. In case of filter divergence, performe necessary actions and go back to step 1. Otherwise, go back to step 2.
	 * 
	 * As it can be seen, as described above correction is optional, which allows filter continuing by prediction. Internal member variables present the current filter state.
	 * 
	 * System model object should be not used in other purposes other than to be used as input to the filter. 
	 * For the case of linear time varying models, it can have their parameters directly updated just before calling prediction/correction methods, as a way to deal with time varying systems. 
	 * For the case of nonlinear time varying models, parameters updating is done in derived model class, where the methods are called from inside filter object. 
	 * Thus, model updating is done in derived class, being not necessary to be done by the user.
	 * 
	 */
	class BayesianFilter
	{
	public:
		Eigen::VectorXd X; //!< State vector predictor \p x(k|k-1) or estimate \p x(k|k), depending on which stage the filter is.
		Eigen::MatrixXd P; //!< State vector error covariance matrix predictor \p P(k|k-1) or \p P(k|k), depending on which stage the filter is.

		Eigen::VectorXd Y_predicted; //!< Predicted measurement \p y(k|k-1).
		Eigen::VectorXd V_predicted; //!< Predicted innovation \p y(k) - \p y(k|k-1).
		Eigen::MatrixXd S_predicted; //!< Predicted innovation covariance matrix.
		double d2_predicted; //!< Predicted innovation Mahalanobis distance, allowing to verify a filter os oprtimistic ou pessimistic, and divergence.

		Eigen::VectorXd Y_corrected; //!< Corrected measurement \p y(k|k).
		Eigen::VectorXd V_corrected; //!< Corrected innovation \p y(k) - \p y(k|k).
		Eigen::MatrixXd S_corrected; //!< Corrected innovation covariance matrix.
		double d2_corrected; //!< Corrected innovation Mahalanobis distance, allowing to verify a filter os oprtimistic ou pessimistic, and divergence.

		BayesianFilter()
		{
			d2_predicted = 0;
			d2_corrected = 0;
		}
		virtual ~BayesianFilter(){}

	protected: // Only derived classes have access
	
		/**
		 * @brief Reset filter, first procedure before prediction/correction.
		 * @param[in] X0 inicial state vector mean value of \p x(0) distribution
		 * @param[in] P0 inicial state error covariance matrix of \p x(0) distribution
		 * @param[in] Nx Number of state variables in vector \p X
		 * @param[in] Ny Number of measurements in vector \p Y
		 */
		void Reset(Eigen::VectorXd &X0, Eigen::MatrixXd &P0, int Nx, int Ny)
		{
			
			X.resize(Nx, 1);
			P.resize(Nx, Nx);
			Y_predicted.resize(Ny, 1);
			V_predicted.resize(Ny, 1);
			S_predicted.resize(Ny, Ny);
			Y_corrected.resize(Ny, 1);
			V_corrected.resize(Ny, 1);
			S_corrected.resize(Ny, Ny);

			d2_predicted = 0;
			d2_corrected = 0;
			X = X0;
			P = P0;
		}		
	};

	/** 
	 * @brief Base class of Bayesian filters for state space linear models. See derived classes.
	 */
	class LinearBayesianFilter : public BayesianFilter
	{
	public:
		virtual ~LinearBayesianFilter() {std::cout << "\nLinearBayesianFilter Base destroyed\n";}

		virtual Result Prediction(Eigen::VectorXd & U, StateSpaceLinearModel& linear_model)=0;
		virtual Result Correction(Eigen::VectorXd & Y, StateSpaceLinearModel& linear_model)=0;
		virtual void Reset(StateSpaceLinearModel& linear_model)=0;

	protected:
		Result CheckModel(StateSpaceLinearModel& linear_model)
		{
			if (linear_model.A.rows()!=X.rows()) return ResultFormat(false,"Number of rows of A should be the same number of rows of X");
			if (linear_model.A.cols()!=X.rows()) return ResultFormat(false,"Number of cols of A should be the same number of rows of X");
			if (linear_model.B.rows()!=X.rows()) return ResultFormat(false,"Number of rows of B should be the same number of rows of X");
			return ResultFormat(true,"");
		}
	};

	/** 
	 * @brief Kalman filter, the best optimal unbiased estimator for stochastic linear state space models. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 */
	class KalmanFilter : public LinearBayesianFilter
	{
	public:
		virtual ~KalmanFilter() {std::cout << "\nKalmanFilter derived class destroyed\n";}

		/**
		 * @brief Reset filter, first procedure before prediction/correction.
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		void Reset(StateSpaceLinearModel& linear_model) override
		{
			BayesianFilter::Reset(linear_model.X0, linear_model.P0, linear_model.Nx, linear_model.Ny);
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Prediction(Eigen::VectorXd & U, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;

			// Predicao
			X = linear_model.A * X + linear_model.B * U;
			P = linear_model.A * P * linear_model.A.transpose() + linear_model.Q;
			return ResultFormat(true,"");
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Correction(Eigen::VectorXd & Y, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;

			// Correcao
			Y_predicted = linear_model.C * X;
			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(linear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			S_predicted = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);

			Eigen::MatrixXd G = P * linear_model.C.transpose() * S_predicted.inverse();
			Eigen::MatrixXd Gi = G * V_predicted;
			for(int i=0; i < Gi.rows(); ++i){
				if(linear_model.X_isangle[i]) Gi(i) = atan2(sin(Gi(i)),cos(Gi(i)));
			}
			X = X + Gi;
			//P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P;
			// More numericaly stable P update:
			P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P *((Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C).transpose()) + G*linear_model.R*G.transpose(); 

			Y_corrected = linear_model.C * X; 
			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(linear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			S_corrected = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);
			
			return ResultFormat(true,"");
		}
	};

	/** 
	 * @brief Adaptive Kalman filter, with \p Q and \p R matrices changing when every measurement comes according two user supplied learning parameters. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 */
	class AdaptiveKalmanFilter : public LinearBayesianFilter
	{
	public:
		double alpha; //!< User supplied learning parameter for updating \p R measurement noise covariance matrix. Must be in interval ]0,1], with 1 leading to fixed \p R matrix.
		double beta;  //!< User supplied learning parameter for updating \p Q process noise covariance matrix. Must be in interval ]0,1], with 1 leading to fixed \p Q matrix.

		virtual ~AdaptiveKalmanFilter() {std::cout << "\nAdaptiveKalmanFilter derived class destroyed\n";}

		/**
		 * @brief Reset filter with user supplied filter specific parameters. Must be called before the prediction/correction cycle.
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 * @param[in] _alpha user provided value for \p alpha member variable
		 * @param[in] _beta user provided value for \p beta member variable
		 */
		void Reset(StateSpaceLinearModel& linear_model, double _alpha, double _beta)
		{
			alpha = _alpha;
			beta = _beta;
			BayesianFilter::Reset(linear_model.X0, linear_model.P0, linear_model.Nx, linear_model.Ny);
		}

		/**
		 * @brief Reset filter with default filter specific parameters (\p alpha = 0.95 and \p beta = 0.99). Must be called before the prediction/correction cycle.
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		void Reset(StateSpaceLinearModel& linear_model) override
		{
			Reset(linear_model, 0.95, 0.99);
			
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Prediction(Eigen::VectorXd & U, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;
			// Predicao
			X = linear_model.A * X + linear_model.B * U;
			P = linear_model.A * P * linear_model.A.transpose() + linear_model.Q;
			return ResultFormat(true,"");
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Correction(Eigen::VectorXd & Y, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;
			
			// Innovation
			Y_predicted = linear_model.C * X;
			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(linear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			S_predicted = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);
			Eigen::MatrixXd P_predicted = P;

			// State update
			Eigen::MatrixXd G = P * linear_model.C.transpose() * S_predicted.inverse();
			Eigen::MatrixXd Gi = G * V_predicted;
			for(int i=0; i < Gi.rows(); ++i){
				if(linear_model.X_isangle[i]) Gi(i) = atan2(sin(Gi(i)),cos(Gi(i)));
			}
			X = X + Gi;
			//P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P;
			// More numericaly stable P update:
			P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P *((Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C).transpose()) + G*linear_model.R*G.transpose(); 

			// Innovation corrected
			Y_corrected = linear_model.C * X; 
			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(linear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			S_corrected = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);

			// R and Q updates
			linear_model.R = alpha * linear_model.R + (1.0-alpha)*(V_predicted * V_predicted.transpose() + linear_model.C * P_predicted * linear_model.C.transpose());
			linear_model.Q = beta * linear_model.Q + (1.0-beta)*(G * V_predicted * V_predicted.transpose() * G.transpose());

			return ResultFormat(true,"");

		}
	};

	/** 
	 * @brief Robust Self Adaptive Kalman filter, with \p Q and \p R matrices changing when every measurement comes using covariance matching strategy. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 * 
	 * From: Chen Y-W, Tu K-M. Robust self-adaptive Kalman filter with application in target tracking. Measurement and Control. 2022;55(9-10):935-944. doi:10.1177/00202940221083548
	 */	
	class RobustSelfAdaptiveKalmanFilter : public LinearBayesianFilter
	{
		Eigen::MatrixXd P_corrected; //!< Previous correction \P matrix, used internly in the algorithm.

	public:

		virtual ~RobustSelfAdaptiveKalmanFilter() {std::cout << "\nRobustSelfAdaptiveKalmanFilter derived class destroyed\n";}

		/**
		 * @brief Reset filter, first procedure before prediction/correction.
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		void Reset(StateSpaceLinearModel& linear_model) override
		{
			BayesianFilter::Reset(linear_model.X0, linear_model.P0, linear_model.Nx, linear_model.Ny);
			P_corrected = linear_model.P0;
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Prediction(Eigen::VectorXd & U, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;
			// Predicao
			X = linear_model.A * X + linear_model.B * U;
			P = linear_model.A * P * linear_model.A.transpose() + linear_model.Q;
			return ResultFormat(true,"");
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] linear_model \p StateSpaceLinearModel object for which the filter will run
		 */
		Result Correction(Eigen::VectorXd & Y, StateSpaceLinearModel& linear_model) override
		{
			Result res = CheckModel(linear_model); if(!res.first) return res;
			
			// Ohmega calculation
			double sigma = linear_model.R(0,0);
			double epsilon = abs(V_corrected(0,0));
			double ohmega = 0.0;
			if (sigma > 0.0){
				ohmega = (epsilon <= sigma) ? 1.0 : sigma/epsilon; 
			}

			// R update
			linear_model.R = ohmega * linear_model.R + (1-ohmega)*(V_corrected * V_corrected.transpose() + linear_model.C * P_corrected * linear_model.C.transpose());

			// Innovation
			Y_predicted = linear_model.C * X;
			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(linear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			S_predicted = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);

			// State update
			Eigen::MatrixXd G = P * linear_model.C.transpose() * S_predicted.inverse();
			Eigen::MatrixXd Gi = G * V_predicted;
			for(int i=0; i < Gi.rows(); ++i){
				if(linear_model.X_isangle[i]) Gi(i) = atan2(sin(Gi(i)),cos(Gi(i)));
			}
			X = X + Gi;
			//P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P;
			// More numericaly stable P update:
			P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C) * P *((Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * linear_model.C).transpose()) + G*linear_model.R*G.transpose(); 

			// Innovation corrected
			Y_corrected = linear_model.C * X; 
			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(linear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			S_corrected = linear_model.C * P * linear_model.C.transpose() + linear_model.R;
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);

			// Q update
			linear_model.Q = ohmega * linear_model.Q + (1-ohmega)*(G * V_corrected * V_corrected.transpose() * G.transpose());
	
			// P corrected, to be used in next interaction
			P_corrected = P;

			return ResultFormat(true,"");

		}
	};

	/** 
	 * @brief Base class of Bayesian filters for state space nonlinear models. See derived classes.
	 */
	class NonlinearBayesianFilter : public BayesianFilter
	{
	public:
		virtual ~NonlinearBayesianFilter() {std::cout << "\nNonlinearBayesianFilter Base destroyed\n";}

		virtual Result Prediction(Eigen::VectorXd & U, double t, StateSpaceNonlinearModelBase& nonlinear_model)=0;
		virtual Result Correction(Eigen::VectorXd & Y, double t, StateSpaceNonlinearModelBase& nonlinear_model)=0;
		virtual void Reset(StateSpaceNonlinearModelBase& nonlinear_model)=0;
	};

	/** 
	 * @brief Extended Kalman filter, the first order approximation-based estimator for stochastic nonlinear state space models. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 */
	class ExtendedKalmanFilter : public NonlinearBayesianFilter
	{
	public:
		virtual ~ExtendedKalmanFilter() {std::cout << "\nExtendedKalmanFilter derived class destroyed\n";}

		/**
		 * @brief Reset filter, first procedure before prediction/correction.
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run
		 */
		void Reset(StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			BayesianFilter::Reset(nonlinear_model.X0, nonlinear_model.P0, nonlinear_model.Nx, nonlinear_model.Ny);
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p nonlinear_model. This procedure calls methods of the derived class object \p nonlinear_model in order to update model parameters. So there is no need the user to do that. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Prediction(Eigen::VectorXd & U, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			// Update model (linearized on previous correction state)
			Eigen::VectorXd W = Eigen::MatrixXd::Zero(nonlinear_model.dfdW.rows(), 1);
			nonlinear_model.UpdatedfdX(X,U,t,W);
			nonlinear_model.UpdatedfdW(X,U,t,W);
						
			// Predicao
			Result res = nonlinear_model.ProcessModel(X,U,t,W);
			if(!res.first) return res;
			P = nonlinear_model.dfdX * P * nonlinear_model.dfdX.transpose() + nonlinear_model.dfdW*nonlinear_model.Q*nonlinear_model.dfdW.transpose();
			
			return res;
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Correction(Eigen::VectorXd & Y, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Result res;

			// Update model (linearized on predicted state)
			
			Eigen::VectorXd V = Eigen::VectorXd::Zero(nonlinear_model.dhdV.rows(), 1);
			nonlinear_model.UpdatedhdX(X,t,V);
			nonlinear_model.UpdatedhdV(X,t,V);
			
			// State correction:
			res = nonlinear_model.MeasurementModel(Y_predicted,X,t,V);
			if(!res.first) return res;
			
   			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			S_predicted = nonlinear_model.dhdX * P * nonlinear_model.dhdX.transpose() + nonlinear_model.dhdV * nonlinear_model.R * nonlinear_model.dhdV.transpose();
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);
						
			Eigen::MatrixXd G = P * nonlinear_model.dhdX.transpose() * S_predicted.inverse(); 
			Eigen::MatrixXd Gi = G * V_predicted;
			for(int i=0; i < Gi.rows(); ++i){
				if(nonlinear_model.X_isangle[i]) Gi(i) = atan2(sin(Gi(i)),cos(Gi(i)));
			}
			X = X + Gi;
			// P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * nonlinear_model.dhdX) * P; 
			// More numericaly stable P update:
			P = (Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * nonlinear_model.dhdX) * P *((Eigen::MatrixXd::Identity(X.rows(), X.rows()) - G * nonlinear_model.dhdX).transpose()) + G*nonlinear_model.R*G.transpose(); 
			
			// Update model (linearized on corrected state)
			V = Eigen::VectorXd::Zero(nonlinear_model.dhdV.rows(), 1);
			nonlinear_model.UpdatedhdX(X,t,V);
			nonlinear_model.UpdatedhdV(X,t,V);
			
			// Generate corrected output, innovation and d^2:
			res = nonlinear_model.MeasurementModel(Y_corrected,X,t,V);
			if(!res.first) return res;
			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			S_corrected = nonlinear_model.dhdX * P * nonlinear_model.dhdX.transpose() + nonlinear_model.dhdV * nonlinear_model.R * nonlinear_model.dhdV.transpose();
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);
			
			return res;

		}
	};

	/** 
	 * @brief Unscented Kalman filter, the Unscented Transform (UT)-based estimator for stochastic nonlinear state space models. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 */	
	class UnscentedKalmanFilter : public NonlinearBayesianFilter
	{
	private:
		/**
		 * @brief Initialize sigma weights. For internal use.
		 * @param[out] Wsigma vector filled with weights
		 * @param[in] kappa parameter of weighting function
		 * @param[in] Nx dimension of state vector
		 */
		void InitializeWeights(Eigen::VectorXd & Wsigma, double & kappa, int Nx)
		{
			int Nsigma = 2*Nx + 1;
			
			Wsigma.resize(Nsigma,1);
		
			kappa = (3 > Nx-3) ? 3 : Nx-3;
			Wsigma(0) = kappa/(Nx+kappa);
			for (int i = 1; i < Wsigma.rows(); ++i) {
				Wsigma(i) = 1/(2.0*(Nx+kappa));
			}
			Wsigma = Wsigma / Wsigma.sum(); // Garante soma 1.
		}

	public:
		// Override virtual members:
		virtual ~UnscentedKalmanFilter() {std::cout << "\nUnscentedKalmanFilter derived class destroyed\n";}
		
		/**
		 * @brief Reset filter, first procedure before prediction/correction.
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run
		 */
		void Reset(StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			BayesianFilter::Reset(nonlinear_model.X0, nonlinear_model.P0, nonlinear_model.Nx, nonlinear_model.Ny);
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p nonlinear_model. This procedure calls methods of the derived class object \p nonlinear_model in order to update model parameters. So there is no need the user to do that. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Prediction(Eigen::VectorXd & U, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Result res = BayesianFiltering::ResultFormat(true,"");

			// Prediction
			// The function x(k) = f(x(k-1),u(k),w(k),t(k)) is converted to x(k) = f(xa(k),u(k),t(k)) with xa(k) = [x(k-1); w(k)]
			int Nx = nonlinear_model.Nx;
			int Nw = nonlinear_model.Nw;
			int DimX = Nx + Nw;
			int DimY = Nx;

			Eigen::VectorXd Wsigma;
			double kappa;
			InitializeWeights(Wsigma, kappa, DimX);
			int Nsigma = Wsigma.rows();
			
			// Generate augmented Xa and Pa
			Eigen::VectorXd Xa(DimX); 
			Xa << X, Eigen::VectorXd::Zero(Nw);
			
			Eigen::MatrixXd Pa(DimX,DimX); 
			Pa << P, Eigen::MatrixXd::Zero(Nx,Nw),
				  Eigen::MatrixXd::Zero(Nw, Nx), nonlinear_model.Q;

			// Generate sigma points:
			Eigen::MatrixXd Xsigma(DimX,Nsigma);
			Eigen::MatrixXd Ysigma(DimY,Nsigma);

			Eigen::LLT<Eigen::MatrixXd> llt((DimX+kappa)*Pa);
			Eigen::MatrixXd L = llt.matrixL();	

			Eigen::VectorXd Xi(Nx);
			Eigen::VectorXd Wi(Nw);
			
			for (int i = 0; i < Nsigma; ++i) {
				if (i==0){
					Xsigma.col(i) = Xa;
				}
				else if (i <= DimX){
					Xsigma.col(i) = Xa + L.col(i-1);
				}
				else{
					Xsigma.col(i) = Xa - L.col(i-1-DimX);
				}				
				Xi = Xsigma.block(0 , i, Nx, 1);
				Wi = Xsigma.block(Nx, i, Nw, 1);
				Result res = nonlinear_model.ProcessModel(Xi,U,t,Wi);
				if(!res.first) return res;
				Ysigma.col(i) = Xi;
			}
			// Compute prediction
			X = Eigen::VectorXd::Zero(Nx);
			for (int i = 0; i < Nsigma; ++i) {
				X += Wsigma(i)*Ysigma.col(i);
			}
			P = Eigen::MatrixXd::Zero(Nx,Nx);
			for (int i = 0; i < Nsigma; ++i) {
				P += Wsigma(i)*(Ysigma.col(i)-X)*((Ysigma.col(i)-X).transpose());
			}

			return res;
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Correction(Eigen::VectorXd & Y, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Result res = BayesianFiltering::ResultFormat(true,"");

			// Correction
			// The function y(k) = h(x(k),v(k)) is converted to x(k) = h(xa(k)) with xa(k) = [x(k); v(k)]
			int Nx = nonlinear_model.Nx;
			int Ny = nonlinear_model.Ny;
			int Nv = nonlinear_model.Nv;
			int DimX = Nx + Nv;
			int DimY = Ny;

			Eigen::VectorXd Wsigma;
			double kappa;
			InitializeWeights(Wsigma, kappa, DimX);
			int Nsigma = Wsigma.rows();

			// Generate augmented Xa and Pa
			Eigen::VectorXd Xa(DimX); 
			Xa << X, Eigen::VectorXd::Zero(Nv);
			
			Eigen::MatrixXd Pa(DimX,DimX); 
			Pa << P, Eigen::MatrixXd::Zero(Nx,Nv),
				  Eigen::MatrixXd::Zero(Nv,Nx), nonlinear_model.R;

			// Generate sigma points:
			Eigen::MatrixXd Xsigma(DimX,Nsigma); Xsigma = Eigen::MatrixXd::Zero(DimX,Nsigma);
			Eigen::MatrixXd Ysigma(DimY,Nsigma); Ysigma = Eigen::MatrixXd::Zero(DimY,Nsigma);

			Eigen::MatrixXd L;
			{
				Eigen::LLT<Eigen::MatrixXd> llt((DimX+kappa)*Pa);
				L = llt.matrixL();	
			}

			Eigen::VectorXd Xi(Nx);
			Eigen::VectorXd Vi(Nv);
			Eigen::VectorXd Yi(Ny);
			for (int i = 0; i < Nsigma; ++i) {
				if (i==0){
					Xsigma.col(i) = Xa;
				}
				else if (i <= DimX){
					Xsigma.col(i) = Xa + L.col(i-1);
				}
				else{
					Xsigma.col(i) = Xa - L.col(i-1-DimX);
				}
				Xi = Xsigma.block(0 , i, Nx, 1);
				Vi = Xsigma.block(Nx, i, Nv, 1);
				res = nonlinear_model.MeasurementModel(Yi,Xi,t,Vi); if(!res.first) return res;
				Ysigma.col(i) = Yi;
			}
			// Compute correction
			Y_predicted = Eigen::VectorXd::Zero(Ny);
			for (int i = 0; i < Nsigma; ++i) {
				Y_predicted += Wsigma(i)*Ysigma.col(i);
			}
			Eigen::MatrixXd Pyy = Eigen::MatrixXd::Zero(Nx,Nx);
			Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(Nx,Ny);
			for (int i = 0; i < Nsigma; ++i) {
	   			Eigen::VectorXd dX = Xsigma.block(0 , i, Nx, 1)-X;
				for(int j=0; j < dX.rows(); ++j){
					if(nonlinear_model.X_isangle[j]) dX(j) = atan2(sin(dX(j)),cos(dX(j)));
				}
	   			Eigen::VectorXd dY = Ysigma.col(i)-Y_predicted;
				for(int j=0; j < dY.rows(); ++j){
					if(nonlinear_model.Y_isangle[j]) dY(j) = atan2(sin(dY(j)),cos(dY(j)));
				}
				Pyy += Wsigma(i)*(dY)*(dY.transpose());
				Pxy += Wsigma(i)*(dX)*(dY.transpose());
			}
   			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			S_predicted = Pyy;
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);
			
			Eigen::MatrixXd G = Pxy * Pyy.inverse(); 
			Eigen::MatrixXd Gi = G * V_predicted;
			for(int i=0; i < Gi.rows(); ++i){
				if(nonlinear_model.X_isangle[i]) Gi(i) = atan2(sin(Gi(i)),cos(Gi(i)));
			}
			X = X + Gi;
			P = P - G*Pyy*G.transpose(); 
			
			// Compute corrected output, innovation and d^2:
			Xa << X, Eigen::VectorXd::Zero(Nv);
			Pa << P, Eigen::MatrixXd::Zero(Nx,Nv),
				  Eigen::MatrixXd::Zero(Nv,Nx), nonlinear_model.R;

			{
				Eigen::LLT<Eigen::MatrixXd> llt((DimX+kappa)*Pa);
				L = llt.matrixL();	
			}
			for (int i = 0; i < Nsigma; ++i) {
				if (i==0){
					Xsigma.col(i) = Xa;
				}
				else if (i <= DimX){
					Xsigma.col(i) = Xa + L.col(i-1);
				}
				else{
					Xsigma.col(i) = Xa - L.col(i-1-DimX);
				}				
				Xi = Xsigma.block(0 , i, Nx, 1);
				Vi = Xsigma.block(Nx, i, Nv, 1);
				res = nonlinear_model.MeasurementModel(Yi,Xi,t,Vi); if(!res.first) return res;
				Ysigma.col(i) = Yi;
			}
			Y_corrected = Eigen::VectorXd::Zero(Ny);
			for (int i = 0; i < Nsigma; ++i) {
				Y_corrected += Wsigma(i)*Ysigma.col(i);
			}
			Pyy = Eigen::MatrixXd::Zero(Nx,Nx);
			for (int i = 0; i < Nsigma; ++i) {
				Pyy += Wsigma(i)*(Ysigma.col(i)-Y_corrected)*((Ysigma.col(i)-Y_corrected).transpose());
			}			
			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			S_corrected = Pyy;
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);
			
			return res;
		}

	};
	
	/** 
	 * @brief Sampling Importance Resampling (SIR) filter, particle-based estimator for stochastic nonlinear state space models. 
	 * 
	 * This implementation is specific for discrete time, continuous state models.  
	 * Follows algorthim described at: M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking", in IEEE Transactions on Signal Processing, vol. 50, no. 2, pp. 174-188, Feb. 2002, doi: 10.1109/78.978374.
	 */	
	class SamplingImportanceResamplingFilter : public NonlinearBayesianFilter
	{
	private:
		/**
		 * @brief Reset particles weight to 1/Nparticles
		 */
		void ResetWeights()
		{
			long int Nparticles = Wparticles.rows();
			Wparticles.setConstant(1.0/((double)(Nparticles)));
		}

	public:
		/**
		 * @brief Enum of type of estimate computed by the filter at \p X vector.
		 */
		enum class EstimateType
		{
			MaximumAPosteriori, //!< Maximum a Posteriori (MAP) estimate
			ConditionalMean, //!< Conditional mean estimate
		};

		/**
		 * @brief Generate text description Enum of type of estimate computed by the filter at \p X vector.
		 */
        static inline const char* EnumToString(EstimateType x)
        {
            const std::map<EstimateType,const char*> Strings {
                { EstimateType::MaximumAPosteriori, "Maximum a Posteriori estimate" },
                { EstimateType::ConditionalMean, "Conditional Mean estimate" },
            };
            auto   it  = Strings.find(x);
            return it == Strings.end() ? "Out of range" : it->second;
        }

		// Specific data
		EstimateType estimate_type; //!< Estimation type to be provided by the filter at \p X vector
		long int Nparticles; 		//!< Number of particles
		Eigen::MatrixXd Xparticles; //!< \p X particles, one per col
		Eigen::MatrixXd Yparticles; //!< \p Y particles, one per col
		Eigen::VectorXd Wparticles; //!< Weight for each particle, set to 1/Nparticles after resampling
		double ESS; 				//!< Effective Sample Size, health of particles distribution. Read after filter correction.

		// Override virtual members:
		virtual ~SamplingImportanceResamplingFilter() {std::cout << "\nSamplingImportanceResamplingFilter derived class destroyed\n";}
		
		/**
		 * @brief Reset filter with user supplied filter specific parameters. Must be called before the prediction/correction cycle.
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase object for which the filter will run
		 * @param[in] _Nparticles user provided value for \p Nparticles member variable
		 * @param[in] _estimate_type user provided value for \p estimate_type member variable
		 */
		void Reset(StateSpaceNonlinearModelBase& nonlinear_model, long int _Nparticles, EstimateType _estimate_type)
		{
			BayesianFilter::Reset(nonlinear_model.X0, nonlinear_model.P0, nonlinear_model.Nx, nonlinear_model.Ny);
			Nparticles = _Nparticles;
			estimate_type = _estimate_type;
			Xparticles.resize(nonlinear_model.Nx, Nparticles);
			Yparticles.resize(nonlinear_model.Ny, Nparticles);
			Wparticles.resize(Nparticles);
			ResetWeights();
			ESS = Nparticles;

			Result res = BayesianFiltering::ResultFormat(true,"");
			Eigen::VectorXd Xi(nonlinear_model.Nx);
			for (long int i = 0; i < Nparticles; ++i) {
				Xparticles.col(i) = X;
				res = MultivariateZeroMeanGaussianSampler(Xi, P);
				if(res.first){
					Xparticles.col(i) = Xi + X;
				}
			}			

		}

		/**
		 * @brief Reset filter with default filter specific parameters (\p Nparticles = 1000 and \p estimate_type = \p EstimateType::ConditionalMean). Must be called before the prediction/correction cycle.
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase object for which the filter will run
		 */
		void Reset(StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Reset(nonlinear_model, 1000, EstimateType::ConditionalMean); // default number of particles
		}

		/**
		 * @brief Normalize particle's weight such that sum(\p Wparticles) = 1.0.
		 */
		Result NormalizeWeights()
		{
			if (Wparticles.sum() > 0){
				double c = 1.0/Wparticles.sum();
				Wparticles *= c;
			}
			else{
				return ResultFormat(false, "SIR filter diverged: sum of weights = %f", Wparticles.sum());
			}
			return BayesianFiltering::ResultFormat(true,"");
		}

		/**
		 * @brief Update ESS (Effective Sample Size) member variable.
		 */
		Result ComputeESS() 
		{
			double sum = 0;

			for (long int i = 0; i < Wparticles.rows(); ++i) {
				sum += Wparticles(i)*Wparticles(i);
			}
			if (sum <= 0.0){
				return ResultFormat(false, "SIR filter diverged: sum of square weights = %f", sum);
			}
			ESS = 1/sum;
			return BayesianFiltering::ResultFormat(true,"");
		}

		/**
		 * @brief Prediction procedure, which will update (in place) the internal member variables \p X and \p P based on input vector \p U and model \p nonlinear_model. This procedure calls methods of the derived class object \p nonlinear_model in order to update model parameters. So there is no need the user to do that. 
		 * @param[in] U input vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Prediction(Eigen::VectorXd & U, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Result res = BayesianFiltering::ResultFormat(true,"");

			// Prediction
			// Generic nonlinear model x(k) = f(x(k-1),u(k),w(k),t(k))
			int Nx = nonlinear_model.Nx;
			int Nw = nonlinear_model.Nw;
			long int Nparticles = Xparticles.cols();

			ResetWeights();

			Eigen::VectorXd Xi(Nx);
			Eigen::VectorXd Wi(Nw);
			for (long int i = 0; i < Nparticles; ++i) {
				Xi = Xparticles.col(i);
				res = MultivariateZeroMeanGaussianSampler(Wi, nonlinear_model.Q);
				if(!res.first) return res;
				res = nonlinear_model.ProcessModel(Xi,U,t,Wi);
				if(!res.first) return res;
				Xparticles.col(i) = Xi;
			}

			// Compute prediction: assuming equal weights, Conditional Mean and Maximum a Posteriori are the same.
			X = Eigen::VectorXd::Zero(Nx);
			for (long int i = 0; i < Nparticles; ++i) {
				X += Wparticles(i)*Xparticles.col(i);
			}
			P = Eigen::MatrixXd::Zero(Nx,Nx);
			for (long int i = 0; i < Nparticles; ++i) {
				P += Wparticles(i)*(Xparticles.col(i)-X)*((Xparticles.col(i)-X).transpose());
			}

			// Particle Weights
			// No need to updade;

			res = ComputeESS(); if(!res.first) return res;

			return res;
		}

		/**
		 * @brief Correction procedure, which will update (in place) the internal member variables \p X and \p P based on measurement vector \p Y and model \p linear_model. In case of time varying systems, \p linear_model class members should be updated before calling this procedure. 
		 * @param[in] Y measurement vector of the linear system process model
		 * @param[in] t time variable, for time varying systems
		 * @param[in] nonlinear_model \p StateSpaceNonlinearModelBase derived object for which the filter will run. Model updating and parameter calculations implemented by the user in the \p StateSpaceNonlinearModelBase derived object are called from inside this method. 
		 */
		Result Correction(Eigen::VectorXd & Y, double t, StateSpaceNonlinearModelBase& nonlinear_model) override
		{
			Result res = BayesianFiltering::ResultFormat(true,"");

			// Correction
			// Generic nonlinear model y(k) = h(x(k),v(k))
			int Nx = nonlinear_model.Nx;
			int Ny = nonlinear_model.Ny;
			int Nv = nonlinear_model.Nv;
			long int Nparticles = Wparticles.rows();

			// Update weights
			// %%% Observar que os pesos q das particulas não são passadas como argumento. De fato, assume-se que estes pesos
			// %%% são todos iguais a 1/Ns, decorrente da ultima execução do estimador. Assim, o calculo da ponderação 
			// %%% q(k) = q(k-1)*p(z(k)|x(k)) seguido da normalização q(k) = q(k)/sum(q) dá na mesma coisa do que fazer 
			// %%% q(k) = p(z(k)|x(k)) seguido de q(k) = q(k)/sum(q), como está implementado. 
			Eigen::VectorXd Xi(Nx);
			Eigen::VectorXd Vi(Nv);
			Eigen::VectorXd Yi(Ny);
			Y_predicted = Eigen::VectorXd::Zero(Ny); // Computer with weights equal, leading to the mean
			for (long int i = 0; i < Nparticles; ++i) {
				Xi = Xparticles.col(i);
				res = MultivariateZeroMeanGaussianSampler(Vi, nonlinear_model.R); if(!res.first) return res;
				res = nonlinear_model.MeasurementModel(Yi,Xi,t,Vi); if(!res.first) return res;
				Y_predicted += Yi; // Simple mean
				Wparticles(i) = exp(((-0.5)*(Y-Yi).transpose()*(nonlinear_model.R.inverse())*(Y-Yi)));				
				Yparticles.col(i) = Yi;
			}
			Y_predicted /= ((double)(Nparticles));
			res = NormalizeWeights(); if(!res.first) return res;

			res = ComputeESS(); if(!res.first) return res;

			S_predicted = Eigen::MatrixXd::Zero(Ny,Ny);
			for (int i = 0; i < Nparticles; ++i) {
	   			V_predicted = Yparticles.col(i)-Y_predicted;
				for(int j=0; j < V_predicted.size(); ++j){
					if(nonlinear_model.Y_isangle[j]) V_predicted(j) = atan2(sin(V_predicted(j)),cos(V_predicted(j)));
				}
				S_predicted += V_predicted*(V_predicted.transpose());
			}
			S_predicted /= ((double)(Nparticles));
   			V_predicted = Y - Y_predicted;
			for(int i=0; i < V_predicted.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_predicted(i) = atan2(sin(V_predicted(i)),cos(V_predicted(i)));
			}
			d2_predicted = (V_predicted.transpose() * S_predicted.inverse() * V_predicted)(0, 0);

			// Resampling
			Eigen::VectorXd WparticlesCumSum(Nparticles);
			WparticlesCumSum(0) = Wparticles(0);
			for (long int i = 1; i < Nparticles; ++i) {
				WparticlesCumSum(i) = WparticlesCumSum(i-1) + Wparticles(i);
			}
			{
				long int i = 0;
				double u1 = UnivariateUniformSampler(0.0, 1.0)/((double)(Nparticles));
				for (long int j = 0; j < Nparticles; ++j) {
					double u = u1 + ((double)(j))/((double)(Nparticles));
					if (u > 1.0){ u = 1.0; }
					while (u > WparticlesCumSum(i)){
						i = i + 1;
						if (i > Nparticles-1){
							i = Nparticles-1;
							break;
						}
					}
					Xparticles.col(j) = Xparticles.col(i);
				}
			}
			ResetWeights();

			// Compute correction
			X = Eigen::VectorXd::Zero(Nx);
			switch (estimate_type){
				case EstimateType::ConditionalMean:
				{
					for (long int i = 0; i < Nparticles; ++i) {
						X += Wparticles(i)*Xparticles.col(i);
					}
				}
				break;
				case EstimateType::MaximumAPosteriori:
				{ 
					long int imax = 0;
					for (long int i = 1; i < Nparticles; ++i) {
						if (Wparticles(i) > Wparticles(imax)) imax = i;
					}					
					X = Xparticles.col(imax);
				}
				break;
			}
			
			P = Eigen::MatrixXd::Zero(Nx,Nx);
			for (long int i = 0; i < Nparticles; ++i) {
	   			Eigen::VectorXd dX = Xparticles.col(i)-X;
				for(int j=0; j < dX.rows(); ++j){
					if(nonlinear_model.X_isangle[j]) dX(j) = atan2(sin(dX(j)),cos(dX(j)));
				}
				P += Wparticles(i)*(dX)*(dX.transpose());
			}
			
			// Compute corrected output, innovation and d^2:
			Y_corrected = Eigen::VectorXd::Zero(Ny); // Computer with weights equal, leading to the mean
			for (long int i = 0; i < Nparticles; ++i) {
				Xi = Xparticles.col(i);
				res = MultivariateZeroMeanGaussianSampler(Vi, nonlinear_model.R); if(!res.first) return res;
				res = nonlinear_model.MeasurementModel(Yi,Xi,t,Vi); if(!res.first) return res;
				Y_corrected += Yi; // Simple mean
				Yparticles.col(i) = Yi;
			}
			Y_corrected /= ((double)(Nparticles));
			S_corrected = Eigen::MatrixXd::Zero(Ny,Ny);
			for (int i = 0; i < Nparticles; ++i) {
	   			V_corrected = Yparticles.col(i)-Y_corrected;
				for(int j=0; j < V_corrected.size(); ++j){
					if(nonlinear_model.Y_isangle[j]) V_corrected(j) = atan2(sin(V_corrected(j)),cos(V_corrected(j)));
				}
				S_corrected += (V_corrected)*(V_corrected.transpose());
			}
			S_corrected /= ((double)(Nparticles));
   			V_corrected = Y - Y_corrected;
			for(int i=0; i < V_corrected.size(); ++i){
				if(nonlinear_model.Y_isangle[i]) V_corrected(i) = atan2(sin(V_corrected(i)),cos(V_corrected(i)));
			}
			d2_corrected = (V_corrected.transpose() * S_corrected.inverse() * V_corrected)(0, 0);

			return res;
		}

	};
}

#endif // BayesianFiltering_H