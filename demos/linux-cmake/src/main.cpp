#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <random>
#include <memory>

// Bayesian filtering header only library:
#include "bayesianfiltering.hpp"

// Extra lib to Eigen, required for .exp() in c2d function of this demo
#include <unsupported/Eigen/MatrixFunctions>

// Matplot++ is a lib for ploting data in C++, no python dependent:
#include <matplot/matplot.h> // https://alandefreitas.github.io/matplotplusplus/

// BayesianFiltering::StateSpaceNonlinearModelBase derived class used in this demo
class NonlinearModel : public BayesianFiltering::StateSpaceNonlinearModelBase
{
    /* Process:  M. Sanjeev Arulampalam, Simon Maskell, Neil Gordon,
        and Tim Clapp, "A Tutorial on Particle Filters for Online
        Nonlinear/Non-Gaussian Bayesian Tracking", IEEE TRANSACTIONS ON SIGNAL
        PROCESSING, VOL. 50, NO. 2, FEBRUARY 2002, pp. 174-188.
        x1(k) = x1(k-1) + 0.5 + w1(k);
        x2(k) = 0.5*x2(k-1)+25*x2(k-1)/(1+x2(k-1)^2)+8*cos(1.2*t(k))+w2(k);
        y1(k) = x1(k)+v1(k);
        y2(k) = (x2(k)^2)/20+v2(k);
    */
public:
	BayesianFiltering::Result ProcessModel(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W) override
    {
        //return nonlinear_ProcessModel(X, U, t, W);
        Eigen::VectorXd Xpos(2,1);  
        Xpos(0,0) = X(0,0) + 0.5 + W(0,0);
        Xpos(1,0) = 0.5*X(1,0)+25*X(1,0)/(1+pow(X(1,0),2))+8.0*cos(1.2*t)+W(1,0);
        X = Xpos;

        return BayesianFiltering::ResultFormat(true,"");        
    }
    BayesianFiltering::Result MeasurementModel(Eigen::VectorXd & Y, Eigen::VectorXd & X, double t, Eigen::VectorXd & V) override
    {
        //return nonlinear_MeasurementModel(Y, X, t, V);
        Y(0,0) = X(0,0) + V(0,0);
        Y(1,0) = pow(X(1,0),2)/20.0+V(1,0);
        
        return BayesianFiltering::ResultFormat(true,"");
    }
    // Necessary only for EKF or other filterisn requiring such derivatives
    BayesianFiltering::Result UpdatedfdX(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W) override
    {
        dfdX(0, 0) = 1.0; dfdX(0, 1) = 0;
        dfdX(1, 0) = 0.0; dfdX(1, 1) = 0.5+(25*(1+pow(X(1,0),2.0))-50.0*(pow(X(1,0),2.0)))/(pow(1.0+pow(X(1,0),2.0),2.0));
        return BayesianFiltering::ResultFormat(true,"");
    }
    // Necessary only for EKF or other filterisn requiring such derivatives
    BayesianFiltering::Result UpdatedfdW(Eigen::VectorXd & X, Eigen::VectorXd & U, double t, Eigen::VectorXd & W) override
    {
        dfdW = Eigen::MatrixXd::Identity(X.rows(), W.rows());
        return BayesianFiltering::ResultFormat(true,"");
    }
    // Necessary only for EKF or other filterisn requiring such derivatives
    BayesianFiltering::Result UpdatedhdX(Eigen::VectorXd & X, double t, Eigen::VectorXd & V) override
    {
        dhdX(0, 0) = 1.0; dhdX(0, 1) = 0;
        dhdX(1, 0) = 0.0; dhdX(1, 1) = 2.0*X(1,0)/20.0;
        return BayesianFiltering::ResultFormat(true,"");
    }
    // Necessary only for EKF or other filterisn requiring such derivatives
    BayesianFiltering::Result UpdatedhdV(Eigen::VectorXd & X, double t, Eigen::VectorXd & V) override
    {
        dhdV = Eigen::MatrixXd::Identity(X.rows(), V.rows());
        return BayesianFiltering::ResultFormat(true,"");
    }
};

// Function to convert continuous state-space to discrete-time via ZOH
void c2d(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, double Ts, Eigen::MatrixXd& Ad, Eigen::MatrixXd& Bd) 
{
    int n = A.rows();
    int m = B.cols();
    
    // Create an augmented matrix of size (n + m) x (n + m)
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + m, n + m);
    
    // Populate the block matrix: [A*Ts , B*Ts ; 0 , 0]
    M.topLeftCorner(n, n) = A * Ts;
    M.topRightCorner(n, m) = B * Ts;
    
    // Compute the matrix exponential
    Eigen::MatrixXd Me = M.exp();
    
    // Extract Ad and Bd from the exponentiated matrix
    Ad = Me.topLeftCorner(n, n);
    Bd = Me.topRightCorner(n, m);
}

// Main:
int main(int argc, char** argv)
{
    std::string demo_name = "";

    if (argc > 1){
        demo_name = argv[1];
    }

    /*************************************************************************/
    // Linear filters
    /*************************************************************************/
    if ((demo_name == "kf")||(demo_name == "rsakf")||(demo_name == "akf")){
        printf("\n*** Linear filtering demo:\n");

        // System models: one used for the filter, another one to simulate the real system
        BayesianFiltering::StateSpaceLinearModel linear_model_filter, linear_model_real;

        // Filter as a unique ptr to allow a single variable be used to different filters, for convinience of the demo.
        std::unique_ptr<BayesianFiltering::LinearBayesianFilter> pfilter;

        // Second order mass-spring system:
        // m*d2pdt2 + c*dpdt + e*p = F
        // m: mass, c: dumping coef., e: spring coef., F: input force, p: position
        // x = [p; dpdt]
        //
        // dxdt =  Ac*x + Bc*y
        // y = Cc*x
        // Ac = [0 1; -e/m -c/m]
        // Bc = [0; 1/m]
        // Cc = [1 0]
        //
        // Using ZOH discrete modeling with sample time h:
        // x(k) = A*x(k-1) + B*u(k) + w(k)
        // y(k) = C*x(k) + v(k)
        // A = (I(2,2)+Ac*h)
        // B = Bc*h
        // C = Cc

        int Nx = 2, Nu = 1, Ny = 1;
        { // Scope limiting, to avoid variables naming confusion with out of scope variables
            double q = 0.1, r = 0.1;
            double h = 0.1;
            double m = 10.0, c = 10.0, e = 50.0; 
            
            Eigen::MatrixXd Ac(2,2); Ac <<  0.0,  1.0,
                                           -e/m, -c/m;
            Eigen::MatrixXd Bc(2,1); Bc <<  0.0,  1.0/m;
            Eigen::MatrixXd Cc(1,2); Cc <<  1.0,  0.0;

            linear_model_real.SetDimensions(Nx,Nu,Ny,Nx,Ny);
            c2d(Ac, Bc, h, linear_model_real.A, linear_model_real.B);
            linear_model_real.C = Cc;
            linear_model_real.Q = Eigen::MatrixXd::Identity(Nx, Nx) * pow(q, 2.0);
            linear_model_real.R = Eigen::MatrixXd::Identity(Ny, Ny) * pow(r, 2.0);
            linear_model_real.X0 << 0, 0;
            linear_model_real.P0 = Eigen::MatrixXd::Identity(Nx, Nx) * 10;

            {
            Eigen::EigenSolver<Eigen::MatrixXd> solver(Ac);
            Eigen::VectorXcd eigenvalues = solver.eigenvalues();
            PRINT_EIGEN_MATRIX(Ac)
            std::cout << "\nContinuous time model poles:\n" << eigenvalues << "\n\n";
            } 
            {
            Eigen::EigenSolver<Eigen::MatrixXd> solver(linear_model_real.A);
            Eigen::VectorXcd eigenvalues = solver.eigenvalues();
            PRINT_EIGEN_MATRIX(linear_model_real.A)
            std::cout << "\nDiscrete time model poles:\n" << eigenvalues << "\n\n";
            }
        }
        linear_model_filter = linear_model_real;

        // Variables to be shown:
        std::vector<double> x1_error, x2_error, v1_predicted, v1_corrected, d2_predicted, d2_corrected;
        std::vector<double> y, u;
        std::vector<double> x1_error_interval_sup, x1_error_interval_inf, x2_error_interval_sup, x2_error_interval_inf;
        std::vector<double> elapsed_time_predict, elapsed_time_correct;
        std::vector<double> trace_Q_filter, trace_Q_real;
        std::vector<double> trace_R_filter, trace_R_real;

        // Inialization of unique pointers to desired filter:
        if (demo_name=="kf"){
            pfilter = std::make_unique<BayesianFiltering::KalmanFilter>();
            pfilter->Reset(linear_model_filter);    
        }
        if (demo_name=="akf"){
            pfilter = std::make_unique<BayesianFiltering::AdaptiveKalmanFilter>();
            double alpha = 0.95, beta = 0.99;
            if (argc>=3) alpha = std::stod(argv[2]);
            if (argc>=4) beta = std::stod(argv[3]);
            dynamic_cast<BayesianFiltering::AdaptiveKalmanFilter*>(pfilter.get())->Reset(linear_model_filter, alpha, beta);
        }
        if (demo_name=="rsakf"){
            pfilter = std::make_unique<BayesianFiltering::RobustSelfAdaptiveKalmanFilter>();
            pfilter->Reset(linear_model_filter);    
        }

        // Main loop:
        BayesianFiltering::Result res;
        Eigen::VectorXd Xreal = linear_model_real.X0;
        for (int k=1; k < 1000; ++k){
            // Real system:
            Eigen::VectorXd U = Eigen::MatrixXd::Identity(Nu, Nu) * (sin(2*M_PI*k/200.0) > 0 ? 100.0 : -100.0);
            Eigen::VectorXd W(2,1);
            res = BayesianFiltering::MultivariateZeroMeanGaussianSampler(W, linear_model_real.Q);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            res = linear_model_real.ProcessModel(Xreal,U,W);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            Eigen::VectorXd Y(1);
            Eigen::VectorXd V(1);
            res = BayesianFiltering::MultivariateZeroMeanGaussianSampler(V, linear_model_real.R);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            res = linear_model_real.MeasurementModel(Y,Xreal,V);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }

            // Prediction:
            auto time_start_predict = std::chrono::steady_clock::now();
            res = pfilter->Prediction(U,linear_model_filter);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            auto time_stop_predict = std::chrono::steady_clock::now();
            double e_predict = 0.001*(std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop_predict - time_start_predict)).count();
        
            // Correction:
            auto time_start_correct = std::chrono::steady_clock::now();
            res = pfilter->Correction(Y,linear_model_filter);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            auto time_stop_correct = std::chrono::steady_clock::now();
            double e_correct = 0.001*(std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop_correct - time_start_correct)).count();

            // Storing variables in vectors to be shown:
            y.push_back(Y(0,0));
            u.push_back(U(0,0));
            x1_error.push_back(Xreal(0,0)-pfilter->X(0,0));
            x2_error.push_back(Xreal(1,0)-pfilter->X(1,0));
            x1_error_interval_sup.push_back(+3.0*sqrt(pfilter->P(0,0)));
            x1_error_interval_inf.push_back(-3.0*sqrt(pfilter->P(0,0)));
            x2_error_interval_sup.push_back(+3.0*sqrt(pfilter->P(1,1)));
            x2_error_interval_inf.push_back(-3.0*sqrt(pfilter->P(1,1)));
            v1_predicted.push_back(pfilter->V_predicted(0,0));
            v1_corrected.push_back(pfilter->V_corrected(0,0));
            d2_predicted.push_back(pfilter->d2_predicted);
            d2_corrected.push_back(pfilter->d2_corrected);
            elapsed_time_predict.push_back(e_predict);
            elapsed_time_correct.push_back(e_correct);
            if ((demo_name=="rsakf")||(demo_name=="akf")){
                trace_Q_real.push_back(linear_model_real.Q.trace());
                trace_Q_filter.push_back(linear_model_filter.Q.trace());
                trace_R_real.push_back(linear_model_real.R.trace());
                trace_R_filter.push_back(linear_model_filter.R.trace());
            }
    
        }
        
        // Plots:
        auto f1 = matplot::figure(); f1->size(800, 800);
        auto f2 = matplot::figure(); f2->size(800, 600);
        auto f3 = matplot::figure(); f3->size(800, 600);

        matplot::figure(f1);
        matplot::subplot(4, 1, 0); matplot::plot(u, "b-"); matplot::title("u"); matplot::ylabel("u");
        matplot::subplot(4, 1, 1); matplot::plot(y, "b-"); matplot::title("y"); matplot::ylabel("y");
        matplot::subplot(4, 1, 2); matplot::plot(x1_error, "b-"); matplot::hold(true); matplot::plot(x1_error_interval_sup, "r-"); matplot::plot(x1_error_interval_inf, "r-"); matplot::title("x_1 error (blue) and 3sigma interval (red)"); matplot::ylabel("x_1(k)-x_1(k|k)");
        matplot::subplot(4, 1, 3); matplot::plot(x2_error, "b-"); matplot::hold(true); matplot::plot(x2_error_interval_sup, "r-"); matplot::plot(x2_error_interval_inf, "r-"); matplot::title("x_2 error (blue) and 3sigma interval (red)"); matplot::xlabel("k"); matplot::ylabel("x_2(k)-x_2(k|k)");

        std::vector<double> x_chisquare = {0, 1000.0};
        std::vector<double> y_chisquare = {3.841, 3.841}; // 5%, 1 dof

        matplot::figure(f2);
        matplot::subplot(2, 2, 0); matplot::plot(v1_predicted, "b-"); matplot::title("innovation v_1(k|k-1)"); 
        matplot::subplot(2, 2, 1); matplot::semilogy(d2_predicted, "b-"); matplot::hold(true); matplot::semilogy(x_chisquare, y_chisquare, "r-"); matplot::title("d^2(k|k-1) (blue) and chi-square 95% limit (red)"); matplot::xlabel("k"); 
        matplot::subplot(2, 2, 2); matplot::plot(v1_corrected, "b-"); matplot::title("innovation v_1(k|k)"); 
        matplot::subplot(2, 2, 3); matplot::semilogy(d2_corrected, "b-"); matplot::hold(true); matplot::semilogy(x_chisquare, y_chisquare, "r-"); matplot::title("d^2(k|k) (blue) and chi-square 95% limit (red)"); matplot::xlabel("k"); 

        matplot::figure(f3);
        matplot::semilogy(elapsed_time_predict, "b-"); matplot::hold(true); matplot::semilogy(elapsed_time_correct, "r-"); matplot::title("Computing times to predict (blue) and correct (red)"); matplot::xlabel("k"); matplot::ylabel("[us]"); 

        if ((demo_name=="rsakf")||(demo_name=="akf")){
            printf("\n*** Adaptive filter results:");
            PRINT_EIGEN_MATRIX(linear_model_real.Q)
            printf("\n    linear_model_real.Q trace: %.10f\n", linear_model_real.Q.trace());
            PRINT_EIGEN_MATRIX(linear_model_filter.Q)
            printf("\n    linear_model_filter.Q trace: %.10f\n", linear_model_filter.Q.trace());
            PRINT_EIGEN_MATRIX(linear_model_real.R)
            printf("\n    linear_model_real.R trace: %.10f\n", linear_model_real.R.trace());
            PRINT_EIGEN_MATRIX(linear_model_filter.R)
            printf("\n    linear_model_filter.R trace: %.10f\n\n", linear_model_filter.R.trace());
            fflush(stdout);

            auto f4 = matplot::figure(); f4->size(800, 600);
            matplot::figure(f4);
            matplot::subplot(2, 1, 0); matplot::plot(trace_Q_filter, "b-")->line_width(2); matplot::hold(true); matplot::plot(trace_Q_real, "r-")->line_width(2);  matplot::title("Filter Q matrix trace (blue) and real (red)"); matplot::xlabel("k"); matplot::ylabel("Trace"); 
            matplot::subplot(2, 1, 1); matplot::plot(trace_R_filter, "b-")->line_width(2); matplot::hold(true); matplot::plot(trace_R_real, "r-")->line_width(2);  matplot::title("Filter R matrix trace (blue) and real (red)"); matplot::xlabel("k"); matplot::ylabel("Trace"); matplot::xlabel("k"); 
        }

        matplot::show();
        printf("\n\n");
        
        return 0;
    }

    /*************************************************************************/
    // Nonlinear filters
    /*************************************************************************/
    if ((demo_name == "ekf")||(demo_name == "ukf")||(demo_name == "sirf")){
        printf("*** Nonlinear filtering demo:\n");
        /* Process:  M. Sanjeev Arulampalam, Simon Maskell, Neil Gordon,
            and Tim Clapp, "A Tutorial on Particle Filters for Online
            Nonlinear/Non-Gaussian Bayesian Tracking", IEEE TRANSACTIONS ON SIGNAL
            PROCESSING, VOL. 50, NO. 2, FEBRUARY 2002, pp. 174-188.
            x1(k) = x1(k-1) + 0.5 + w1(k);
            x2(k) = 0.5*x2(k-1)+25*x2(k-1)/(1+x2(k-1)^2)+8*cos(1.2*t(k))+w2(k);
            y1(k) = x1(k)+v1(k);
            y2(k) = (x2(k)^2)/20+v2(k);
        */
        
        // System models: one used for the filter, another one to simulate the real system
        NonlinearModel nonlinear_model_filter, nonlinear_model_real;

        // Filter as a unique ptr to allow a single variable be used to different filters, for convinience of the demo.
        std::unique_ptr<BayesianFiltering::NonlinearBayesianFilter> pfilter;

        // Model initialization
        int Nx = 2, Nu = 0, Ny = 2;
        nonlinear_model_real.SetDimensions(Nx,Nu,Ny,Nx,Ny);
		nonlinear_model_real.Q(0, 0) = 0.001; nonlinear_model_real.Q(0, 1) = 0.0;
		nonlinear_model_real.Q(1, 0) = 0.0; nonlinear_model_real.Q(1, 1) = 2.0;
		nonlinear_model_real.R(0, 0) = 0.01; nonlinear_model_real.R(0, 1) = 0.0;
		nonlinear_model_real.R(1, 0) = 0.0; nonlinear_model_real.R(1, 1) = 0.05;
		nonlinear_model_real.X0 << 0.8, -1.5;
        nonlinear_model_real.P0 << 0.05, 0.00,
                              0.00, 0.20;

        nonlinear_model_filter = nonlinear_model_real;

        // Variables to be shown:
        std::vector<double> x1_error, x2_error, v2_predicted, v2_corrected, d2_predicted, d2_corrected;
        std::vector<double> x1_error_interval_sup, x1_error_interval_inf, x2_error_interval_sup, x2_error_interval_inf;
        std::vector<double> y1, y2;
        std::vector<double> ess; // particle filters only
        std::vector<double> elapsed_time_predict, elapsed_time_correct;

        // Inialization of unique pointers to desired filter:
        if (demo_name=="ekf"){
            pfilter = std::make_unique<BayesianFiltering::ExtendedKalmanFilter>();
            pfilter->Reset(nonlinear_model_filter);    
        }
        if (demo_name=="ukf"){
            pfilter = std::make_unique<BayesianFiltering::UnscentedKalmanFilter>();
            pfilter->Reset(nonlinear_model_filter);    
        }
        if (demo_name=="sirf"){
            pfilter = std::make_unique<BayesianFiltering::SamplingImportanceResamplingFilter>();
            long int Nparticles = 1000; 
            BayesianFiltering::SamplingImportanceResamplingFilter::EstimateType estimate_type = BayesianFiltering::SamplingImportanceResamplingFilter::EstimateType::ConditionalMean;
            if (argc>=3) Nparticles = std::stol(argv[2]);
            if (argc>=4){
                if (std::string(argv[3]) == "MAP"){
                    estimate_type = BayesianFiltering::SamplingImportanceResamplingFilter::EstimateType::MaximumAPosteriori;
                }
                else if (std::string(argv[3]) == "MEAN"){
                    estimate_type = BayesianFiltering::SamplingImportanceResamplingFilter::EstimateType::ConditionalMean;
                }
                else{
                    printf("\n\nInvalid value of estimate type for SIRF. Type \"demo\" to see options.");
                    return -1;
                }
            }
            dynamic_cast<BayesianFiltering::SamplingImportanceResamplingFilter*>(pfilter.get())->Reset(nonlinear_model_filter, Nparticles, estimate_type);
            printf("\nSIR Filter set with %li particles and %s\n",dynamic_cast<BayesianFiltering::SamplingImportanceResamplingFilter*>(pfilter.get())->Nparticles, \
                BayesianFiltering::SamplingImportanceResamplingFilter::EnumToString(dynamic_cast<BayesianFiltering::SamplingImportanceResamplingFilter*>(pfilter.get())->estimate_type));
        }
        
        // Main loop:
        BayesianFiltering::Result res;
        Eigen::VectorXd Xreal = nonlinear_model_real.X0;
        for (int k=1; k < 1000; ++k){
            double t = k * 0.2;

            // Real system:
            Eigen::VectorXd U = Eigen::VectorXd::Zero(Nu, 1);
            Eigen::VectorXd W(2,1);
            res = BayesianFiltering::MultivariateZeroMeanGaussianSampler(W, nonlinear_model_real.Q);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            res = nonlinear_model_real.ProcessModel(Xreal,U,t,W);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            Eigen::VectorXd Y(2,1);
            Eigen::VectorXd V(2,1);
            res = BayesianFiltering::MultivariateZeroMeanGaussianSampler(V, nonlinear_model_real.R);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            res = nonlinear_model_real.MeasurementModel(Y,Xreal,t,V);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }

            // Prediction:
            auto time_start_predict = std::chrono::steady_clock::now();
            res = pfilter->Prediction(U, t, nonlinear_model_filter);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            auto time_stop_predict = std::chrono::steady_clock::now();
            double e_predict = 0.001*(std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop_predict - time_start_predict)).count();

            // Correction:
            auto time_start_correct = std::chrono::steady_clock::now();
            res = pfilter->Correction(Y, t, nonlinear_model_filter);
            if (!res.first){
                printf("\nError at iteration k=%i: %s",k,res.second.c_str());
            }
            auto time_stop_correct = std::chrono::steady_clock::now();
            double e_correct = 0.001*(std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop_correct - time_start_correct)).count();

            // Storing variables in vectors to be shown:
            y1.push_back(Y(0,0));
            y2.push_back(Y(1,0));
            x1_error.push_back(Xreal(0,0)-pfilter->X(0,0));
            x2_error.push_back(Xreal(1,0)-pfilter->X(1,0));
            x1_error_interval_sup.push_back(+3.0*sqrt(pfilter->P(0,0)));
            x1_error_interval_inf.push_back(-3.0*sqrt(pfilter->P(0,0)));
            x2_error_interval_sup.push_back(+3.0*sqrt(pfilter->P(1,1)));
            x2_error_interval_inf.push_back(-3.0*sqrt(pfilter->P(1,1)));
            v2_predicted.push_back(pfilter->V_predicted(1,0));
            v2_corrected.push_back(pfilter->V_corrected(1,0));
            d2_predicted.push_back(pfilter->d2_predicted);
            d2_corrected.push_back(pfilter->d2_corrected);
            elapsed_time_predict.push_back(e_predict);
            elapsed_time_correct.push_back(e_correct);
            if (demo_name=="sirf"){
                 ess.push_back(dynamic_cast<BayesianFiltering::SamplingImportanceResamplingFilter*>(pfilter.get())->ESS);
            }
        }
        
        // Plots:
        auto f1 = matplot::figure(); f1->size(800, 600);
        auto f2 = matplot::figure(); f2->size(800, 600);
        auto f3 = matplot::figure(); f3->size(1000, 600);

        matplot::figure(f1);
        matplot::subplot(3, 1, 0); matplot::plot(y2, "b-"); matplot::title("y_2(k)"); matplot::ylabel("y_2");
        matplot::subplot(3, 1, 1); matplot::plot(x1_error, "b-"); matplot::hold(true); matplot::plot(x1_error_interval_sup, "r-"); matplot::plot(x1_error_interval_inf, "r-"); matplot::title("x_1 error (blue) and 3sigma interval (red)"); matplot::ylabel("x_1(k)-x_1(k|k)");
        matplot::subplot(3, 1, 2); matplot::plot(x2_error, "b-"); matplot::hold(true); matplot::plot(x2_error_interval_sup, "r-"); matplot::plot(x2_error_interval_inf, "r-"); matplot::title("x_2 error (blue) and 3sigma interval (red)"); matplot::xlabel("k"); matplot::ylabel("x_2(k)-x_2(k|k)");

        std::vector<double> x_chisquare = {0, 1000.0};
        std::vector<double> y_chisquare = {5.991, 5.991}; // 5%, 2 dof

        matplot::figure(f2);
        matplot::subplot(2, 2, 0); matplot::plot(v2_predicted, "b-"); matplot::title("innovation v_2(k|k-1)"); 
        matplot::subplot(2, 2, 1); matplot::semilogy(d2_predicted, "b-"); matplot::hold(true); matplot::semilogy(x_chisquare, y_chisquare, "r-"); matplot::title("d^2(k|k-1) (blue) and chi-square 95% limit (red)"); matplot::xlabel("k"); 
        matplot::subplot(2, 2, 2); matplot::plot(v2_corrected, "b-"); matplot::title("innovation v_2(k|k)"); 
        matplot::subplot(2, 2, 3); matplot::semilogy(d2_corrected, "b-"); matplot::hold(true); matplot::semilogy(x_chisquare, y_chisquare, "r-"); matplot::title("d^2(k|k) (blue) and chi-square 95% limit (red)"); matplot::xlabel("k"); 

        matplot::figure(f3);
        matplot::semilogy(elapsed_time_predict, "b-"); matplot::hold(true); matplot::semilogy(elapsed_time_correct, "r-"); matplot::title("Computing times to predict (blue) and correct (red)"); matplot::xlabel("k"); matplot::ylabel("[us]"); 

        if (demo_name=="sirf"){
            long int Nparticles = dynamic_cast<BayesianFiltering::SamplingImportanceResamplingFilter*>(pfilter.get())->Wparticles.rows();
            std::vector<double> x_nparticles = {0, 1000.0};
            std::vector<double> y_nparticles = {(double)(Nparticles), (double)(Nparticles)}; 
            auto f4 = matplot::figure(); f4->size(1000, 600);
            matplot::figure(f4);
            matplot::plot(ess, "b-"); matplot::hold(true); matplot::plot(x_nparticles, y_nparticles, "r-");  matplot::title("Effective Sample Size of particle filters (blue) and number of particles (red)"); matplot::xlabel("k"); matplot::ylabel("ESS"); 
        }

        matplot::show();        
        printf("\n\n");

        return 0;
    }

    // If reaches here, something gone wrong with arguments.
    printf("\nError: missing type of demo! Try the following:");
    printf("\n    \"kf\", for linear Kalman filter");
    printf("\n    \"akf\", for adaptive Kalman filter with default alpha (Q matrix learning) and beta (R matrix learning) parameters");
    printf("\n    \"akf alpha\", for adaptive Kalman filter with user alpha value (Q matrix learning) and default beta (R matrix learning) parameters");
    printf("\n    \"akf alpha beta\", for adaptive Kalman filter with user alpha value (Q matrix learning) and user beta value (R matrix learning) parameters");
    printf("\n    \"rsakf\", for robustified self adaptive linear Kalman filter");
    printf("\n    \"ekf\", for extended Kalman filter");
    printf("\n    \"ukf\", for unscented Kalman filter");
    printf("\n    \"sirf\", for sampling importance resampling particle with default number of particles and conditional mean estimate");
    printf("\n    \"sirf Nparticles\", for sampling importance resampling particle with Nparticles number of particles and conditional mean estimate");
    printf("\n    \"sirf Nparticles MEAN\", for sampling importance resampling particle with Nparticles number of particles and conditional mean estimate");
    printf("\n    \"sirf Nparticles MAP\", for sampling importance resampling particle with Nparticles number of particles and maximum a posteriori estimate");
    printf("\n\n");

    return -1;

}

