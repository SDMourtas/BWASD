%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  A 3-layer feed-forward neuronet model, trained by a BWASD        %
%  algorithm. (version 1.0)                                         %
%                                                                   %
%  Developed in MATLAB R2022a                                       %
%                                                                   %
%  Author and programmer: S.D.Mourtas, V.N.Katsikis,                %
%                         P.S. Stanimirovic, L.A. Kazakovtsev       %
%                                                                   %
%   e-Mail: spirmour@econ.uoa.gr, vaskatsikis@econ.uoa.gr           %
%           pecko@pmf.ni.ac.rs, levk@bk.ru                          %
%                                                                   %
%   Main paper: S.D.Mourtas, V.N.Katsikis, P.S. Stanimirovic,       %
%               L.A. Kazakovtsev, "Loan Approval Classification     %
%               Using a Bio-inspired Neural Network", (submitted)   %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
clc

% Choose modeling problem (for x = 1 to 4)
x=1;
[X_train,Y_train,X_test,Y_test,p,nmax]=problem(x);

%% Training
% Data Preprocessing
[X_N,X_min,X_max]=Normalization(X_train); % Normalization
[Y_N,~,~]=Normalization(Y_train);

% Neuronet model
tmax=20;d=5;delta=5;
tic
[W_BAS,Em,N_BAS,c_BAS,E,EE,pp]=BWASD(X_N,Y_N,nmax,tmax,p,d,delta);  % BWASD's optimal hidden-layer structure
toc
tic
[W,~,N,c]=MWASD(X_N,Y_N,p,nmax);  % MWASD's optimal hidden-layer structure
toc
tic
FKNN_Model=FKNN([X_train,Y_train]);        % Fine KNN model 
toc
tic
FineTree_Model=FTR([X_train,Y_train]);     % Tree: Fine Tree model 
toc
tic
LinearSVM_Model=LSVM([X_train,Y_train]);   % Linear SVM model 
toc
tic
KNB_Model=KNB([X_train,Y_train]);          % Kernen Naive Bayes model 
toc

%% Predict
% Prediction on train set
PTr_BAS=predictN(X_train,W_BAS,N_BAS,c_BAS,X_min,X_max);
PTr=predictN(X_train,W,N,c,X_min,X_max); 
PTr_KNN = FKNN_Model.predictFcn(X_train);
PTr_FT = FineTree_Model.predictFcn(X_train); 
PTr_SVM = LinearSVM_Model.predictFcn(X_train);
PTr_KNB = KNB_Model.predictFcn(X_train); 

% Prediction on test set
PTe_BAS=predictN(X_test,W_BAS,N_BAS,c_BAS,X_min,X_max); 
PTe=predictN(X_test,W,N,c,X_min,X_max);
PTe_KNN = FKNN_Model.predictFcn(X_test); 
PTe_FT = FineTree_Model.predictFcn(X_test); 
PTe_SVM = LinearSVM_Model.predictFcn(X_test);
PTe_KNB = KNB_Model.predictFcn(X_test);

% Error on test set
EPTe_BAS=error_pred(PTe_BAS,Y_test); 
EPTe=error_pred(PTe,Y_test); 
EPTe_KNN=error_pred(PTe_KNN,Y_test);
EPTe_FT=error_pred(PTe_FT,Y_test);
EPTe_SVM=error_pred(PTe_SVM,Y_test);
EPTe_KNB=error_pred(PTe_KNB,Y_test);

[h,pv] = McNemar_test([PTe_BAS,PTe,PTe_KNN,PTe_FT,PTe_SVM,PTe_KNB],Y_test);

%% Figures
Ytr_pr=[PTr_BAS,PTr,PTr_KNN,PTr_FT,PTr_SVM,PTr_KNB];
Yte_pr=[PTe_BAS,PTe,PTe_KNN,PTe_FT,PTe_SVM,PTe_KNB];

Problem_figures(Ytr_pr,Y_train,Yte_pr,Y_test,E,Em,EE)
