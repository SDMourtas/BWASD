# BWASD
A Bio-inspired Neural Network for Loan Approval Classification
Implementation of a 3-layer feed-forward neural network model that is trained using a bio-inspired weights-and-structure-determination (BWASD) algorithm for loan approval classification.\
The purpose of this package is to determine whether or not to approve a loan through a BWASD neural network.\
The main articles used are the followings:
*	S.D. Mourtas, V.N. Katsikis, P.S. Stanimirovic and L.A. Kazakovtsev, "Loan Approval Classification Using a Bio-inspired Neural Network," (submitted)

Also, the kit includes the following four datasets:
*	Loan-Approval-Prediction.csv, taken from https://www.kaggle.com/datasets/ninzaami/loan-predication?resource=download
*	cs-training.csv, taken from https://www.kaggle.com/brycecf/give-me-some-credit-dataset?select=cs-training.csv
*	clean_dataset.csv, taken from https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
*	german.csv, taken from https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv

# M-files Description
*	Main_BWASD.m: the main function
*	problem.m: input data of the neural networks
*	data_prep.m: function for preprocessing data, changes strings with numbers
*	problem_figures.m: figures illustrating the problem's findings
*	BWASD.m: function for finding the optimal number of hidden-layer neurons, along with the optimal weights of the neural network
*	MWASD.m: a WASD algorithm for binary classification taken from https://doi.org/10.1016/j.asoc.2022.109351
*	Kmatrix.m: function for calculating the matrix K
*	Normalization.m: function for normalization
*	fitness.m: the objective function that must be minimized
*	predictN.m: function for WASD neural network model predictions
*	error_pred.m: function for calculating the models' statistics
*	FTR.mat: MATLAB's fine Tree model 
*	FKNN.mat: MATLAB's fine KNN model 
*	LSVM.mat: MATLAB's linear SVM model
*	KNB.mat: MATLAB's kernen naive Bayes model
*	McNemar_test.m: function for mid-p-value McNemar test

# Installation
*	Unzip the file you just downloaded and copy the BWASD directory to a location,e.g.,/my-directory/
*	Run Matlab/ Octave, Go to /my-directory/BWASD/ at the command prompt
*	run 'Main_BWASD (Matlab/Octave)

# Results
After running the 'Main_BWASD.m file, the package outputs are the following:
*	The optimal number of hidden-layer neurons.
*	The neural network models' statistics on the testing set of the datasets.
*	The graphic illustration of the classification performance.

# Environment
The BWASD kit has been tested in Matlab 2022a on OS: Windows 10 64-bit.
