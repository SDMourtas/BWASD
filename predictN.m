function Y=predictN(X,W,N,c,x_min,x_max)
% function for predicting

M=size(X,2);
x_N=Normalization(X,x_min,x_max);

Q=Kmatrix(x_N,M,size(X,1),N,c);
Y_N=Q*W; 

Y=zeros(size(Y_N));
Y(Y_N>=-0.375)=1;