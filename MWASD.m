function [W,Em,N,c,E,EE]=MWASD(X,Y,p,nmax)
% function for finding the optimal hidden-layer neurons weights of the
% neuronet

% Initialization
N=[]; c=[];% the neurons number of the hidden layer (i.e., hidneurons)
Em=inf; E=zeros(nmax+1,1); Ev=zeros(4,1); EE=zeros(nmax+1,1); [G,M]=size(X);
G1=round(p*G); % size of data fitting
G2=G-G1; % size of data validation

for d=0:nmax
% WDD Method
for i=1:4
K=Kmatrix(X,M,G,[N;d],c,i);
W=pinv(K(1:G1,:))*Y(1:G1); 
Ev(i)=100/G2*sum(abs(K(G1+1:G,:)*W-Y(G1+1:G))); % MAE
end
E(d+1)=min(Ev);
if E(d+1)<Em
    r=find(Ev==min(Ev));
    Em=E(d+1);N=[N;d];c=[c;r(1)];
end
K=Kmatrix(X,M,G,N,c);
W=pinv(K)*Y; 
EE(d+1)=100/G*sum(abs(K*W-Y)); % MAE;
end

% output
K=Kmatrix(X,M,G,N,c);
W=pinv(K)*Y; 