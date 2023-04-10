function [Ev,Ef]=fitness(X,Y,G,M,N,c,G1,G2)
% fitness function

r=find(N<0); N(r)=[]; c(r)=[];
K=Kmatrix(X,M,G,N,c);
W=pinv(K(1:G1,:))*Y(1:G1); % WDD method
Ef=100/G1*sum(abs(K(1:G1,:)*W-Y(1:G1))); % MAE: fitting set
Ev=100/G2*sum(abs(K(G1+1:G,:)*W-Y(G1+1:G)));   % MAE: validation set