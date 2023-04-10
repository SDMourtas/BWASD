function [W,Em,Nbest,cbest,E,EE,pbest]=BWASD(X,Y,n,tmax,p,d,delta)
% function for finding the optimal hidden-layer neurons powers and weights
% of the neuronet

[G,M]=size(X);eta_delta=0.95;eta_d=0.95;
G1=round(p*G); % size of data fitting
G2=G-G1;       % size of data validation
x0=(1-round(n/2):n-round(n/2))';
E=zeros(tmax+1,1);EE=E;c0=randi(4,[n,1]);
[Em,EEm]=fitness(X,Y,G,M,x0,c0,G1,G2);E(1)=Em;EE(1)=EEm;x0=[p;c0;x0];

% iteration
for t=1:tmax 
    r=rands(2*n+1,1);
    r=r/(eps+norm(r));
    xr=round(x0+d*r); xr(xr(1)<0.3)=0.3; xr(xr(1)>0.95)=0.95;
    xl=round(x0-d*r); xl(xl(1)<0.3)=0.3; xl(xl(1)>0.95)=0.95;
    xr(xr(1:n+1)>4)=4; xr([0;xr(2:n+1)<1]==1)=1; %xr(end)=0;
    xl(xl(1:n+1)>4)=4; xl([0;xl(2:n+1)<1]==1)=1; %xl(end)=0;
    G1r=round(xr(1)*G); G2r=G-G1r;
    G1l=round(xl(1)*G); G2l=G-G1l;
    Er=fitness(X,Y,G,M,xr(n+2:end),xr(2:n+1),G1r,G2r);
    El=fitness(X,Y,G,M,xl(n+2:end),xl(2:n+1),G1l,G2l);
    x=round(x0+delta*r*sign(Er-El)); x(x(1)<0.3)=0.3; x(x(1)>0.95)=0.95;
    x(x(1:n+1)>4)=4; x([0;x(2:n+1)<1]==1)=1; %x(end)=0;
    G1x=round(x(1)*G); G2x=G-G1x;
    [Ev,Ef]=fitness(X,Y,G,M,x(n+2:end),x(2:n+1),G1x,G2x);
    E(t+1)=Ev; EE(t+1)=Ef;
    if E(t+1)<Em
        x0=x; Em=E(t+1);
    end
    delta=eta_delta*delta;
    d=eta_d*d+0.001;
end

% output
pbest=x0(1); cbest=x0(2:n+1); x0=x0(n+2:end); Nbest=x0(x0>=0); cbest=cbest(x0>=0);
K=Kmatrix(X,M,G,Nbest,cbest);
W=pinv(K)*Y; 
