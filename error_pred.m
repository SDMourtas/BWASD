function E=error_pred(X,Y)

E{9,2}=[];
R=X-Y; T=length(Y);
E{1,2}=sum(abs(R))/T; % MAE
E{1,1}='MAE';

Y2=find(Y==1); 
R2=sum(X(Y2)==1)/length(Y2); % TP: true positive
E{2,2}=R2; % MAE
E{2,1}='TP';

R3=sum(X(Y2)==0)/length(Y2); % FP: false positive
E{3,2}=R3;
E{3,1}='FP';

Y3=find(Y==0); 
E{4,2}=sum(X(Y3)==0)/length(Y3); % TN: true negative
E{4,1}='TN';

R5=sum(X(Y3)==1)/length(Y3); % FN: false negative
E{5,2}=R5;
E{5,1}='FN';

prec=R2/(R2+R3);
E{6,2}=prec;
E{6,1}='Precision';

rec=R2/(R2+R5);
E{7,2}=rec;
E{7,1}='Recal';

E{8,2}=sum(X==Y)/T; % accuracy
E{8,1}='Accuracy';

E{9,2}=2*prec*rec/(prec+rec); % F-score
E{9,1}='F-score';