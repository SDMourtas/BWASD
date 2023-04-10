function Problem_figures(Y1_pred,Y1,Y2_pred,Y2,E,Em,EE)

figure
plot(1:length(EE),EE,'Color',[0.4660 0.6740 0.1880])
hold on
plot(1:length(E),E,'Color',[0.4940 0.1840 0.5560])
plot(find(E==Em),Em,'h','Color',[0.9290 0.6940 0.1250],...
    'MarkerSize',12)
xlabel('Iteration');ylabel('MAE %');xlim([1 length(E)])
legend('Fitting Set Error','Validation Set Error','Minimum Point')
hold off

figure
n=size(Y1_pred,2); cor=zeros(n,1); incor=cor;
for i=1:n
test_results=Y1_pred(:,i)==Y1; 
len=length(test_results); cor(i)=sum(test_results); incor(i)=len-cor(i);
end

b=[cor,incor];
Bar=bar(1:n,b);
ctr=zeros(2,n);ydt=ctr;
for i = 1:2
    ctr(i,:)=bsxfun(@plus, Bar(i).XData, Bar(i).XOffset'); 
    ydt(i,:)=Bar(i).YData;                                  
end
for i=1:n
    text(ctr(:,i),ydt(:,i),sprintfc('%d',ydt(:,i)),...
        'horiz','center', 'vert','bottom')
end
set(Bar(1),'FaceColor',[0.4940 0.1840 0.5560]) ;
set(Bar(2),'FaceColor',[0.9290 0.6940 0.1250]) ;
box on
ylabel('Training Set Samples')
xlabel('Loan Approval Classification Results')
xticklabels({'BWASD','MWASD','FKNN','FTR','LSVM','KNB'})
legend('Correct','Incorrect')

figure
n=size(Y1_pred,2); cor=zeros(n,1); incor=cor;
for i=1:n
test_results=Y2_pred(:,i)==Y2; 
len=length(test_results); cor(i)=sum(test_results); incor(i)=len-cor(i);
end

b=[cor,incor];
Bar=bar(1:n,b);
ctr=zeros(2,n);ydt=ctr;
for i = 1:2
    ctr(i,:)=bsxfun(@plus, Bar(i).XData, Bar(i).XOffset'); 
    ydt(i,:)=Bar(i).YData;                                  
end
for i=1:n
    text(ctr(:,i),ydt(:,i),sprintfc('%d',ydt(:,i)),...
        'horiz','center', 'vert','bottom')
end
set(Bar(1),'FaceColor',[0.4940 0.1840 0.5560]) ;
set(Bar(2),'FaceColor',[0.9290 0.6940 0.1250]) ;
box on
ylabel('Testing Set Samples')
xlabel('Loan Approval Classification Results')
xticklabels({'BWASD','MWASD','FKNN','FTR','LSVM','KNB'})
legend('Correct','Incorrect')
