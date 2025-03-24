clear;
% Noise detection in Time domain
%% load dataset
% load('classify_net_TF2.mat');
% load('classify_net.mat');
load('CNN_classify_net.mat');

load('classify_Test_dataset1.mat')
load('Test_labels2.mat')


idx1 = 25600;
idx2 = 6400;
%% format the data
classify_Test_dataset = double(reshape(classify_Test_dataset(1:idx2,:)',200,1,1,[]));

Test_labels2 = double(Test_labels2(1:idx2,:));

for i = 1:size(Test_labels2,1)
    [~,Test_y(i,1)]=max(Test_labels2(i,:));
end

Test_y  =  categorical(Test_y);

pred = classify(net, classify_Test_dataset); 

accuracy=sum(Test_y==pred)/length(pred);



label =double(Test_y);
p =double(pred);

confu = zeros(2,2);
x_label=[]; y_label=[]; yy=[];
xx=1;

for i=1:size(p,1)  
        % if p(i)>=value
        %     box(i) = 1;
        % else
        %     box(i) = 2;
        % end

        if p(i,1)==2 && label(i,1)==2
            confu(2,2) = confu(2,2) + 1;     %TN
        elseif p(i,1)==2 && label(i,1)==1
            confu(1,2) = confu(1,2) + 1;     %FN
        elseif p(i,1)==1 && label(i,1)==2
            confu(2,1) = confu(2,1) + 1;     %FP
        elseif p(i,1)==1 && label(i,1)==1
            confu(1,1) = confu(1,1) + 1;     %TP
        end
end

    TN=confu(2,2); FN=confu(1,2); FP=confu(2,1); TP=confu(1,1);

    precision=TP/(TP+FP);
    recall=TP/(TP+FN);
    F=2*((precision*recall)/(precision+recall));

    FPR = FP/(FP+TN);
    TPR = TP/(TP+FN);