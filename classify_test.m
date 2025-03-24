clear;
% Noise detection in Time & frequency domain
%% load dataset
% load('classify_net_TF2.mat');
load('classify_net_TF2.mat');
% load('Freq_classify_net.mat');
load('CNN_classify_net.mat');
load('time_and_freq_test_dataset.mat')
load('time_and_freq_test_labels.mat')

idx1 = 25600;
idx2 = 6400;
%% format the data

time_and_freq_test_dataset = double(reshape(time_and_freq_test_dataset(1:idx2,:)',400,1,1,[]));

time_and_freq_test_labels = double(time_and_freq_test_labels(1:idx2,:));

for i = 1:size(time_and_freq_test_labels,1)
    [~,Test_y(i,1)]=max(time_and_freq_test_labels(i,:));
end

numClasses = max(Train_y);  

Train_y =  categorical(Train_y);
Test_y  =  categorical(Test_y);

pred = classify(net, time_and_freq_test_dataset); 

accuracy=sum(Test_y==pred)/length(pred);   % ACC

label =double(Test_y);
p =double(pred);

confu = zeros(2,2);  % Confusion matrix
x_label=[]; y_label=[]; yy=[]; 
xx=1;

for i=1:size(p,1)  
        % if p(i)>=value
        %     box(i) = 1;
        % else
        %     box(i) = 2;
        % end

        if p(i,1)==2 && label(i,1)==2
            confu(2,2) = confu(2,2) + 1;     % TN
        elseif p(i,1)==2 && label(i,1)==1
            confu(1,2) = confu(1,2) + 1;     % FN
        elseif p(i,1)==1 && label(i,1)==2
            confu(2,1) = confu(2,1) + 1;     % FP
        elseif p(i,1)==1 && label(i,1)==1
            confu(1,1) = confu(1,1) + 1;     % TP
        end
end

    TN=confu(2,2); FN=confu(1,2); FP=confu(2,1); TP=confu(1,1);

    precision=TP/(TP+FP);
    recall=TP/(TP+FN);
    F=2*((precision*recall)/(precision+recall));

    FPR = FP/(FP+TN);
    TPR = TP/(TP+FN);