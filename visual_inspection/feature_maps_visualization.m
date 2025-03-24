clear
load('net_2.mat')
load('noisydata_sample1.mat')
net=net_trained;
%If want to test your own data, please use 200×1×1 input

%% Visual inspection of DPFF-Net
if canUseGPU
    X = gpuArray(A1');
end
fd=200;

Layer11 = "relu_4_1_2";
LCFeatures = activations(net,X,Layer11);
% Calculate the average activation value for each feature map
LCavg_activations = mean(mean(LCFeatures, 1), 2);
LCavg_activations = squeeze(LCavg_activations);
% Sort the feature maps by their average activation value and select the top 64
[LCsorted_activations, LCsorted_indices] = sort(LCavg_activations, 'descend');
LCtop_feature_maps = LCFeatures(:, :, LCsorted_indices(1:64));
LCtop_feature_maps=squeeze(LCtop_feature_maps);

%plot feature maps extracted by local Conv block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(LCtop_feature_maps(:, i));%13   
end
title("local conv extracted feature maps");

Layer1 = "multiplication_2";
LSE1block2Features = activations(net,X,Layer1);
% Calculate the average activation value for each feature map
LSE1avg_activations = mean(mean(LSE1block2Features, 1), 2);
LSE1avg_activations = squeeze(LSE1avg_activations);
% Sort the feature maps by their average activation value and select the top 64
[LSE1sorted_activations, LSE1sorted_indices] = sort(LSE1avg_activations, 'descend');
LSE1top_feature_maps = LSE1block2Features(:, :, LSE1sorted_indices(1:64));
LSE1top_feature_maps=squeeze(LSE1top_feature_maps);

%plot feature maps extracted by first local SE block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(LSE1top_feature_maps(:, i));%13   
end
title("local SE1 extracted feature maps");

Layer2 = "relu_5_1_2";
LDFeatures = activations(net,X,Layer2);
LDavg_activations = mean(mean(LDFeatures, 1), 2);
LDavg_activations = squeeze(LDavg_activations);
[LDsorted_activations, LDsorted_indices] = sort(LDavg_activations, 'descend');
LDtop_feature_maps = LDFeatures(:, :, LDsorted_indices(1:64));
LDtop_feature_maps=squeeze(LDtop_feature_maps);

%plot feature maps extracted by local DWS block

figure;
num_maps = 64;
for i = 1:num_maps
    % Create a subplot for the activation map in a 5x5 grid
    subplot(8, 8, i);
   plot(LDtop_feature_maps(:, i));
    
end
title( "local DWSconv extracted feature maps");

Layer3 = "multiplication_4_1";
LSE2Features = activations(net,X,Layer3);
LSE2avg_activations = mean(mean(LSE2Features, 1), 2);
LSE2avg_activations = squeeze(LSE2avg_activations);
[LSE2sorted_activations, LSE2sorted_indices] = sort(LSE2avg_activations, 'descend');
LSE2top_feature_maps = LSE2Features(:, :, LSE2sorted_indices(1:64));
LSE2top_feature_maps=squeeze(LSE2top_feature_maps);

%plot feature maps extracted by second local SE block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(LSE2top_feature_maps(:, i));
    
end
title( "local SE2 extracted feature maps");


Layer4 = "swish_3";
FFFeatures = activations(net,X,Layer4);
FFavg_activations = mean(mean(FFFeatures, 1), 2);
FFavg_activations = squeeze(FFavg_activations);
[FFsorted_activations, FFsorted_indices] = sort(FFavg_activations, 'descend');
FFtop_feature_maps = FFFeatures(:, :, FFsorted_indices(1:64));
FFtop_feature_maps=squeeze(FFtop_feature_maps);

%plot feature maps extracted by FF block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(FFtop_feature_maps(:, i));
    
end
title( "FF extracted feature maps");

Layer5 = "concat_4";
ReductionFeatures = activations(net,X,Layer5);
Reductionavg_activations = mean(mean(ReductionFeatures, 1), 2);
Reductionavg_activations = squeeze(Reductionavg_activations);
[Reductionsorted_activations, Reductionsorted_indices] = sort(Reductionavg_activations, 'descend');
Reductiontop_feature_maps = ReductionFeatures(:, :, Reductionsorted_indices(1:64));
Reductiontop_feature_maps=squeeze(Reductiontop_feature_maps);

%plot feature maps extracted by Reduction block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(Reductiontop_feature_maps(:, i));
    
end
title( "Reduction extracted feature maps");

Layer11 = "relu_4_1_2_1";
GCFeatures = activations(net,X,Layer11);
% Calculate the average activation value for each feature map
GCavg_activations = mean(mean(GCFeatures, 1), 2);
GCavg_activations = squeeze(GCavg_activations);
% Sort the feature maps by their average activation value and select the top 64
[GCsorted_activations, GCsorted_indices] = sort(GCavg_activations, 'descend');
GCtop_feature_maps = GCFeatures(:, :, GCsorted_indices(1:64));
GCtop_feature_maps=squeeze(GCtop_feature_maps);

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(GCtop_feature_maps(:, i));%13   
end
title("Global conv extracted feature maps");

Layer1 = "multiplication";
GSE1Features = activations(net,X,Layer1);
% Calculate the average activation value for each feature map
GSE1avg_activations = mean(mean(GSE1Features, 1), 2);
GSE1avg_activations = squeeze(GSE1avg_activations);
% Sort the feature maps by their average activation value and select the top 64
[GSE1sorted_activations, GSE1sorted_indices] = sort(GSE1avg_activations, 'descend');
GSE1top_feature_maps = GSE1Features(:, :, GSE1sorted_indices(1:64));
GSE1top_feature_maps=squeeze(GSE1top_feature_maps);
%plot feature maps extracted by first global SE block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(GSE1top_feature_maps(:, i));%13   
end
title("Global SE1 extracted feature maps");

Layer2 = "relu_5_1_1";
GDFeatures = activations(net,X,Layer2);
GDavg_activations = mean(mean(GDFeatures, 1), 2);
GDavg_activations = squeeze(GDavg_activations);
[GDsorted_activations, GDsorted_indices] = sort(GDavg_activations, 'descend');
GDtop_feature_maps = GDFeatures(:, :, GDsorted_indices(1:64));
GDtop_feature_maps=squeeze(GDtop_feature_maps);

%plot feature maps extracted by global DWS block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(GDtop_feature_maps(:, i));
    
end
title( "Global DWSconv extracted feature maps");

Layer3 = "multiplication_4";
GSE2Features = activations(net,X,Layer3);
GSE2avg_activations = mean(mean(GSE2Features, 1), 2);
GSE2avg_activations = squeeze(GSE2avg_activations);
[GSE2sorted_activations, GSE2sorted_indices] = sort(GSE2avg_activations, 'descend');
GSE2top_feature_maps = GSE2Features(:, :, GSE2sorted_indices(1:64));
GSE2top_feature_maps=squeeze(GSE2top_feature_maps);

%plot feature maps extracted by second global SE block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(GSE2top_feature_maps(:, i));
    
end
title( "Global SE2 extracted feature maps");


Layer4 = "concat_3";
IncepFeatures = activations(net,X,Layer4);
Inceptavg_activations = mean(mean(IncepFeatures, 1), 2);
Inceptavg_activations = squeeze(Inceptavg_activations);
[Inceptsorted_activations, Inceptsorted_indices] = sort(Inceptavg_activations, 'descend');
Incepttop_feature_maps = IncepFeatures(:, :, Inceptsorted_indices(1:64));
Incepttop_feature_maps=squeeze(Incepttop_feature_maps);

%plot feature maps extracted by Inception block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(Incepttop_feature_maps(:, i));
    
end
title( "Inception extracted feature maps");


Layer5 = "relu_conv1_1";
Conv1Features = activations(net,X,Layer5);
Conv1avg_activations = mean(mean(Conv1Features, 1), 2);
Conv1avg_activations = squeeze(Conv1avg_activations);
[Conv1sorted_activations, Conv1sorted_indices] = sort(Conv1avg_activations, 'descend');
Conv1top_feature_maps = Conv1Features(:, :, Conv1sorted_indices(1:64));
Conv1top_feature_maps=squeeze(Conv1top_feature_maps);

%plot feature maps extracted by first Conv block

figure;
num_maps = 64;
for i = 1:num_maps
    subplot(8, 8, i);
   plot(Conv1top_feature_maps(:, i));
    
end
title( "Conv1 extracted feature maps");

%% Reconstruction
MT=A1;Dnoise=[];
for i = 1:length(MT)
    if mod(i,fd)==0
        a = MT(i-fd+1:i);
        In=reshape(a,fd,1);
        In=reshape(In,[fd,1,1,1]);
        out=predict(net,In);
        DEX=reshape(out,fd,1);
        Dnoise=[Dnoise,DEX'];
    end
end
CONTOUR=A1;

%% Plot
figure
subplot(4, 4, 1)
plot(A1,Color='k'),title("Noisy MT data");text('Units', 'normalized', 'String', '(a)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 5)
plot(FFtop_feature_maps(:, 32),Color='k'),title("FF Block");text('Units', 'normalized', 'String', '(b)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 9)
plot(Reductiontop_feature_maps(:, 64),Color='k'),title("Reduction Block");text('Units', 'normalized', 'String', '(b)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 13)
plot(LCtop_feature_maps(:, 9),Color='k'),title("local conv");text('Units', 'normalized', 'String', '(b)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 2)
plot(LSE1top_feature_maps(:, 4),Color='k'),title("local SE1");text('Units', 'normalized', 'String', '(c)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 6)
plot(LDtop_feature_maps(:, 49),Color='k'),title("local DWSconv");text('Units', 'normalized', 'String', '(b)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 10)
plot(LSE2top_feature_maps(:, 17),Color='k'),title("local SE2");text('Units', 'normalized', 'String', '(e)', 'Position',[0.01 0.93 0], 'FontSize', 12);

subplot(4, 4, 14)
plot(CONTOUR,Color='k')
hold on
plot(Dnoise,Color='r');title("Output Noise");text('Units', 'normalized', 'String', '(f)', 'Position',[0.01 0.93 0], 'FontSize', 12);
legend('Network output','Reconstructed signal')

subplot(4, 4, 3)
plot(Conv1top_feature_maps(:, 29),Color='k'),title("conv 1");text('Units', 'normalized', 'String', '(g)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(4, 4, 7)
plot(GCtop_feature_maps(:, 39),Color='k'),title("GC");text('Units', 'normalized', 'String', '(h)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(4, 4, 11)
plot(GSE1top_feature_maps(:, 2),Color='k'),title("GSE1");text('Units', 'normalized', 'String', '(i)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(4, 4, 4)
plot(GDtop_feature_maps(:, 9),Color='k'),title("GD");text('Units', 'normalized', 'String', '(j)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(4, 4, 8)
plot(GSE2top_feature_maps(:, 2),Color='k'),title("GSE2");text('Units', 'normalized', 'String', '(k)', 'Position',[0.01 0.93 0], 'FontSize', 12);
subplot(4, 4, 12)
plot(Incepttop_feature_maps(:, 21),Color='k'),title("inception");text('Units', 'normalized', 'String', '(e)', 'Position',[0.01 0.93 0], 'FontSize', 12);


