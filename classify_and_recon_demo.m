clear
load('classify_net_TF2.mat');
load("net_1.mat");
load('test_noisydata.mat');
load('test_cleandata.mat');
load("t1.mat");load("t2.mat");load("t3.mat");load("t4.mat")
load("s1.mat");load("s2.mat");load("s3.mat");load("s4.mat")
load("n1.mat");load("n2.mat");load("n3.mat");load("n4.mat")
% noisy_data=test_noisydata;
% EX=test_cleandata;
% noisy_data=t3;
noisy_data=s4+n4;EX=s4;    % can be replaced by s1/n1, ...
% noisy_data=noisy_data-mean(noisy_data);

fd=200;

signal = noisy_data;
N = length(signal);

segments_length=200;

%% Time and frequency domain data storage
num_segments = N / segments_length;

time_matrix=zeros(segments_length, num_segments);

for i = 1:num_segments
    time_segment = signal((i-1)*segments_length + 1 : i*segments_length);
    time_matrix(:, i) = time_segment;
end

fft_matrix = zeros(segments_length, num_segments);

for i = 1:num_segments
    segment = signal((i-1)*segments_length + 1 : i*segments_length);
    fft_segments = fft(segment); % FFT
    fft_spectrum = abs(fft_segments);
    fft_matrix(:, i) = fft_spectrum;
end

time_matrix=time_matrix';
fft_matrix =fft_matrix';
fft_matrix=real(fft_matrix);
time_and_freq_combined=[time_matrix fft_matrix]; % combine the time and frequency domain data

Noisy_Freq_data=time_and_freq_combined;

Clean=noisy_data;

Noise=noisy_data;

Y=zeros(1,100);
noisy_data=[noisy_data Y];
Dnoise=[];
Dnoise1=[];
Dnoise2=[];

Predicted_data=zeros(1,length(noisy_data))*nan;

for i = 1:num_segments
      a = Noisy_Freq_data(i,:);
      test(:,1,1,1) = a(:);
      Pred = classify(net,test);
      pred = any(Pred=='2')|any(Pred=='3');
      if pred            % process noisy data
          X=noisy_data(i*200+1:i*200+100);
          test1=noisy_data((i-1)*200+1:i*200);
          test1=[test1 X];
          for q = 1:length(test1)
              if mod(q,fd)==0
                  a1 = test1(q-fd+1:q);
                  b = test1(q-fd+101:q+100);

                  In1=reshape(a1,fd,1);
                  In1=reshape(In1,[fd,1,1,1]);
                  out1=predict(net_trained,In1);

                  In2=reshape(b,fd,1);
                  In2=reshape(In2,[fd,1,1,1]);
                  out2=predict(net_trained,In2);

                  DEX1=reshape(out1,fd,1);
                  DEX2=reshape(out2,fd,1);

                  Dnoise1=DEX1';
                  Dnoise2=DEX2';

                  average=Dnoise1(101:200)+Dnoise2(1:100);
                  Dnoise1=[Dnoise1(1:100) (average)/2];
              end
         end

         Predicted_data((i-1)*200+1:i*200)=Dnoise1;
         Dnoise=[Dnoise Dnoise1];
         Clean((i-1)*200+1:i*200) = NaN;
      else            % keep high-quality data
          Dnoise=[Dnoise Noise((i-1)*200+1:i*200)];
          Noise((i-1)*200+1:i*200) = NaN;
      end
end
preEX=Predicted_data;

%% Time domain waveform of reconstructed data
figure
L=length(noisy_data);M=max(noisy_data);N=min(noisy_data);
j=1;k=20000;
subplot(211),plot(Clean,'k'),hold on,plot(Noise,'b'),hold on,plot(preEX,'r'),title("(a)")
ylabel('Amplitude','FontName','Times New Roman');xlabel('Samples','FontName','Times New Roman');
legend('Recognized as clean','Recognized as noise','Predicted data','FontName','Times New Roman')

%% Spectrum of original data, noisy data, and reconstructed data
fs=150; N=5000;f=(0:N-1)*fs/N;
cs1=EX';
pucs1=abs(fft(cs1)/N);
cs2=noisy_data';
pucs2=abs(fft(cs2)/N);
cs3=Dnoise;
pucs3=abs(fft(cs3)/N);

subplot(212),plot(f(1:N/2),pucs1(1:N/2),'k','LineWidth',1),title("Spectrum comparison")
hold on,plot(f(1:N/2),pucs2(1:N/2),'b','LineWidth',1);
hold on,plot(f(1:N/2),pucs3(1:N/2),'r','LineWidth',1)
legend('Original EX data','Noisy data','Denoised data','FontName','Times New Roman')
ylabel('Amplitude','FontName','Times New Roman');xlabel('Frequency(Hz)','FontName','Times New Roman');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

[snr,mse,ncc,sr]=snrmsenccsr(EX,Dnoise);         % Compute SNR, MSE, CORC