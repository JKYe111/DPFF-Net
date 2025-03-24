clear
load('test_noisydata.mat');
load('test_cleandata.mat');
EX=test_cleandata;
noisy_data=test_noisydata;
load('net_1.mat');
fd=200;

Clean=noisy_data;

Y=zeros(1,300);
noisy_data=[noisy_data Y];

Dnoise=[];
Dnoise1=[];
Dnoise2=[];

Predicted_data=zeros(1,5000)*nan;

window_length=100;
tic
for i = 1:5000
  if mod(i,200)==0 
      a = noisy_data(i-199:i);
      X=noisy_data(i+1:i+window_length);
      test1=a;
      test1=[test1 X];
          for q = 1:length(test1)
              if mod(q,fd)==0
                  a1 = test1(q-fd+1:q);
                  b = test1(q-fd+window_length+1:q+window_length);

                  In1=reshape(a1,fd,1);
                  In1=reshape(In1,[fd,1,1,1]);
                  out1=predict(net_trained,In1);

                  In2=reshape(b,fd,1);
                  In2=reshape(In2,[fd,1,1,1]);
                  out2=predict(net_trained,In2);

                  out1=reshape(out1,fd,1);
                  out2=reshape(out2,fd,1);

                  Dnoise1=out1';
                  Dnoise2=out2';

                  average=Dnoise1(window_length+1:200)+Dnoise2(1:200-window_length);
                  Dnoise1=[Dnoise1(1:window_length) (average)/2];
              end
          end
     Predicted_data(i-199:i)=Dnoise1;
     Dnoise=[Dnoise Dnoise1];
  end
end
toc
preEX=Predicted_data(1:5000);
noisy_data=noisy_data(1:5000);
%% Time domain waveform of reconstructed data
figure(1)
L=length(noisy_data);M=max(noisy_data);N=min(noisy_data);
j=1;k=2000;
subplot(311),plot(Clean,'k'),hold on,plot(preEX,'r'),title("(a)")
ylabel('Amplitude','FontName','Times New Roman');xlabel('Samples','FontName','Times New Roman');
legend('Recognized as clean','Recognized as noise','Predicted data','FontName','Times New Roman')

subplot(312),plot(noisy_data(j:k),'k'),hold on,plot(preEX(j:k)','r'),title("A segment of data in (a)")
legend('Recognized as clean','Recognized as noise','Predicted data','FontName','Times New Roman'),ylabel('Amplitude','FontName','Times New Roman');xlabel('Samples','FontName','Times New Roman');

%% Spectrum of original data, noisy data, and reconstructed data
fs=150; N=5000;f=(0:N-1)*fs/N;
cs1=EX';
pucs1=abs(fft(cs1,N))/N*2;
cs2=noisy_data';
pucs2=abs(fft(cs2,N))/N*2;
cs3=Dnoise;
pucs3=abs(fft(cs3,N))/N*2;

subplot(313),plot(f(1:N/2),pucs1(1:N/2),'k','LineWidth',1),title("Spectrum comparison")
hold on,plot(f(1:N/2),pucs2(1:N/2),'b','LineWidth',1);
hold on,plot(f(1:N/2),pucs3(1:N/2),'r','LineWidth',1)
legend('Original EX data','Noisy data','Denoised data','FontName','Times New Roman')
ylabel('Amplitude','FontName','Times New Roman');xlabel('Frequency(Hz)','FontName','Times New Roman');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

[snr,mse,ncc,sr]=snrmsenccsr(EX,Dnoise);
