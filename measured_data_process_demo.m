clear
load('classify_net_TF2.mat');
load("net_2.mat");
load("30180ex.mat");load("30180ey.mat");load("30180hx.mat");load("30180hy.mat")

hx=hx-mean(hx);    % can be replaced by ey,hx,hy

fd=200;

signal = hx;
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
    fft_segments = fft(segment);
    fft_spectrum = abs(fft_segments)*4.4;
    fft_matrix(:, i) = fft_spectrum;
end

time_matrix=time_matrix';
fft_matrix =fft_matrix';
fft_matrix=real(fft_matrix);
time_and_freq_combined=[time_matrix fft_matrix]; % combine the time and frequency domain data

Noisy_Freq_data=time_and_freq_combined;

Clean=hx;

Noise=hx;

Y=zeros(1,100);
hx=[hx Y];
Dnoise=[];
Dnoise1=[];
Dnoise2=[];

Predicted_data=zeros(1,length(hx))*nan;

for i = 1:num_segments
      a = Noisy_Freq_data(i,:);
      test(:,1,1,1) = a(:);
      Pred = classify(net,test);
      pred = any(Pred=='2')|any(Pred=='3');
      if pred            % process noisy data
          X=hx(i*200+1:i*200+100);
          test1=hx((i-1)*200+1:i*200);
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
L=length(hx);M=max(hx);N=min(hx);
j=1;k=20000;
plot(Clean,'k'),hold on,plot(Noise,'b'),hold on,plot(preEX,'r'),title("(a)")
ylabel('Amplitude','FontName','Times New Roman');xlabel('Samples','FontName','Times New Roman');
legend('Recognized as clean','Recognized as noise','Predicted data','FontName','Times New Roman')