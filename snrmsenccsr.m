function [snr,mse,ncc,sr]=snrmsenccsr(x1,x2)% x1 is original signal, x2 is reconstructed signal

%snr=20*log10(norm(x1)rm(x1-x2));  % SNR
 snr=10*log10((sum(x1.^2))/sum((x1-x2).^2));  % SNR
% mse=(sum(x1.^2)-sum(x2.^2))/length(x1);
mse=sum((x1-x2).^2)/length(x1);  % MSE
ncc=sum(x1.*x2)/sqrt((sum(x1.*x1)*(sum(x2.*x2)))); % NCC / CORC
y5=diff(x1);y6=diff(x2);
sr=sum(y5.*y5)/sum(y6.*y6);
    %a=sum(x2(1:2048).*b');
%b=sqrt(sum(x2(1:2048).^2)*sum(b'.^2));
%out=a/b;
% MSE=(sum(y1(1:2048).^2)-sum(signal_reconstruct'.^2))/sum(y1(1:2048).^2)
% 
% a=sum(y1(1:2048).*signal_reconstruct');
% b=sqrt(sum(y1(1:2048).^2)*sum(signal_reconstruct'.^2));
% ncc=a/b
% SNR=10*log10(sum(y1(1:2048).^2)/sum((y1(1:2048)-signal_reconstruct').^2))
