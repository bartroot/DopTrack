clear all;close all;clc;

C = load('test_prediction.txt');
D = load('../../Software/TLE software/AIAA-2006-6753/python/Prediction_5day.txt');

t1 = datenum(C(1:end,8),C(1:end,9),C(1:end,10),C(1:end,11),C(1:end,12),C(1:end,13));
t2 = datenum(D(:,1),D(:,2),D(:,3),D(:,4),D(:,5),D(:,6));

r1 = sqrt(C(1:end,2).^2+C(1:end,3).^2+C(1:end,4).^2);
r2 = sqrt(D(:,7).^2+D(:,8).^2+D(:,9).^2);
%%

figure
plot(t1,r1)
hold on
plot(t2,r2,'r')
hold off

