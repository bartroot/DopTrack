clear all;close all;clc;

% physical parameters
GM = 398600.4415;

% read in the data

C = load('TLE_Delfin3Xt_23.xyz');

X = [C(:,2) C(:,3) C(:,4)];
V = [C(:,5) C(:,6) C(:,7)];

% read in the vectors
H = zeros(length(X),3);
W = zeros(length(X),3);
RR = zeros(length(X),1);
h =zeros(length(X),1);
for ii = 1:length(X)
    RR(ii) = dot(X(ii,:)',V(ii,:)');
    H(ii,:) = cross(X(ii,:)',V(ii,:)');
    h(ii) = norm(H(ii,:));
    W(ii,:) = H(ii,:)./h(ii);
end

r = sqrt(X(:,1).^2 + X(:,2).^2 + X(:,3).^2);
v = sqrt(V(:,1).^2 + V(:,2).^2 + V(:,3).^2);

i = atan2(sqrt(W(:,1).^2+W(:,2).^2),W(:,3));
OO = atan2(W(:,1),-W(:,2));

p = h.^2./GM;

a = (2./r - v.^2./GM).^(-1);

n = sqrt(GM./a.^3);

e = sqrt(1-p./a);

E = atan2(RR./(a.^2.*n),1-r./a);

u = atan2(X(:,3),(-X(:,1).*W(:,2)+X(:,2).*W(:,1)));

theta = atan2(sqrt(1-e.^2).*sin(E),(cos(E)-e));

oo = u-theta;
oo(oo<0) = oo(oo<0) + 2*pi;

t = C(:,1)-C(1,1);

% plot the figures

figure
subplot(2,3,1)
plot(t,a,'Linewidth',2)
ylabel('Semi-major axis [km]','FontSize',20)
xlabel('Time [min]','FontSize',20)
set(gca,'FontSize',20)
grid on
subplot(2,3,2)
plot(t,e,'Linewidth',2)
xlabel('Time [min]','FontSize',20)
ylabel('Eccentricity [-]','FontSize',20)
set(gca,'FontSize',20)
grid on
subplot(2,3,3)
plot(t,rad2deg(i),'Linewidth',2)
xlabel('Time [min]','FontSize',20)
ylabel('Inclination [deg]','FontSize',20)
set(gca,'FontSize',20)
grid on
subplot(2,3,4)
plot(t,rad2deg(OO),'Linewidth',2)
xlabel('Time [min]','FontSize',20)
ylabel('RAAN [deg]','FontSize',20)
set(gca,'FontSize',20)
grid on
subplot(2,3,5)
plot(t,rad2deg(oo),'Linewidth',2)
xlabel('Time [min]','FontSize',20)
ylabel('Argument of perigee [deg]','FontSize',20)
set(gca,'FontSize',20)
grid on
subplot(2,3,6)
plot(t,rad2deg(theta),'Linewidth',2)
xlabel('Time [min]','FontSize',20)
ylabel('True Anomaly [deg]','FontSize',20)
set(gca,'FontSize',20)
grid on