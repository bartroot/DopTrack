function [check] = insideHorizon(infile,station)
%
% This function will calculate when a satellite is inside horizon of the
% tracking station.
% 
% input:
%           - infile: location of the TLE output
%           - station: location of the station
%
% !!!!!!!!!!!Not finished yet!!!!!!!!!!!!!!
% NOTE!: check this code, because done by students
% 
% Change log:
%
%   - Bart Root, 17 MAy 2015: initial development
%
% Code usage:
%
%   - uses data file from get5daysPrediction
%   - needs readPrediction.m
%   - gstime.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constants used in script
%OMEGAE = 7.29211586D-5;  % Earth rotation rate in rad/s
Re = 6378136.00;         % Earth mean radius in meters

% start program

[output] = readPrediction(infile);

satrec.jdsatepoch = jday( output.init(8),output.init(9),output.init(10)...
                         ,output.init(11),output.init(12),output.init(13) );

% initialize prediction
tsince = output.prediction(:,1);     % [seconds] 
ro = output.prediction(:,2:4).*1000; % [meters]
%vo = output.prediction(:,5:7).*1000; % [m/s]
time = output.prediction(:,8:13);    % time
                     
% Transform orbit to Earth centered Earth fixed frame
% Compute Greenwich Apparent Siderial Time
gst=gstime(satrec.jdsatepoch+tsince/1440);
CGAST = cos(gst); 
SGAST = sin(gst);
% Transformation of the coordinates
xsat_ecf(:,1)= ro(:,1).*CGAST+ro(:,2).*SGAST;
xsat_ecf(:,2)=-ro(:,1).*SGAST+ro(:,2).*CGAST;
xsat_ecf(:,3)= ro(:,3);
%Apply rotation to convert velocity vector from ECI to ECEF coordinates
%vsat_ecf(1)= vo(1)*CGAST+vo(2)*SGAST+OMEGAE*xsat_ecf(2);
%vsat_ecf(2)=-vo(1)*SGAST+vo(2)*CGAST-OMEGAE*xsat_ecf(1);
%vsat_ecf(3)= vo(3);

% Transform orbit to Earth Latitude, longitude, and height
[LLA] = ecef2lla(xsat_ecf);

%% Check if satellite is inview of tracking station

% Define horizon
meanH = mean(LLA(:,3)); % mean height of the orbit
cosgamma = Re/(Re+meanH);  % cos(gamma) of the horizon

% Compute the cos(gamma) of every point in the orbit
vstation = lla2ecef([station.lat station.lon 0]);

satgamma = zeros(size(xsat_ecf(:,1)));
for i = 1:length(satgamma)
    satgamma(i) = dot(vstation,xsat_ecf(i,:))/sqrt(dot(xsat_ecf(i,:),xsat_ecf(i,:)))/sqrt(dot(vstation,vstation));
end

inside_horizon = zeros(size(satgamma));
inside_horizon(satgamma>cosgamma) = 1;

% calculate azimuth and elevation
[AZ, EL, dum] = geodetic2aer(LLA(:,1), LLA(:,2), LLA(:,3), station.lat.*ones(size(LLA(:,1))), station.lon.*ones(size(LLA(:,2))), station.h.*ones(size(LLA(:,3))), referenceEllipsoid('GRS80','meters'));

check = [inside_horizon time AZ EL];
