% Main prediction satellite pass
clear all;
close all;
clc;

% obtain current TLE

% see in README getTLE

%% %%%%%%%%%%%%%%%%%%%%%%

TLE_loc = '~/PhD/Side_projects/DopTrack/TLE/getTLE/TLE.txt';
output_file = 'test_prediction.txt';
time = datestr(now-2/24);

station.lon = 4.36;%4.373378;   % Longitude
station.lat = 52.01;%51.999142;  % Latitude
station.h = 0;            % hieght [meters]

% get 5 day prediction
tic;
get5dayPrediction(TLE_loc,output_file,time);
toc

% check when the satellite is on the horizon
tic;
[check] = insideHorizon('test_prediction.txt',station);
toc

Time_hor = (jday(check(:,2),check(:,3),check(:,4),check(:,5),check(:,6),check(:,7)));
pass_hor = check(:,1);
AZ = check(:,8);
EL = check(:,9);

figure
plot(Time_hor,pass_hor)

%% Quantify the passes

pass = 0;
inview = 0;
scenario = 0;
k = 1;
start = 0;
for i = 1:length(pass_hor)
    % check scenario of current (i) epoch
    if i==1
        if pass_hor(1)==1
            pass = 1;
            inview = 1;
            start = 1; 

            % satellite is in view
            scenario = 1 - pass_hor(i);            
            
            % update logfile
            logfile(k,:) = [pass inview i Time_hor(i) AZ(i) EL(i)];
            k = k+1;
        else
            scenario = 0 - pass_hor(i);
        end
    else
        % check scenario
        scenario = pass_hor(i-1) - pass_hor(i); 
    end
    %scenario
    
    % scenario: out of view = 1, in view = -1, no change = 0
    if scenario==1
        % out of view: pass ends
        inview = 0 ;
        
        % Find elevation of TCA
        if start ==1
            start = 0;
        else
            start_passID = logfile(k-1,3);
        
            % search for maximum Elevation in pass
            max_ELvec = EL(start_passID:i);
            max_EL = max(max_ELvec);
            maxID = find(max_EL==max_ELvec) + start_passID -1;
            
            % update logfile
            logfile(k,:) = [pass 1 maxID Time_hor(maxID) AZ(maxID) max_EL];
            k = k+1;
            
        end
                      
        % update logfile
        logfile(k,:) = [pass inview i-1 Time_hor(i-1) AZ(i-1) EL(i-1)];
        k = k+1;
        
    elseif scenario == -1
        % in view: pass begins
        pass = pass + 1;
        inview = 1;
        
        % update logfile
        logfile(k,:) = [pass inview i Time_hor(i) AZ(i) EL(i)];
        k = k+1;
        
    elseif scenario == 0
        % no change
        
    else
        error(['This scenario is not documented! Scenario: ' num2str(scenario)])
    end
end   

%% log file to N2YO format

if ~mod(length(logfile),3)==0   
    logfile(1,:) = [];
end

fid = fopen('Prediction.txt','w');

Passes = length(logfile)/3;
Status = zeros(Passes,7);
for ll = 1:3:length(logfile)
    
    % time of passage
    [year1,mon1,day1,hr1,min1,sec1] = invjday (logfile(ll,4));
    [year2,mon2,day2,hr2,min2,sec2] = invjday (logfile(ll+1,4));
    [year3,mon3,day3,hr3,min3,sec3] = invjday (logfile(ll+2,4));
    
    % UTC to local 
    hr1 = hr1 + 2;
    hr2 = hr2 + 2;
    hr3 = hr3 + 2;
    
    if round(logfile(ll+1,6))<10
        % do nothing
    else
        % print pass prediction
        fprintf(fid,'%2.0d-%2.0d %2.0d:%2.0d %3.0d || ',day1,mon1,hr1,min1,round(logfile(ll,5)));
        fprintf(fid,'%2.0d:%2.0d %3.0d || %2.0d || ',hr2,min2,round(logfile(ll+1,5)),round(logfile(ll+1,6)));
        fprintf(fid,'%2.0d:%2.0d %3.0d\n',hr3,min3,round(logfile(ll+2,5)));
    end

end

fclose(fid);

    