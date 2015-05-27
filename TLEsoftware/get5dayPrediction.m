function [] = get5dayPrediction(infilename,outfilename,date)
%
% This script uses code from testing the SGP4 propagator.

% Author: 
%   Jeff Beck 
%   beckja@alumni.lehigh.edu 

% Version Info: 
%   1.0 (051019) - Initial version from Vallado C++ version. 
%   1.0 (aug 14, 2006) - update for paper
%   2.0 (apr 2, 2007) - update for manual operations
%   3.0 (3 jul, 2008) - update for opsmode operation afspc or improved
%   3.1 (2 dec, 2008) - fix tsince/1440.0 in jd update
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modified by Bart Root, 16 May 2015, TUDelft
%
% Should output 5 day prediction of chosen satellite: infile
%
% input:
%   - infile: string giving location of inpur TLE
%   - outfile: string giving location of output txt file
%   - data: starting date and time of the 5 day prediction.
%
% output:
%   - XYZ: matrix with time and location of the orbit
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Change log:
%
% - 16 May 2015: initial development - Bart Root
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Start of the program ---

% input checks

if ~ischar(infilename)
error('infilename must be a string')
end

if ~ischar(outfilename)
error('outfilename must be a string')
end

if ~ischar(date)
error('date must be a string, which tells time')
end

% these are set in sgp4init
global tumin mu radiusearthkm xke j2 j3 j4 j3oj2  

global opsmode

% // ------------------------  implementation   --------------------------

% add operation smode for afspc (a) or improved (i)
% opsmode= input('input opsmode afspc a, improved i ','s');
opsmode= 'i';

% //typerun = 'c' compare 1 year of full satcat data
% //typerun = 'v' verification run, requires modified elm file with
% //typerun = 'm' maunual operation- either mfe, epoch, or dayof yr
% //              start stop and delta times
% typerun = input('input type of run c, v, m: ','s');
typerun = 'm';
typeinput = 'e5';

% whichconst = input('input constants 721, 72, 84 ');
whichconst = 84;

% // ---------------- setup files for operation ------------------
% // input 2-line element set file

infile = fopen(infilename, 'r');
if (infile == -1)
    fprintf(1,'Failed to open file: %s\n', infilename);
    return;
end

outfile = fopen(outfilename, 'wt');

global idebug dbgfile

% // ----------------- test simple propagation -------------------
while (~feof(infile))
    longstr1 = fgets(infile, 130);
    while ( (longstr1(1) == '#') && (feof(infile) == 0) )
        longstr1 = fgets(infile, 130);
    end

    if (feof(infile) == 0)

        longstr2 = fgets(infile, 130);

%       // convert the char string to sgp4 elements
%       // includes initialization of sgp4
        [satrec, startmfe, stopmfe, deltamin] = twoline2rv( whichconst, ...
                   longstr1, longstr2, typerun, typeinput);

%       // For typeinput e5 do conversion.
        if strcmp(typeinput,'e5')
            startmfe = (juliandate(datenum(date)) - satrec.jdsatepoch) * 1440.0;
            stopmfe  = (juliandate(datenum(date))+5 - satrec.jdsatepoch) * 1440.0;
            deltamin = 15/60;
        end
               
        %fprintf(outfile, 'Satellite ID: %d xx\n', satrec.satnum);      

%       // call the propagator to get the initial state vector value
        [satrec, ro ,vo] = sgp4 (satrec,  0.0);

        jd = satrec.jdsatepoch + 0.0/1440.0;
                [year,mon,day,hr,minute,sec] = invjday ( jd );
        
        fprintf(outfile,...
                    ' %16.8f %16.8f %16.8f %16.8f %12.9f %12.9f %12.9f %5i%3i%3i %2i %2i %9.6f \n',...
                    0.0,ro(1),ro(2),ro(3),vo(1),vo(2),vo(3),year,mon,day,hr,minute,sec );
                    
        tsince = startmfe;

%       // check so the first value isn't written twice
        if ( abs(tsince) > 1.0e-8 )
            tsince = tsince - deltamin;
        end

%       // loop to perform the propagation
        while ((tsince < stopmfe) && (satrec.error == 0))

            tsince = tsince + deltamin;

            if(tsince > stopmfe)
                tsince = stopmfe;
            end

            [satrec, ro, vo] = sgp4 (satrec,  tsince);
            if (satrec.error > 0)
               fprintf(1,'# *** error: t:= %f *** code = %3i\n', tsince, satrec.error);
            end  
            
            if (satrec.error == 0)
                
                jd = satrec.jdsatepoch + tsince/1440.0;
                [year,mon,day,hr,minute,sec] = invjday ( jd );

                fprintf(outfile,...
                    ' %16.8f %16.8f %16.8f %16.8f %12.9f %12.9f %12.9f %5i%3i%3i %2i %2i %9.6f \n',...
                    tsince,ro(1),ro(2),ro(3),vo(1),vo(2),vo(3),year,mon,day,hr,minute,sec );
                       
            end %// if satrec.error == 0                        
            

        end %// while propagating the orbit

    end %// if not eof

end %// while through the input file

fclose(infile);
fclose(outfile);





