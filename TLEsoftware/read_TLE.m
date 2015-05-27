clear all;
close all;
clc;

% read out xyz_issh.txt

fid = fopen('TLE_Delfin3Xt_23.xyz');

c = textscan(fid,'%s');
xx = c{1,1};
k = 1;

test_end = 0;
s =0;
for i = 1:1001
    ax = test_end;
    
    l = mod(i,13);

    if l==0
        l = 13;        
    end
    
    if (test_end==1&&l==1)
        test_end=0;
    end
    bx = l;
    cx = test_end;
    
    
    if test_end==0
    num(i,:)= [ax bx cx k i l xx(i)] ;   

        switch l 
            case 1
                time(k) = str2num(cell2mat(xx(i-s)));
            case 2
                px(k) = str2num(cell2mat(xx(i-s)));
            case 3
                py(k) = str2num(cell2mat(xx(i-s)));
            case 4
                pz(k) = str2num(cell2mat(xx(i-s)));
            case 5
                vx(k) = str2num(cell2mat(xx(i-s)));
            case 6
                vy(k) = str2num(cell2mat(xx(i-s)));
            case 7
                vz(k) = str2num(cell2mat(xx(i-s)));
            case 8
                year(k) = ((xx(i-s)));
            case 9
                month(k) = ((xx(i-s)));
            case 10
                day(k) = ((xx(i-s)));
            case 11
                hour(k) = ((xx(i-s)));
            case 12
                minute(k) = ((xx(i-s)));
                test = cell2mat(xx(i-s));
                test_end = ~strcmp(test(end),':');
                
                if test_end==1
                    k=k+1;
                    s=s+1;
                end
            case 13
                sec(k) = ((xx(i-s)));
                k = k+1;
            otherwise
                disp('Error!!')
        end
        
    end
end
