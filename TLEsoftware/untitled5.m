% inspect density distribution profiles ion the leicester model

% Load crustal model 
disp('loading model for mass analyses')
d.h0 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.top.gmt'));
d.h1 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.topo.gmt'));
[d.h2,Lon,Lat] = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t1.gmt'));
d.h3 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t2.gmt'));
d.h4 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t3.gmt'));
d.h5 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t4.gmt'));
d.h6 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t5.gmt'));
d.h7 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t6.gmt'));
d.h8 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t7.gmt'));
d.h9 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t8.gmt'));
d.h10 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t9.gmt'));
d.h11 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t10.gmt'));
d.h12 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t11.gmt'));
d.h13 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t12.gmt'));
d.h14 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t13.gmt'));
d.h15 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t14.gmt'));
d.h16 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.t15.gmt'));


d.r1 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rhow.gmt'));
d.r2 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho1.gmt'));
d.r3 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho2.gmt'));
d.r4 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho3.gmt'));
d.r5 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho4.gmt'));
d.r6 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho5.gmt'));
d.r7 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho6.gmt'));
d.r8 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho7.gmt'));
d.r9 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho8.gmt'));
d.r10 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho9.gmt'));
d.r11 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho10.gmt'));
d.r12 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho11.gmt'));
d.r13 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho12.gmt'));
d.r14 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho13.gmt'));
d.r15 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho14.gmt'));
d.r16 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.rho15.gmt'));
d.r17 = gmt2matrix(load('/Users/bartroot/PhD/Data/Leicester/Lcrust10.r3330.gmt'));

% Load Lithospherice model

