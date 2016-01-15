function [px,py,pz,vx,vy,vz] = KeplerOrbit( major,eccen,incli,ranode,omega,t,t0,gm )

% [px,py,pz,vx,vy,vz] = KeplerOrbit( major,eccen,incli,ranode,omega,t,t0,gm )
%
% major   semi-major axis
% eccen   eccentricity
% incli   inclination
% ranode  right ascension of the ascending node
% omega   argument of perigee
% t       epoch
% t0      time of last perigee passage
% gm      gravitational constant 
% px      inertial x position
% py      inertial y position
% pz      inertial z position
% vx      inertial x velocity
% vy      inertial y velocity
% vz      inertial z velocity

nloop = 100;
[True,Eccen,Mean,Check] = KeplerEq(t,t0,gm,major,eccen,nloop);
[q1,q2,r,v,q3,q4] = OrbitPlane(major,eccen,gm,True);
[px,py,pz] = EarthCoord(q1,q2,incli,omega,ranode);
[vx,vy,vz] = EarthCoord(q3,q4,incli,omega,ranode);
