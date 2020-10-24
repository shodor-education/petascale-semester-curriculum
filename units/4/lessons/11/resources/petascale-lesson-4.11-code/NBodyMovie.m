% Blue Waters Petascale Semester Curriculum v1.0
% Unit 4: OpenMP
% Lesson 11: N-Body Mechanics in OpenMP
% File: NBodyMovie.m
% Developed by Justin Oelgoetz for the Shodor Education Foundation, Inc.
% 
% Copyright (c) 2020 The Shodor Education Foundation, Inc.
% 
% Browse and search the full curriculum at
% <http://shodor.org/petascale/materials/semester-curriculum>.
% 
% We welcome your improvements! You can submit your proposed changes to this
% material and the rest of the curriculum in our GitHub repository at
% <https://github.com/shodor-education/petascale-semester-curriculum>.
% 
% We want to hear from you! Please let us know your experiences using this
% material by sending email to petascale@shodor.org
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

%A=importdata('out3.txt');
Npart=10000;
Ntime=100;
fx=fopen('xout.bin','r');
fy=fopen('yout.bin','r');
fz=fopen('zout.bin','r');
x=fread(fx,[Npart,Ntime],'double');
y=fread(fy,[Npart,Ntime],'double');
z=fread(fz,[Npart,Ntime],'double');
%x=reshape(A(:,1),[Npart,Ntime]);
%y=reshape(A(:,2),[Npart,Ntime]);
%z=reshape(A(:,3),[Npart,Ntime]);
%clear A

maxx=2*max(abs(x(:,1)));
maxy=2*max(abs(y(:,1)));
maxz=2*max(abs(z(:,1)));
clear F;
F(Ntime) = struct('cdata',[],'colormap',[]);

%now we plot the frames
for j = 1:Ntime
    hold off
    plot3(x(:,j),y(:,j),z(:,j),'o')
    axis([-maxx maxx -maxy maxy -maxz maxz])
    drawnow
    F(j) = getframe;
end
% Once this script is run, you can replay the
% movie interactively with the command
% movie(F)
