function v = fftrot(u,theta,x0,y0)
% v = fftrot(u,theta,x0,y0): Rotate an image with Fourier interpolation
% 
% author: Lionel Moisan                                   
%
% The output image v is defined (up to periodic conditions) by
% v(y,x) = U(Y,X)  where U is the Fourier interpolate of u and
% 
% X = x0 + cos(theta)*(x-x0) - sin(theta)*(y-y0)
% Y = y0 + sin(theta)*(x-x0) + cos(theta)*(y-y0)
% 
% by default (if x0 and y0 are not both specified) the rotation center is
% the image center: (x0,y0) = ( (nx+1)/2, (ny+1)/2 )
%
% The rotation angle (theta) is counted counterclockwise, in degrees
%
% Notes: 
% - the rotation is realized with 3 shears (fftshear function)
% - this function is intended for translation angles in [-45°,45°]
%   angles with larger magnitudes should be reduced to this interval
%   by using an appropriate pre-rotation with an angle multiple of 90° 
%
% v1.0 (11/2017): first version (LM)

  if nargin<4
    x0 = (size(u,2)+1)/2;
    y0 = (size(u,1)+1)/2;
  end
  t = tand(theta/2);
  s = -sind(theta);
  v = fftshear(fftshear(fftshear(u,'x',t,y0),'y',s,x0),'x',t,y0);

