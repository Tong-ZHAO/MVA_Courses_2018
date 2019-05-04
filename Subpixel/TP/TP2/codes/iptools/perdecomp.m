function [p,s] = perdecomp(u)
%------------------------------ PERDECOMP ------------------------------
%       Periodic plus Smooth Image Decomposition
%
%               author: Lionel Moisan
%
%   This program is freely available on the web page
%
%   http://www.mi.parisdescartes.fr/~moisan/p+s
%
%   I hope that you will find it useful.
%   If you use it for a publication, please mention 
%   this web page and the paper it makes reference to.
%   If you modify this program, please indicate in the
%   code that you did so and leave this message.
%   You can also report bugs or suggestions to 
%   lionel.moisan [AT] parisdescartes.fr
%
% This function computes the periodic (p) and smooth (s) components
% of an image (2D array) u
%
% usage:    p = perdecomp(u)    or    [p,s] = perdecomp(u)
%
% note: this function also works for 1D signals (line or column vectors)
%
% v1.0 (01/2014): initial Matlab version from perdecomp.sci v1.2

  [ny,nx] = size(u); 
  u = double(u);
  X = 1:nx; Y = 1:ny;
  v = zeros(ny,nx);
  v(1,X)  = u(1,X)-u(ny,X);
  v(ny,X) = -v(1,X);
  v(Y,1 ) = v(Y,1 )+u(Y,1)-u(Y,nx);
  v(Y,nx) = v(Y,nx)-u(Y,1)+u(Y,nx);
  fx = repmat(cos(2.*pi*(X -1)/nx),ny,1);
  fy = repmat(cos(2.*pi*(Y'-1)/ny),1,nx);
  fx(1,1)=0.;   % avoid division by 0 in the line below
  s = real(ifft2(fft2(v)*0.5./(2.-fx-fy)));
  p = u-s;
  
