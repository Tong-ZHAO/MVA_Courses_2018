function v = normsat(u,saturation)
%------------------------------ NORMSAT ------------------------------
% Normalize a matrix with an affine function 
%
% Without the second parameter (saturation), the affine function maps
% the minimum and maximum of the matrix onto 0 and 1
%
% Otherwise, <saturation> percent of the values are saturated (thresholded
% to 0 or 1)
%
% author: Lionel Moisan                                   
%
% v1.0 (10/2012): first version (LM)

  if nargin==2
    r = sort(u(:));
    n = length(r);
    p = floor(saturation*0.01*n);
    v = r(n-p:n)-r(1:p+1);
    [m,i] = min(v);
    m = r(i);
    d = r(n-p+i-1)-m;
  else
    m = min(u(:));
    d = max(u(:))-m;
  end
  if d==0.
    v = 0.5*ones(size(u));
  else
    v = (u-m)/d;
    v(v>1.) = 1.;
    v(v<0.) = 0.;
  end

