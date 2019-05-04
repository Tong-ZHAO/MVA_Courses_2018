function v = fftshear(u,d,a,b)
% v = fftshear(u,d,a,b) : Apply an horizontal or vertical shear to an image with Fourier interpolation
%
% author: Lionel Moisan                                   
%
% The input variable d specifies the coordinate along which the variable
% translation is applied (d = 'x' or 'y')
%
% If d is 'x', the output image v is defined by
% v(y,x) = U(y,x+a(y-b))    x=1..nx, y=1..ny
% where U is the Fourier interpolate of u
%
% v1.0 (10/2012): first version (LM)

  [ny,nx] = size(u);
  v = u;
  if d=='x'
    for y=1:ny
      v(y,:) = ffttrans(u(y,:),a*(y-b));
    end
  elseif d=='y'
    for x=1:nx
      v(:,x) = ffttrans(u(:,x)',a*(x-b))';
    end
  else
    error("Unrecognized shear direction.");
  end


