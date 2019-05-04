function v = ffttrans(u,tx,ty)
% Non-integer signal/image translation using Fourier interpolation 
%
% author: Lionel Moisan                                   
%
% 1D translation in Fourier domain
% v(l,k) = U(k+tx,l+ty), where U is the Fourier interpolate of u
%
% also works when u is a (line or column) vector
%
% v1.0 (11/2017): first version (LM)

  [ny,nx] = size(u);
  u = double(u);
  if nargin==2; % 1D translation 
    if size(u,1)==1 
      ty = 0; 
    else 
      ty = tx; 
      tx = 0; 
    end
  end
  mx = floor(nx/2);
  my = floor(ny/2);
  Tx = exp(-2*i*pi*tx/nx*( mod(mx:mx+nx-1,nx)-mx ));
  Ty = exp(-2*i*pi*ty/ny*( mod(my:my+ny-1,ny)-my ));
  v = real(ifft2(fft2(u).*(Ty.'*Tx))); % we need non-conjugate transposition


