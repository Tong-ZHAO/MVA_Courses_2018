function v = fftzoom(u,z)
%------------------------------ FFTZOOM ------------------------------
% Zoom / Unzoom of an image with Fourier interpolation
% (zero-padding / frequency cutoff)
%
% author: Lionel Moisan                                   
%
% v1.0 (10/2017): first version (LM)

  if nargin==1 
    z = 2; 
  end
  [ny,nx] = size(u);
  mx = floor(z*nx);
  my = floor(z*ny);
  dx = floor(nx/2)-floor(mx/2);
  dy = floor(ny/2)-floor(my/2);
  if z>=1 
    %===== zoom in
    v = zeros([my,mx]);
    v(-dy+1:-dy+ny,-dx+1:-dx+nx) = fftshift(fft2(u));
  else 
    %===== zoom out
    f = fftshift(fft2(u));
    v = f(dy+1:dy+my,dx+1:dx+mx);
    if mod(mx,2)==0 
      v(:,1) = 0;  % cancel non-Shannon frequencies
    end
    if mod(my,2)==0 
      v(1,:) = 0;  % cancel non-Shannon frequencies
    end
  end
  v = z*z*real(ifft2(ifftshift(v)));
