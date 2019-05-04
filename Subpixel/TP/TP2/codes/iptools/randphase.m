function v=randphase(u)
%------------------------------ RANDPHASE ------------------------------
% Randomize the phase of the Fourier Transform of an image
%
% note: to avoid special treatment of critical frequencies, we use
% the phases of a white Gaussian noise
%
% author: Lionel Moisan                                   
%
% v1.0 (10/2012): first version (LM)
% v1.1 (07/2013): added average preservation (LM)

  f = fft2(randn(size(u)));
  f(1,1) = 1; % preserve average value of input
  f(f==0) = 1;
  f = f./abs(f);
  v = real(ifft2(fft2(u).*f));

