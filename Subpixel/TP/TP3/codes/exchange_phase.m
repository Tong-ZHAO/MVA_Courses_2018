function [ U, V ] = exchange_phase( u, v )
%change the module and the phase of two images.

  fu = fft2(u);
  fv = fft2(v);
  
  absu = abs(fu);
  absv = abs(fv);
  
  phaseu = angle(fu);
  phasev = angle(fv);
  
  fU = absv .* exp(1i * phaseu);
  fV = absu .* exp(1i * phasev);
  
  U = real(ifft2(fU));
  V = real(ifft2(fV));

end

