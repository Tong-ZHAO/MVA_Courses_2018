function [ k_ftm ] = ftm_diffraction( r, c )
%Module de la transform?e de Fourier
%   k_ftm = c * (arccos(r) - r * sqrt(1 - r^2))

  k_ftm = c .* (acos(r) - r .* sqrt(1 - r.^2));
  
end

