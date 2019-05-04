function [ occ_df ] = occlusion_diffraction( r, delta, c )
%Calculer le noyau de diffraction sur une pupille de t?l?scope
%   Detailed explanation goes here

  term_r = 2 .* besselj(1, r) ./ r;
  term_delta = (2 * delta) .* besselj(1, delta .* r) ./ r;
  occ_df = c .* (term_r - term_delta) .^ 2;

end

