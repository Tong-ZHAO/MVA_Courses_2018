function [ k_df ] = kernel_diffraction( r, c )
%calculate the kernel of the diffraction
%   k_df(r) = C * (2 J1(r) / r)^2

   if r == 0
    k_df = c;
   else
    k_df = c .* ((2 .* besselj(1, r) ./ r).^2);
   end

end

