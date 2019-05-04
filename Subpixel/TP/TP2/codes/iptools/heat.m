function u = heat(u,n,s)
% u = heat(u,n,s) : Image evolution according to Heat Equation  u_t = Delta u
% 
% author: Lionel Moisan                                   
%
% The output image v is obtained by iterating n times the operator T
% that is, v = T^n u where
% T u (l,k) = (1-4s) u(l,k) + s ( u(l+1,k)+u(l-1,k)+u(l,k+1)+u(l,k-1)
% 
% Neuman boundary conditions are used, e.g. u(0,k) := u(2,k)
%
% v1.0 (11/2017): initial version (LM)

  if nargin<3
    s = 0.25; 
  end
  if nargin<2
    n = 1; 
  end
  [ny,nx] = size(u);
  x = 1:nx; xp = [2:nx,nx-2]; xm = [2,1:nx-1];
  y = 1:ny; yp = [2:ny,ny-2]; ym = [2,1:ny-1];
  for i=1:n
    u(y,x) = (1-4*s)*u(y,x) + s * ( u(yp,x)+u(ym,x)+u(y,xp)+u(y,xm) );
  end

