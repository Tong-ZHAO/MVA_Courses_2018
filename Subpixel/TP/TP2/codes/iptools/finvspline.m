function out = finvspline(c,order)
% 2D inverse B-spline transform
%
% author: Lionel Moisan                                   
%
% v1.0 (11/2017): initial version (LM)

  % initialize poles of associated z-filter 
  switch order
    case 2 
      z = [-0.17157288];  % sqrt(8)-3 
    case 3 
      z = [-0.26794919];  % sqrt(3)-2 
    case 4 
      z = [-0.361341,-0.0137254];
    case 5 
      z = [-0.430575,-0.0430963];
    case 6 
      z = [-0.488295,-0.0816793,-0.00141415];
    case 7 
      z = [-0.53528,-0.122555,-0.00914869];
    case 8 
      z = [-0.574687,-0.163035,-0.0236323,-0.000153821];
    case 9 
      z = [-0.607997,-0.201751,-0.0432226,-0.00212131];
    case 10 
      z = [-0.636551,-0.238183,-0.065727,-0.00752819,-0.0000169828];
    case 11 
      z = [-0.661266,-0.27218,-0.0897596,-0.0166696,-0.000510558];
    otherwise
      error('finvspline: order should be in 2..11.');
  end
  out = invspline1D(invspline1D(c,z)',z)';


% 1D inverse z-Transform (on all lines) with poles z
function c = invspline1D(c,z)
  npoles = length(z);
  N = size(c,2);
  c = c*prod((1-z).*(1-1 ./z)); % normalization 
  for k=1:npoles  % loop on poles
    % initialize causal filter, symmetric boundary conditions
    c(:,1) = c*[1,z(k).^[1:N-2]+z(k).^[2*N-3:-1:N],z(k)^(N-1)]'/(1-z(k)^(2*N-1));
    for n=2:N       % forward recursion 
      c(:,n) = c(:,n)+z(k)*c(:,n-1);
    end
    % initialize anticausal filter, symmetric boundary conditions
    c(:,N) = (z(k)/(z(k)^2-1))*(z(k)*c(:,N-1)+c(:,N));
    for n=N-1:-1:1  % backwards recursion 
      c(:,n) = z(k)*(c(:,n+1)-c(:,n));
    end
  end


