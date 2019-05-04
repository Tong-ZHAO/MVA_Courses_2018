function v = fzoom(u,z,order)
% Zoom an image by a factor z 
%
% author: Lionel Moisan                                   
%
% z is not necessarily an integer (default value: 2)
%
% available interpolations are:
%     order = 0           nearest neighbor (default)
%     order = 1           bilinear
%     order = -3          bicubic Keys
%     order = 3,5,7,9,11  spline with specified order
%
% v1.0 (11/2017): first version (LM)
  
  if nargin<3 
    order = 0; 
  end
  if nargin<2 
    z = 2; 
  end
  [ny,nx] = size(u);
  sx = floor(nx*z);
  sy = floor(ny*z);
  if order==0
    % nearest neighbor is a special case (symmetric interpolation grid)
    X = 1+(nx-1-(sx-1)/z)/2+(0:sx-1)/z; iX = sx-floor(sx-X+0.5);
    Y = 1+(ny-1-(sy-1)/z)/2+(0:sy-1)/z; iY = sy-floor(sy-Y+0.5);
    v = u(iY,iX);
  else
    X = (0:sx-1)/z; iX = floor(X); X = X-iX; iX = iX+1;
    Y = (0:sy-1)/z; iY = floor(Y); Y = Y-iY; iY = iY+1;
    if order==-3 % bicubic Keys interpolation
      n2 = 2;
      c = mat_coeff_keys(-1/2);
      order = 3;
    elseif any([1,3,5,7,9,11]==order) % spline interpolation
      n2 = (order+1)/2;
      c = mat_coeff_splinen(order);
      if order>1 
	u = finvspline(u,order); 
      end
    else
      error('Unrecognized interpolation order.')
    end 
    n1 = 1-n2;
    % compute interpolation coefficients
    cx = c*( (ones(order+1,1)*X).^([0:order]'*ones(1,sx)) );
    cy = c*( (ones(order+1,1)*Y).^([0:order]'*ones(1,sy)) );
    % add (symmetrical) borders to avoid undefined coordinates
    u = [u(:,1-n1:-1:2),u,u(:,end-1:-1:end-n2)];
    u = [u(1-n1:-1:2,:);u;u(end-1:-1:end-n2,:)];
    v = zeros(sy,sx);
    for dx = n1:n2
      for dy = n1:n2
	v = v + (cy(n2+1-dy,:)'*cx(n2+1-dx,:)).*u(iY+dy-n1,iX+dx-n1);
      end
    end
  end


% coefficients of piecewise polynomial bicubic Keys function
function c = mat_coeff_keys(a)
  c = [0,0,a,-a;0,-a,2*a+3,-(a+2);1,0,-a-3,a+2;0,a,-2*a,a];
%endfunction

% coefficients of piecewise polynomial spline function with order n
function c = mat_coeff_splinen(n)
  c = zeros(n+1,n+1);
  a(1) = 1/prod(1:n);
  for k=1:n+1
    a(k+1) = -a(k)*(n+2-k)/k;
  end
  for k=0:n+1
    for p=0:n
      xp = prod((n-p+1:n)./(1:p))*k^(n-p);
      for i=k:n
	c(i+1,p+1) = c(i+1,p+1)+a(i-k+1)*xp;
      end
    end
  end
%endfunction

