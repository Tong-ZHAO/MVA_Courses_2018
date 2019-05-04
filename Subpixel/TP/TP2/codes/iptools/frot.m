function v = frot(u,theta,order,x0,y0)
% v = frot(u,theta,order,x0,y0): Rotate an image 
% 
% author: Lionel Moisan                                   
%
% The output image v is defined by
% v(y,x) = U(Y,X)  where U is the specified interpolate of u and
% 
% X = x0 + cos(theta)*(x-x0) - sin(theta)*(y-y0)
% Y = y0 + sin(theta)*(x-x0) + cos(theta)*(y-y0)
% 
% by default (if x0 and y0 are not both specified) the rotation center is
% the image center: (x0,y0) = ( (nx+1)/2, (ny+1)/2 )
%
% The rotation angle (theta) is counted counterclockwise, in degrees
%
% available interpolations are:
%     order = 0           nearest neighbor (default)
%     order = 1           bilinear
%     order = -3          bicubic Keys
%     order = 3,5,7,9,11  spline with specified order
%
% v1.0 (11/2017): first version (LM)


  [ny,nx] = size(u);
  if nargin<5 
    x0 = (size(u,2)+1)/2;
    y0 = (size(u,1)+1)/2;
  end
  if nargin<3 
    order = 1; 
  end
  x = ones(ny,1)*(1:nx);
  y = (1:ny)'*ones(1,nx);
  X = x0 + cosd(theta)*(x-x0) - sind(theta)*(y-y0);
  Y = y0 + sind(theta)*(x-x0) + cosd(theta)*(y-y0);
  if order==0 % nearest neighbor interpolation
    iX = floor(X+0.5); iY = floor(Y+0.5);
    n1 = 0; n2 = 0;
  else % other interpolations :
    iX = floor(X); iY = floor(Y);
    X = X-iX; Y = Y-iY;
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
      error("Unrecognized interpolation order.");
    end   
    n1 = 1-n2;
  end
  % add (symmetrical) borders to avoid undefined coordinates
  iX = iX(:);
  iY = iY(:);
  supx1 = max(0,1-min(iX))-n1;
  supx2 = max(0,max(iX)-nx)+n2;
  u = [u(:,1+supx1:-1:2),u,u(:,nx-1:-1:nx-supx2)];
  iX = iX + supx1;
  supy1 = max(0,1-min(iY))-n1;
  supy2 = max(0,max(iY)-ny)+n2;
  u = [u(1+supy1:-1:2,:);u;u(ny-1:-1:ny-supy2,:)];
  iY = iY + supy1;
  if order==0 % nearest neighbor interpolation
    v = reshape(u(iY+(iX-1)*size(u,1)),ny,nx);
  else % other interpolations 
    % compute interpolation coefficients
    cx = c*( (ones(order+1,1)*(X(:)')).^([0:order]'*ones(1,nx*ny)) );
    cy = c*( (ones(order+1,1)*(Y(:)')).^([0:order]'*ones(1,nx*ny)) );
    v = zeros(1,nx*ny);
    for dx = n1:n2
      for dy = n1:n2
	v = v+cy(n2+1-dy,:).*cx(n2+1-dx,:).*u(iY+dy+(iX+dx-1)*size(u,1))';
      end
    end
    v = reshape(v,ny,nx);
  end



% coefficients of piecewise polynomial bicubic Keys function
function c = mat_coeff_keys(a)
  c = [0,0,a,-a;0,-a,2*a+3,-(a+2);1,0,-a-3,a+2;0,a,-2*a,a];


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
