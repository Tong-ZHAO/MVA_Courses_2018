function d = distcenter(u,opt)
% d = distcenter(u,opt) : Return an image coding the relative distance to the image center
%
% author: Lionel Moisan                                   
%
% The image size is the same as u
%
% If the syntax distcenter(u,'n') is used, the distance is normalized
% in such a way that the distance of the center to the nearest image border is 1
%
% v1.0 (11/2017): first version (LM)

  [ny,nx] = size(u);
  x = (1:nx)-(nx+1)/2;
  y = (1:ny)-(ny+1)/2;
  d = sqrt( (ones(ny,1)*x).^2 + (y'*ones(1,nx)).^2 );
  if nargin==2 && opt=='n' 
      d = 2*d/(1+min(nx,ny));
    end
  end

