function v=fshift(u,dx,dy)
%------------------------------ FSHIFT ------------------------------
% Apply a periodic translation of vector (dx,dy) to an image
% a shift of (-dx,-dy) cancels a shift of (dx,dy)
%
% author: Lionel Moisan                                   
%
% v1.0 (10/2012): first version (LM)
% v1.1 (10/2012): removed bound constraints on |dx| and |dy| (LM)

  [ny,nx] = size(u);
  dx = mod(dx,nx);
  dy = mod(dy,ny);
  if dx>=0
    v = [u(:,end+1-dx:end),u(:,1:end-dx)];
  else
    v = [u(:,1-dx:end),u(:,1:-dx)];
  end
  if dy>=0
    v = [v(end+1-dy:end,:);v(1:end-dy,:)];
  else
    v = [v(1-dy:end,:);v(1:-dy,:)];
  end
