function v = contrast(u,s,t)
% v = contrast(u,s,t) : Apply a contrast change to an image
%
% author: Lionel Moisan                                   
%
% The output image v is defined by v=g(u)
% where g is the unique piecewise linear function defined by
% g(s(i)) = t(i) for all i, g(-infinity) = t(1), g(+infinity) = t(end)
%
% the vector s must be stricltly increasing
% examples:
% contrast(u,[min(u(:)),max(u(:))],[0,255]) is the contrast change applied by imshow(u,[])
% contrast(u,[128,128*(1+eps)],[0,1]) computes the level set {u>128}
% contrast(u,0:255,sin([0:255]/10))
%
% v1.0 (11/2017): first version (LM)

  [ny,nx] = size(u);
  n = length(s);
  if length(t)~=n 
    error("Size of s and t should match."); 
  end
  if min(u(:))<min(s) 
    s = [min(u(:))-1;s(:)]; t = [t(1);t(:)]; 
  end
  if max(u(:))>max(s) 
    s = [s(:);max(u(:))+1]; 
    t = [t(:);t(end)]; 
  end
  [n,e,I] = histcounts(u,s);
  p = (u-reshape(s(I),ny,nx))./(reshape(s(I+1)-s(I),ny,nx));
  v = (ones(size(p))-p).*reshape(t(I),ny,nx)+p.*reshape(t(I+1),ny,nx);

