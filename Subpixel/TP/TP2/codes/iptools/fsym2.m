function v = fsym2(u)
%------------------------------ FSYM2 ------------------------------
% Symmetrization of an image along each coordinate
%
% author: Lionel Moisan                                   
%
% v1.0 (11/2017): first version (LM)

  v = [u,u(:,end:-1:1);u(end:-1:1,:),u(end:-1:1,end:-1:1)];
