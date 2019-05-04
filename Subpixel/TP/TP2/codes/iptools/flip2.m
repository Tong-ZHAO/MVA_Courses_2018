function flip2(u,v)
% flip2(u,v) : Flip between two images
% 
% author: Lionel Moisan
%
% v1.0 (11/2017): first version (LM)
  
  figure();
  while 1
    imshow(u);
    title(inputname(1));
    pause;
    imshow(v);
    title(inputname(2));
    pause;
  end
