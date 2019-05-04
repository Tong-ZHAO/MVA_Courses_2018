function cmview(u,delay,nrep)
% cmview(u,delay,nrep) : Play an image sequence u as a movie
% 
% author: Lionel Moisan
%
% delay: delay between frames (if delay<0, wait for key stroke)
% nrep: number of repetitions 
%
% v1.0 (11/2017): first version (LM)
  
  if nargin<3
    nrep = 1;
  end
  if nargin<2
    delay = 0;
  end
  figure();
  nt = size(u,3);
  for r=1:nrep
    for t=1:nt
      imshow(u(:,:,t));
      if delay<0
	pause;
      else
	pause(delay);
      end
    end
  end
