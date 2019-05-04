u = double(imread('images/bouc.pgm'));
v = ffttrans(u,0.5,0.5);

figure(1);
subplot(1, 3, 1);
imshow(u, []);
title('u');
subplot(1, 3, 2);
imshow(v, []);
title('v');
subplot(1, 3, 3);
imshow(u - v, []);
title('u - v');

%% Solution

[p,s] = perdecomp(u);
pv = ffttrans(p,0.5,0.5);

figure(2);
subplot(1, 3, 1);
imshow(u, []);
title('p');
subplot(1, 3, 2);
imshow(v, []);
title('pv');
subplot(1, 3, 3);
imshow(u - v, []);
title('p - pv');