%%% Exercise 11

%% Q2

u = double(imread('images/lena.pgm')) / 255.;
[p,s] = perdecomp(u);

% plot p and s
figure(1);
subplot(1, 3, 1);
imshow(u, []);
title('u');
subplot(1, 3, 2);
imshow(p, []);
title('p');
subplot(1, 3, 3);
imshow(s, []);
title('s');

% Verify p + s = u
figure(2);
subplot(1, 3, 1);
imshow(u);
title('u');
subplot(1, 3, 2);
imshow(p + s);
title('p+s');
subplot(1, 3, 3);
imshow(p+s-u);
title('diff');

figure(3);
subplot(1, 3, 1);
imshow(kron(ones(2,2),u),[ ]);
title('u');
subplot(1, 3, 2);
imshow(kron(ones(2,2),p),[ ]);
title('p');
subplot(1, 3, 3);
imshow(kron(ones(2,2),s),[ ]);
title('s');

fu = fft2(u);
fp = fft2(p);
fs = fft2(s);

figure(4);
subplot(2, 3, 1);
imshow(fftshift(log(abs(fu))),[]);
title('log(abs(u))');
subplot(2, 3, 4);
imshow(fftshift(angle(fu)),[]);
title('phase(u)');
subplot(2, 3, 2);
imshow(fftshift(log(abs(fp))),[]);
title('log(abs(p))');
subplot(2, 3, 5);
imshow(fftshift(angle(fp)),[]);
title('phase(p)');
subplot(2, 3, 3);
imshow(fftshift(log(abs(fs))),[]);
title('log(abs(s))');
subplot(2, 3, 6);
imshow(fftshift(angle(fs)),[]);
title('phase(s)');

% symmetrie de u
symu = fsym2(u);
fsymu = fft2(symu);

figure(5);
subplot(1, 3, 1);
imshow(symu,[]);
title('sym u');
subplot(1, 3, 2);
imshow(fftshift(log(abs(fsymu))),[]);
title('log(abs(sym u))');
subplot(1, 3, 3);
imshow(fftshift(angle(fsymu)),[]);
title('phase(qym u)');