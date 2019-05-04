%% Exercice 7

u = double(imread('images/lena.pgm'));
f = fft2(u);
figure(1);
subplot(2, 2, 1);
imshow(f);
title('fft(u)');
subplot(2, 2, 2);
imshow(abs(f));
title('abs(fft(u))');
subplot(2, 2, 3);
imshow(abs(f),[]);
title('Normalized abs(fft(u))');
figure(2);
subplot(1, 2, 1);
imshow(normsat(abs(f),1));
title('Saturated abs(fft(u))')
subplot(1, 2, 2);
imshow(normsat(fftshift(abs(f)),1));
title('Shifted & Saturated abs(fft(u))')

figure(3);
subplot(2, 2, 1);
imshow(normsat(fftshift(real(f)), 1));
title('Real Part');
subplot(2, 2, 2);
imshow(normsat(fftshift(imag(f)), 1));
title('Imaginary Part');
subplot(2, 2, 3);
imshow(normsat(fftshift(angle(f)), 1));
title('Phase');
subplot(2, 2, 4);
imshow(normsat(fftshift(log(abs(f))), 1));
title('Module');

%%% Exercice 8

%% Q3

u = double(imread('lena.pgm'));
figure(2);
subplot(1, 2, 1);
imshow(u,[]);
title('u');
v = fshift(u,-30,-30);
subplot(1, 2 ,2);
imshow(v,[]);
title('v');

fu = fft2(u);
fv = fft2(v);
figure(3);
subplot(2, 2, 1);
imshow(normsat(fftshift(log(abs(fu))), 1));
title('log(module of u)');
subplot(2, 2, 2);
imshow(normsat(fftshift(angle(fu)), 1));
title('angle of u');
subplot(2, 2, 3);
imshow(normsat(fftshift(log(abs(fv))), 1));
title('log(module of v)');
subplot(2, 2, 4);
imshow(normsat(fftshift(angle(fv)), 1));
title('angle of v');

%%% Exercice 10

%% Q1
u = double(imread('images/lena.pgm'))/255;
v = double(imread('images/room.pgm'))/255;

[U, V] = exchange_phase(u, v);

figure(4);
subplot(2, 2, 1);
imshow(u, []);
title('u');
subplot(2, 2, 2);
imshow(v, []);
title('v');
subplot(2, 2, 3);
imshow(U);
title('U');
subplot(2, 2, 4);
imshow(V);
title('V');

%% Q2
u = imread('images/circle.jpeg');
if prod(size(size(u))) == 3
    u = double(rgb2gray(u)) / 255;
else
    u = double(u) / 255;
end
p_perde = perdecomp(u);
phasep = randphase(u);

figure(5);
subplot(1, 2, 1);
imshow(p_perde, []);
title('periodical u');
subplot(1, 2, 2);
imshow(phasep, []);
title('new U');

%% Q3

v = double(imread('images/texture.pgm')) / 255;

w = 500;
h = 300;
Iw = -w/2:-w/2+w-1;
Ih = -h/2:-h/2+h-1;
[X, Y] = meshgrid(Iw, Ih);
R = sqrt(1 * X.^2 + 3 * Y.^2);
u = (R<30);
randu = randphase(u);

figure(6);
subplot(1, 3, 1);
imshow(u, []);
title('u');
subplot(1, 3, 2);
imshow(v, []);
title('texture');
subplot(1, 3, 3);
imshow(randu, []);
title('new U');