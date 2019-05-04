%---------------- TP2 ---------------%

%%% Exercice 4 - Echantillonnage

%% Q1
u = double(imread('room.pgm'))/255;
imshow(u);

lambda = 3;
v = u(1:lambda:end,1:lambda:end);
w = kron(v,ones(lambda));
[ny,nx] = size(u);
figure(1);
imshow([u,w(1:ny,1:nx)]);

%% Q2
u = double(imread('room.pgm'))/255;
[ny,nx] = size(u);

llambda = [3, 4, 5, 6];
n = length(llambda);

figure(2);
for i = 1:n
    lambd = llambda(i);
    v = u(1:lambd:end,1:lambd:end);
    w = kron(v,ones(lambd));

    subplot(1, n, i);
    imshow(w(1:ny,1:nx));
    title(['lambda = ', num2str(lambd)]);
    hold on;
end

%% Q3

f = zeros(512);
f(190,50) = 2;
[w, h] = size(f);

a = -h/2;
b = h/2-1;

onde = real(ifft2(f));

figure(3);
subplot(1, 3, 1);
imshow(onde,[])
title('Onde');
subplot(1, 3, 2);
imshow(fftshift(abs(fft2(onde))),[],'xdata',a:b,'ydata',a:b)
title('FT before sampling')

lambda = 2;
v = onde(1:lambda:end,1:lambda:end);
[nw, nh] = size(v);
onde_fft = fft2(v);
na = -nh / 2;
nb = nh / 2 - 1;

subplot(1, 3, 3);
imshow(fftshift(abs(onde_fft)),[], 'xdata',na:nb,'ydata',na:nb);
title('FT after sampling')

%%% Exercice 5

%% Q1

f = zeros(512);
f(190,50) = 2;
onde = real(ifft2(f));
figure(4);
subplot(1, 2, 1);
imshow(onde,[])
title('onde');
subplot(1, 2, 2);
imshow(onde.^2,[]);
title('onde square');

%% Q2

onde_new = fftzoom(onde,2);
figure(5);
subplot(1, 3, 1);
imshow(onde,[]);
title('onde');
subplot(1, 3, 2);
imshow(onde.^2,[]);
title('onde square');
subplot(1, 3, 3);
imshow(onde_new.^2,[]);
title('onde sauqre (upsampling)')

%% Q3

u = double(imread('nimes.pgm'))/255;
figure(6);
subplot(1, 3, 1);
imshow(u, []);
title('Raw Image');
subplot(1, 3, 2);
%gradu = gradn(u);
gradu = gradn_bilateral(u);
imshow(gradu, []);
%title('Grad Image');
title('Bilateral Grad Image');
subplot(1, 3, 3);
imshow(gradu(260:300, 210:250), []);
title('Grad Image (Detail)');
