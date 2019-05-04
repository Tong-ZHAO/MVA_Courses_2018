%%% Exercice 17

%% Interpolation de l'image

%u = double(imread('images/crop_bouc.pgm'));
u = double(imread('images/crop_cameraman.pgm'));

ln = [-3 0 1 3 5 7];

figure(1);

for i = 1:length(ln)
    v = fzoom(u, 16, ln(i));
    subplot(2, 3, i);
    imshow(v, []);
    title(strcat('n = ',int2str(ln(i))));
end 

%% FFT de l'image

v = double(imread('images/bouc.pgm'));
%v = double(imread('images/cameraman.pgm'));

fv = fft2(v);
figure(2);
subplot(1, 3, 1);
imshow(v, []);
title('original image');
subplot(1, 3, 2);
imshow(fftshift(log(abs(fv))), []);
title('abs(fft(v))');
subplot(1, 3, 3);
imshow(fftshift(angle(fv)), []);
title('phase(fft(v))');



