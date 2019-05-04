%%% Exercice 1
%%% Diffraction par une pupille circulaire

% %% Part 1
% 
% lr = linspace(0.001, 10, 1000);
% lk_df = kernel_diffraction(lr, 1);
% 
% figure(1);
% %plot(lr, lk_df);
% plot(lr, min(0.1, lk_df));
% title('Noyau de Diffraction');
% xlabel('r');
% ylabel('k\_diffraction');

% %% Part 2
% 
% [lx, ly] = meshgrid(linspace(-10, 10, 100), linspace(-10, 10, 100));
% lr = sqrt(lx .^ 2 + ly .^ 2);
% lk_df = kernel_diffraction(lr, 1);
% 
% figure(2);
% %imshow(lk_df,'InitialMagnification','fit');
% % R?sultat avec saturation
% imshow(lk_df,[0, .1],'InitialMagnification','fit');
% title('Tache d Airy');

% %% Part 3

% lr = linspace(0.001, 1, 100);
% lk_ftm = ftm_diffraction(lr, 1);
% 
% figure(3);
% plot(lr, lk_ftm);
% title('FTM - 1D');
% xlabel('r');
% ylabel('hat(K\_diffraction)');
% [lx, ly] = meshgrid(-1:0.005:1, -1:0.005:1);
% lr = sqrt(lx .^ 2 + ly .^ 2);
% lk_ftm = abs(ftm_diffraction(lr, 1));
% lk_ftm(lr == 0) = 0;
% 
% [lx_fft, ly_fft] = meshgrid(-100:1:100, -100:1:100);
% lr_fft = sqrt(lx_fft .^ 2 + ly_fft .^ 2);
% lk_df = kernel_diffraction(lr_fft, 1);
% lk_df(lr_fft == 0) = 0;
% lfft = fftshift(abs(fft2(lk_df)));
%
% figure(4);
% subplot(1, 2, 1);
% imshow(lk_ftm, [], 'InitialMagnification', 'fit');
% title('FTM');
% subplot(1, 2, 2);
% imshow(lfft, [], 'InitialMagnification', 'fit');
% title('FFT');

% %% Part 4
% 
% lr = linspace(-10, 10, 1000);
% lk_df = kernel_diffraction(lr, 1);
% 
% k = 195;
% dist = 20. / 1000 * k;
% 
% lk_sum = lk_df;
% lk_sum(k+1:end) = lk_sum(k+1:end) + lk_df(1:1000-k);
% 
% figure(5);
% a1 = plot(lr, lk_sum);
% hold on;
% a2 = plot(lr(k+1:end), lk_df(1:1000-k));
% hold on;
% a3 = plot(lr, lk_df);
% legend([a1; a2; a3], 't1+t2', 't2', 't1');
% title(sprintf('dist = %f', dist));

%% Part 6
% 
[lx, ly] = meshgrid(-1:1:100, -1:1:100);
lr = sqrt(lx .^ 2 + ly .^ 2);
lk_ftm = kernel_diffraction(lr, 1);
lk_ftm_o = occlusion_diffraction(lr, 0.25, 1);

lk_ftm(lr == 0) = 0;
lk_ftm_o(lr == 0) = 0;

figure(6);
subplot(1, 2, 1);
imshow(fftshift(abs(fft2(lk_ftm))), [], 'InitialMagnification', 'fit');
title('FFT - Tache Airy');
subplot(1, 2, 2);
imshow(fftshift(abs(fft2(lk_ftm_o))), [], 'InitialMagnification', 'fit');
title('FFT - Occlusion');






