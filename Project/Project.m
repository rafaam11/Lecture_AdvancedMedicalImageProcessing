clear all;
close all;
clc;

%% 데이터 로드 및 변환전 PSNR 계산
image_NDCT=rgb2gray(double(imread('NDCT image.jpg'))/255);
image_LDCT=rgb2gray(double(imread('LDCT image.jpg'))/255);

% PSNR score 출력 ( 변환전LDCT 영상 vs NDCT 영상 )
psnr_LDCT_NDCT = psnr(image_LDCT, image_NDCT);


% NDCT image 플롯
figure(1);
subplot(3,4,1);
imshow(image_NDCT);
title({'NDCT image';'(Ground-truth)'},'fontsize',16);


% Original LDCT 플롯
subplot(3,4,2);
imshow(image_LDCT);
title({'Original LDCT';'(Noised image)';
    ['\color{red}PSNR score : ', num2str(psnr_LDCT_NDCT)]},'fontsize',16);



%% 논문 구현


%% Anisotropic filter만 사용 (1)
image_Aniso = image_LDCT;

% Anisotropic Diffusion Filtering 구현
image_Aniso = imdiffusefilt(image_Aniso);

% PSNR score 출력 ( Anisotropic-diffusion filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_A_NDCT = psnr(image_Aniso, image_NDCT);

% Anisotropic-diffusion filtering 후 LDCT 영상 플롯
subplot(3,4,5);
imshow(image_Aniso);
title({'(1) Anisotropic-Diffusion Filtering';
    ['\color{red}PSNR score : ', num2str(psnr_A_NDCT)]},'fontweight','bold', 'fontsize',16);


%% Bilateral filter만 사용 (2)
image_Bilat = image_LDCT;

% Bilateral Filtering 구현
image_Bilat = imbilatfilt(image_Bilat);

% PSNR score 출력 ( Bilateral filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_B_NDCT = psnr(image_Bilat, image_NDCT);

% Bilateral filtering후 LDCT 영상 플롯
subplot(3,4,6);
imshow(image_Bilat);
title({'(2) Bilateral Filtering';
    ['\color{red}PSNR score : ', num2str(psnr_B_NDCT)]},'fontweight','bold', 'fontsize',16);


%% Bilateral -> Anisotropic 순서로 사용 (3)
image_Bilat_Aniso = image_LDCT;

% Bilateral Filtering 구현
image_Bilat_Aniso = imbilatfilt(image_Bilat_Aniso);

% Anisotropic Diffusion Filtering 구현
image_Bilat_Aniso = imdiffusefilt(image_Bilat_Aniso);

% PSNR score 출력 ( Bilateral -> Anisotropic 후 LDCT 영상 vs NDCT 영상 )
psnr_BA_NDCT = psnr(image_Bilat_Aniso, image_NDCT);

% Bilateral -> Anisotropic 후 LDCT 영상 플롯
subplot(3,4,7);
imshow(image_Bilat_Aniso);
title({'(3) Bilateral -> Anisotropic-Diffusion Filtering';
    ['\color{red}PSNR score : ', num2str(psnr_BA_NDCT)]},'fontweight','bold', 'fontsize',16);


%% Anisotropic -> Bilateral 순서로 사용 (4)
image_Aniso_Bilat = image_LDCT;

% Anisotropic Diffusion Filtering 구현
image_Aniso_Bilat = imdiffusefilt(image_Aniso_Bilat);

% Bilateral Filtering 구현
image_Aniso_Bilat = imbilatfilt(image_Aniso_Bilat);

% PSNR score 출력 ( Anisotropic -> Bilateral 후 LDCT 영상 vs NDCT 영상 )
psnr_AB_NDCT = psnr(image_Aniso_Bilat, image_NDCT);

% Anisotropic -> Bilateral 후 LDCT 영상 플롯
subplot(3,4,8);
imshow(image_Aniso_Bilat);
title({'(4) Anisotropic-Diffusion -> Bilateral Filtering';
    ['\color{red}PSNR score : ', num2str(psnr_AB_NDCT)]},'fontweight','bold', 'fontsize',16);


%% Assignment 1


%% Box filtering (a)

% Box filtering
filterSize = 3;
image_a = imboxfilt(image_LDCT, filterSize);

% PSNR score 출력 ( Box filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_Box_NDCT = psnr(image_a, image_NDCT);

% Box filtering 후 영상 플롯
subplot(3,4,9);
imshow(image_a);
title({'(a) Box Filtered image of LDCT';
    ['\color{red}PSNR score : ', num2str(psnr_Box_NDCT)]},'fontsize',16);


%% Gaussian filtering (b)

% Gaussian filtering
sigma = 0.5;
image_b = imgaussfilt(image_LDCT, sigma);

% PSNR score 출력 ( Gaussian filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_Gaussian_NDCT = psnr(image_b, image_NDCT);

% Gaussian filtering 후 영상 플롯
subplot(3,4,10);
imshow(image_b);
title({'(b) Gaussian Filtered image of LDCT';
    ['\color{red}PSNR score : ', num2str(psnr_Gaussian_NDCT)]},'fontsize',16);



%% Sharpening filtering (c)

% Sharpening filtering
image_c = imsharpen(image_LDCT,'Radius',2,'Amount',1);

% PSNR score 출력 ( Sharpening filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_Sharpening_NDCT = psnr(image_c, image_NDCT);

% Sharpening filtering 후 영상 플롯
subplot(3,4,11);
imshow(image_c);
title({'(c) Sharpening Filtered image of LDCT';
    ['\color{red}PSNR score : ', num2str(psnr_Sharpening_NDCT)]},'fontsize',16);



%% Median filtering (d)

% Median filtering
kernel = [7 7];
image_d = medfilt2(image_LDCT,kernel);

% PSNR score 출력 ( Median filtering 후 LDCT 영상 vs NDCT 영상 )
psnr_Median_NDCT = psnr(image_d, image_NDCT);

% Median filtering 후 영상 플롯
subplot(3,4,12);
imshow(image_d);
title({'(d) Median Filtered image of LDCT';
    ['\color{red}PSNR score : ', num2str(psnr_Median_NDCT)]},'fontsize',16);


