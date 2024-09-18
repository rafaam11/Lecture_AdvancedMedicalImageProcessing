%% BILATERAL FILTERING ON RGB color IMAGES

% Illustrates the use of bilateral filtering on RGB images.
% Bilateral filtering is a technique to smooth images while preserving edges, 
% by means of a nonlinear combination of nearby
% image values. The method is noniterative, local, and simple.
% The bilateral filter is controlled by two parameters: sigma s (spatial parameter) and sigma r (range parameter)

% Code Developed by :
% Gregory Kalliatakis (December 2014) - gkalliatakis@yahoo.gr
% Masters in Computer Vision
% University of Burgundy, France
% Advanced Image Analysis - Homework Project on Bilateral Filtering

%% Clear everything
clear all;
close all;
clc;
%%
image_NDCT=double(imread('NDCT image.jpg'));
image_ref = image_NDCT/255;
image=double(imread('LDCT image.jpg'));
image_LDCT_Ori = image/255;

figure(1);
imshow(image_LDCT_Ori);
title('Original LDCT Image');

figure(2);
imshow(image_ref);
title('Original NDCT Image');

% Implementation of Bilateral Filter
image=image/255;
Rchannel=image(:,:,1);
Gchannel=image(:,:,2);
Bchannel=image(:,:,3);
w=9; % window size
sigma_d=1;
sigma_r=30;
tic;
[Rout] = bilateral_each_channel(w,sigma_r,sigma_d,Rchannel);
[Gout] = bilateral_each_channel(w,sigma_r,sigma_d,Gchannel);
[Bout] = bilateral_each_channel(w,sigma_r,sigma_d,Bchannel);

% Implementation of Anisotropic Diffusion Filter
[Rout]= imdiffusefilt(Rout);
[Gout]= imdiffusefilt(Gout);
[Bout]= imdiffusefilt(Bout);

% Reconstruct image
image(:,:,1)=(Rout);
image(:,:,2)=(Gout);
image(:,:,3)=(Bout);
elapsed_time=toc;
fprintf('Total elapsed time is: %f seconds \n', elapsed_time); 

figure(3);
imshow(image);
title('Result of Bilateral Filtering','fontweight','bold','fontsize',14);

psnr1 = psnr(image, image_LDCT_Ori)
psnr2 = psnr(image_LDCT_Ori, image_ref)
        
