load('mandrill.mat');
im=im2uint8(ind2gray(X,map));

figure
imshow(im, []);

figure
im_brighter = adjust_brightness_lineer(im, 25);
subplot(2,3,1), imshow(im_brighter, []);
stitle=sprintf('brigther with x = 25');
title(stitle);
im_brighter = adjust_brightness_lineer(im, 50);
subplot(2,3,2), imshow(im_brighter, []);
stitle=sprintf('brigther with x = 50');
title(stitle);
im_brighter = adjust_brightness_lineer(im, 100);
subplot(2,3,3), imshow(im_brighter, []);
stitle=sprintf('brigther with x = 100');
title(stitle);

im_darker = adjust_brightness_lineer(im, -25);
subplot(2,3,4), imshow(im_darker, []);
stitle=sprintf('darker with x = -25');
title(stitle);
im_darker = adjust_brightness_lineer(im, -50);
subplot(2,3,5), imshow(im_darker, []);
stitle=sprintf('darker with x = -50');
title(stitle);
im_darker = adjust_brightness_lineer(im, -100);
subplot(2,3,6), imshow(im_darker, []);
stitle=sprintf('darker with x = -100');
title(stitle);

figure
im_negative = image_negative(im, 255);
imshow(im_negative, []);
stitle=sprintf('image negative');
title(stitle);

figure
im_norm = double(im)/255;
im_log = logarithmic_transformation(im_norm, 2);
imshow(im_log, []);
stitle=sprintf('logaritmic transformation with c = 2');
title(stitle);


figure
im_gamma = gamma_correction(im, 1, 0.1);
subplot(2,3,1), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 0.1');
title(stitle);
im_gamma = gamma_correction(im, 1, 0.25);
subplot(2,3,2), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 0.25');
title(stitle);
im_gamma = gamma_correction(im, 1, 0.5);
subplot(2,3,3), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 0.5');
title(stitle);
im_gamma = gamma_correction(im, 1, 1.5);
subplot(2,3,4), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 2');
title(stitle);
im_gamma = gamma_correction(im, 1, 3);
subplot(2,3,5), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 3');
title(stitle);
im_gamma = gamma_correction(im, 1, 4);
subplot(2,3,6), imshow(im_gamma, []);
stitle=sprintf('gamma correction with c = 1 gamma = 4');
title(stitle);


figure
im_cont = adjust_contrast(im, 2, 25);
subplot(3,3,1), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 2 s = 25');
title(stitle);
im_cont = adjust_contrast(im, 5, 25);
subplot(3,3,2), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 5 s = 25');
title(stitle);
im_cont = adjust_contrast(im, 10, 25);
subplot(3,3,3), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 10 s = 25');
title(stitle);


im_cont = adjust_contrast(im, 2, 25);
subplot(3,3,4), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 2 s = 25');
title(stitle);
im_cont = adjust_contrast(im, 2, 50);
subplot(3,3,5), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 2 s = 50');
title(stitle);
im_cont = adjust_contrast(im, 2, 100);
subplot(3,3,6), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 2 s = 100');
title(stitle);


im_cont = adjust_contrast(im, 0.5, 25);
subplot(3,3,7), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 0.5 s = 25');
title(stitle);
im_cont = adjust_contrast(im, 0.5, 50);
subplot(3,3,8), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 0.5 s = 50');
title(stitle);
im_cont = adjust_contrast(im, 0.5, 100);
subplot(3,3,9), imshow(im_cont, []);
stitle=sprintf('contrast adjusting with a = 0.5 s = 100');
title(stitle);

figure
im_cs = contrast_strecthing(im, 0, 255);
imshow(im_cs, []);
stitle=sprintf('contrast stretching with mj = 0 Mj = 255');
title(stitle);

figure
im_cont = contrast_thresholding(im, 127);
subplot(1,3,1), imshow(im_cont, []);
stitle=sprintf('contrast thresholding with threshold = 127');
title(stitle);

load('mandrill.mat');
im=im2uint8(ind2gray(X,map));
im_cont = contrast_thresholding(im, 25);
subplot(1,3,2), imshow(im_cont, []);
stitle=sprintf('contrast thresholding with threshold = 25');
title(stitle);

load('mandrill.mat');
im=im2uint8(ind2gray(X,map));
im_mean = image_mean(im)
im_cont = contrast_thresholding(im, im_mean);
subplot(1,3,3), imshow(im_cont, []);
stitle=sprintf('contrast thresholding with mean threshold');
title(stitle);



function [out_im] = adjust_brightness_lineer(im, x)
    out_im = im + x;
end

function [out_im] = image_negative(im, L)
    out_im = L - im;    
end

function [out_im] = logarithmic_transformation(im, c)
    out_im = c .* log(1+im); 
end

function [out_im] = gamma_correction(im, c, gamma)
    out_im = uint8(c .* (double(im).^ gamma));
end

function [out_im] = adjust_contrast(im, a, s)
    out_im = a .* (im - s) + s;
end

function [out_im] = contrast_strecthing(im, mj, Mj)
    mI = min(im,[],'all');
    MI = max(im,[],'all');
    out_im = (Mj - mj) .* ((im - mI) / (MI - mI)) + mj;  
end

function [out_im] = contrast_thresholding(im, threshold)
    k = find(im < threshold);
    im(k) = 0;

    k = find(im > threshold);
    im(k) = 1;
    out_im = im;    
end


function [mean] = image_mean(im)
    [M,N] = size(im)
    sum = 0;
    for i=1:M
        for k=1:N
            sum = double(sum) + double(im(i,k));
        end
    end  
    mean = sum ./ (M.*N);
end