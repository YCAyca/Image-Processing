clc;
clear all;
close all;

gray_image = imread('cameraman.tif');
rgb_image = imread('peppers.png');
 
% ----- RGB Vs Gray Images -----

figure
subplot(2,1,1), imshow(gray_image)
subplot(2,1,2), imhist(gray_image)

figure
subplot(2,1,1), imshow(rgb_image)
subplot(2,1,2),imhist(rgb_image)

R=imhist(rgb_image(:,:,1));
G=imhist(rgb_image(:,:,2));
B=imhist(rgb_image(:,:,3));

figure, 
hold on,
plot(R,'r') 
plot(G,'g')
plot(B,'b'), 
legend(' Red channel','Green channel','Blue channel');
hold off,

R = [0, 50, 100, 75, 69, 0, 68, 200, 0; 255, 120, 0, 100, 250, 20, 200, 15, 0; 100, 20, 0, 10, 50, 250, 10, 255, 66];
G = [0, 20, 100, 255, 255, 70, 175, 200, 0; 10, 0, 200, 0, 200, 170, 190, 125, 100; 125, 120, 0, 100, 50, 70, 30, 255, 194];
B = [0, 200, 100 255, 150, 70, 10, 90, 85; 100, 50, 150, 0, 200, 180, 255, 120,0; 15, 200, 120, 10, 50, 20, 24, 255, 85];

rgb_image2 = zeros(3,9,3);
rgb_image2(:,:,1) = R;
rgb_image2(:,:,2) = G;
rgb_image2(:,:,3) = B;

disp(rgb_image2)
rgb_image2 = uint8(rgb_image2)

figure
subplot(2,1,1), imshow(rgb_image2)
subplot(2,1,2), imhist(rgb_image2)

%----- pdf, cdf, CDF & Histogram Equalization -------

% First example  - please look at the result on command window, you can try with your own example matrices!

matrix = [3, 1, 1; 1, 7, 6; 0, 2, 1]

histogram = calculate_histogram(matrix, 3)
[M,N] = size(matrix);
pdf = calculate_pdf(histogram, M*N)
cdf = calculate_cdf(histogram, M*N)
image_equalized = histogram_equalization(matrix, histogram)

%--

histogram = calculate_histogram(gray_image, 8);
[M,N] = size(gray_image);
pdf = calculate_pdf(histogram, M*N);
cdf = calculate_cdf(histogram, M*N);
image_equalized = histogram_equalization(gray_image, histogram);

figure
subplot(2,1,1), imshow(image_equalized) 
subplot(2,1,2), imhist(image_equalized) 

figure
subplot(2,1,1), imshow(gray_image)
subplot(2,1,2), imhist(histeq(gray_image))


% --- Histogram Matching ----

matrix = [3, 1, 1; 1, 7, 6; 0, 2, 1]

matrix_target = [2, 3, 3; 4, 2, 4; 5, 4, 6]

histogram = calculate_histogram(matrix, 3)
target_histogram = calculate_histogram(matrix_target, 3)
image_matched = histogram_matching(matrix, histogram, target_histogram)


%---
reference_image = imread('pout.tif');

histogram = calculate_histogram(gray_image, 8);
target_histogram = calculate_histogram(reference_image, 8);
[M,N] = size(gray_image);
image_matched = histogram_matching(gray_image, histogram, target_histogram);

figure
subplot(2,3,1), imshow(gray_image)
subplot(2,3,2), imshow(reference_image)
subplot(2,3,3), imshow(image_matched)

subplot(2,3,4), imhist(gray_image)
subplot(2,3,5), imhist(reference_image)
subplot(2,3,6), imhist(image_matched)

figure
subplot(2,3,1), imshow(gray_image)
subplot(2,3,2), imshow(reference_image)
subplot(2,3,3), imshow(imhistmatch(gray_image, reference_image))

subplot(2,3,4), imhist(gray_image)
subplot(2,3,5), imhist(reference_image)
subplot(2,3,6), imhist(imhistmatch(gray_image, reference_image))

function [histogram] = calculate_histogram(image, gray_scale_level)
    [M,N] = size(image);
    gray_scale_range = power(2,gray_scale_level)-1;
    histogram = zeros(gray_scale_range,1);
    count = 0;
    for j=0:1:gray_scale_range
        for i=1:1:M
            for k=1:1:N
                if image(i,k) == j
                    count = count + 1;
                end
            end
        end
        histogram(j+1) = count;
        count = 0;
    end    
end

function [pdf] = calculate_pdf(histogram, im_size)
    pdf = zeros(size(histogram));
    for i=1:size(histogram)
        pdf(i) = histogram(i)/im_size;
    end
end

function [cdf] = calculate_cdf(histogram, im_size)
    pdf = calculate_pdf(histogram, im_size);
    cdf(1) = pdf(1);
    for i=2:size(pdf)
        cdf(i) = cdf(i-1) + pdf(i);
    end
end

function [equalized_im] = histogram_equalization(image, histogram)
    [M,N] = size(image);
    im_size = M*N;
    tmp = size(histogram);
    gray_scale_range = tmp(1)-1;
    cdf = calculate_cdf(histogram, im_size);
    CDF = round(cdf * gray_scale_range);
    equalized_im = image;
    for i=0:1:gray_scale_range
        equalized_im(image == i) = CDF(i+1);
    end    
end


function [matched_im] = histogram_matching(image, histogram, target_histogram)
    [M,N] = size(image);
    im_size = M*N;
    tmp = size(histogram);
    gray_scale_range = tmp(1)-1;
    cdf = calculate_cdf(histogram, im_size);
    
    tmp = size(target_histogram);
    im_size = 0;
    for i=1:1:tmp(1)
        im_size = im_size + target_histogram(i);
    end    
    cdf_target = calculate_cdf(target_histogram, im_size);
    for i=1:1:gray_scale_range+1
        [Value,Index] =  min(abs(cdf(i)-cdf_target));
        minValue(i) = cdf_target(Index);
        minIndex(i) = Index-1;
    end
    
    matched_im = image;
    for i=0:1:gray_scale_range
        matched_im(image == i) = minIndex(i+1);
    end     
end

