img = imread('football.jpg');
imgry = rgb2gray(img);

% imfinfo('football.jpg') use if you want to know image type details

figure 
imshow(imgry, []);

figure 
for i=1:8
    quantimg = quant_yca(imgry, (power(2,i)));
%     if i == 5
%         disp(quantimg);
%     end    
    subplot(2,4,i), imshow(quantimg, []);
    stitle=sprintf('gray level : %d bit', i);
    title(stitle);
end

img = imread('cameraman.tif');

[M,N]=size(img);

figure
imshow(img);

figure
k = 1;
for i=2.^[0:5]
    cam_sampled = img(1:i:end, 1:i:end, :);
    subplot(2,4,k), imshow(cam_sampled, []);
    stitle=sprintf('sampling %d x %d pixel ', uint8(M/i), uint8(N/i));
    title(stitle);
    k = k+1;
end


function [out_im] = quant_yca(im, gray_level)
    im_range = 255;
    [M,N] = size(im);
    quant = uint8(im_range/gray_level);
    for k=1:M
        for l=1:N
            for i=gray_level-1:-1:1
                if im(k,l) >= (quant + (quant * (i-1)))
                    im(k,l) = i;
                    break
                elseif i == 1   
                    im(k,l) = 0;
                end    
            end
        end
    end    
    out_im = im;
end


