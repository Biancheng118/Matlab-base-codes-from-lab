clear all; close all;
% Q1 image
im1=imread('fig1039a.png');
% Q2 image
im2=imread('shapes23.png');
% Q2-2 image
im3=imread('office3027.jpg');
% Q3 image
im5=imread('gateway_arch.jpg');

if size(im1,3)>1
    im1=rgb2gray(im1);
end
im1=double(im1);

if size(im2,3)>1
    im2=rgb2gray(im2);
end
im2=double(im2);

if size(im3,3)>1
    im3=rgb2gray(im3);
end
im3=double(im3);

if size(im5,3)>1
    im5=rgb2gray(im5);
end
im5=double(im5);


smoothedImage = medfilt2(im1);

% Detect circles in the image
[centers, radii] = imfindcircles(smoothedImage, [10 100], 'ObjectPolarity', 'dark', 'Sensitivity', 0.92);

% Sort the radii and get the indices of the largest two
[sortedRadii, sortIndex] = sort(radii, 'descend');

% Check if at least two circles were found
if length(sortedRadii) >= 2
    % Get the centers of the largest two circles
    largestCenters = centers(sortIndex(1:2), :);

    % Get the radii of the largest two circles
    largestRadii = sortedRadii(1:2);

    % Draw only the largest two circles
    figure(1);
    imshow(im1, []);
    viscircles(largestCenters, largestRadii, 'EdgeColor', 'r');
else
    disp('Less than two circles detected.');
end


%algorithm parameters
%students should change the values below and report their findings
% Q2
 sigma=1;
 thresh=150000000;
 radius=4;
 alpha=0.06;

 
% Q2-2
% sigma=3;
% thresh=50000;
% radius=4;
% alpha=0.09;

% Use sobel operators
% Derivative masks

%students fill this code
% Compute image derivatives using Sobel operators
dx = [-1 -1 -1; 0 0 0; 1 1 1];  % Sobel operator for x-direction
dy = dx';                        % Sobel operator for y-direction

% Caluclate the derivatives
%students fill this code - use correlation with dx and dy
Ix = imfilter(im3,dx,'same');
Iy = imfilter(im3,dy,'same');


% Create Gaussian filter of size 6*sigma (+/- 3sigma) 
% and of minimum size 1x1.
%students fill this code
g = fspecial('gaussian', round(6*sigma), sigma);

% Generate the Smoothed image derivatives Ix^2,Iy^2, and Ixy
% Ix2 = conv2(Ix.^2, g); 
% Iy2 = conv2(Iy.^2, g);
% Ixy = conv2(Ix.*Iy, g);
Sxx = conv2(Ix.^2, g, 'same'); 
Syy = conv2(Iy.^2, g, 'same');
Sxy = conv2(Ix.*Iy, g, 'same');
   
%Calculate cornerness measure R
% students try Harris and Szelski methods

% Calculate cornerness measure R using Harris method
R = (Sxx.*Syy - Sxy.^2)-alpha*((Sxx + Syy).^2); 

% Thresholding and non-maximum suppression
% R(isnan(R)) = 0;
% Threshold R
N = 2 * radius + 1; 
Rdilated = imdilate(R, strel('disk',N));
corners = R > thresh & (R == Rdilated); % Find local maxima

	
% find all corners above a certain thresh
[r,c] = find(corners);                 % Find row,col coords.
% fprintf('sigma: %d\n', thresh);

%Part 1.2 in the assignment
%Students: refine the code to here to extract local maxima 
%by performing a grey scale morphological dilation 
%and then finding points in the corner strength image thatå
% match the dilated image and are also greater than the threshold.

% overlay corners on original image
% Q2
  % figure(2), imagesc(im2), axis image, colormap(gray), hold on
  % 	    plot(c,r,'rs'), title('corners detected');
% Q2-2
   figure(3), imagesc(im3), axis image, colormap(gray), hold on
   	    plot(c,r,'r*'), title('corners detected');

E = edge(im5, 'canny',0.3,0.9);
% figure(4), imshow(E,[]);
% choose parabola sizes to try 
 C = 0.01:0.001:0.1;
 % C = 0.005:0.001:0.015;
c_length = numel(C); 
[M,N] = size(im5);
% accumulator array H(N,M,C) initialized with zeros 
H = zeros(M,N,c_length);
% vote to fill H 
[y_edge, x_edge] = find(E); % get edge points
for i = 1:length(x_edge)    % for all edge points
    for c_idx=1:c_length    % for all c
        for a = 1:N
            b = round(y_edge(i)-C(c_idx)*(x_edge(i)-a)^2);
            if(b < M && b >= 1) H(b,a,c_idx)=H(b,a,c_idx)+1; 
            end
        end
    end
end
% show only third slice of H 
figure(5);  
imshow(H(:,:,3),[]); 
% title(sprintf('Slice of H at C = %f', C(3))); 
% we need code here to find the peaks and draw the parabolas


% Set a significant threshold - adjust based on your image characteristics
some_significant_threshold = max(H(:)) * 0.5;

% Find the global maximum in the Hough space
[max_value, max_index] = max(H(:));

% If no significant parabola was found, exit
if max_value < some_significant_threshold
    disp('No significant parabola detected.');
    return;
end

% Convert the linear index to subscripts
[b_max, a_max, c_idx_max] = ind2sub(size(H), max_index);

% Generate y values for the most significant a and c
x_values = 1:N;
y_values = round(C(c_idx_max)*(x_values - a_max).^2 + b_max);

% Filter out invalid y values
valid_indices = y_values >= 1 & y_values <= M;
y_values = y_values(valid_indices);
x_values = x_values(valid_indices);

% Overlay the most significant parabola on the original image
figure(5);  
imshow(im5, []);
hold on; 
plot(x_values, y_values, 'r.', 'MarkerSize', 3); 
title('Most Significant Detected Parabola');
hold off;

