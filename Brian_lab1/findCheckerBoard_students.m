% Find a 8x8 checkerboard in the image I.

function findCheckerBoard_students(I)
    if size(I,3)>1
        I = rgb2gray(I);
    end
    
    % Do edge detection using canny.
    %try different thresholds (0.5thresh - 5 thresh) to get clean edges
    
    %Students write your code here - use E as the name of edge image

    % Perform Otsu's thresholding to determine the edge detection threshold
    otsu_thresh = graythresh(I);

    % Use Canny edge detection with the computed threshold
    E = edge(I, 'canny', otsu_thresh);

    % Display the edge-detected image
    figure(9), imshow(E), title('Edges');
    pause
    
    % Do Hough transform to find lines.
    [H,thetaValues,rhoValues] = hough(E);
    
    % Extract peaks from the Hough array H. Parameters for this:
    % houghThresh: Minimum value to be considered a peak. Default
    % is 0.5*max(H(:))
        
    %try different number of peaks and different thresholds 
    
    %Students write your code here 

    % Define the number of peaks to be detected in the Hough transform
    nPeaks =50;
    % Determine the threshold for peak detection based on a percentage of the maximum Hough transform value
    % This thresholding helps identify prominent peaks in the Hough space
    myThresh = ceil(0.35*max(H(:)));
    % Use the houghpeaks function to find peaks in the Hough transform matrix H
    % The 'Threshold' parameter specifies the minimum value for a peak to be considered
    peaks = houghpeaks(H,nPeaks,'Threshold',myThresh);
    
    %Display Hough array and draw peaks on Hough array.
    figure(10), imshow(H, []), title('Hough'), impixelinfo;
    for i=1:size(peaks,1)
        rectangle('Position', ...
        [peaks(i,2)-3, peaks(i,1)-3, ...
        7, 7], 'EdgeColor', 'r');
    end
    pause
    
    % Show all lines.
    figure(11), imshow(E), title('All lines');
    drawLines( ...
    rhoValues(peaks(:,1)), ... % rhos for the lines
    thetaValues(peaks(:,2)), ... % thetas for the lines
    size(E), ... % size of image being displayed
    'y'); % color of line to display
    pause
    
    
    % Find two sets of orthogonal lines.
    [lines1, lines2] = findOrthogonalLines( ...
    rhoValues(peaks(:,1)), ... % rhos for the lines
    thetaValues(peaks(:,2))); % thetas for the lines
    pause



    % Sort the lines, from top to bottom (for horizontal lines) and left to
    % right (for vertical lines).
    lines1 = sortLines(lines1);
    lines2 = sortLines(lines2);
    
    % Show the two sets of lines.
    figure(12), imshow(E), title('Orthogonal lines');
    drawLines( lines1(2,:), ... % rhos for the lines
        lines1(1,:), ... % thetas for the lines
        size(E), ... % size of image being displayed
        'g'); % color of line to display
    
    drawLines( lines2(2,:), ... % rhos for the lines
        lines2(1,:), ... % thetas for the lines
        size(E), ... % size of image being displayed
        'r'); % color of line to display
    fprintf('Number of lines for horizontal angle: %d\n', size(lines1, 2));
    fprintf('Number of lines for vertical angle: %d\n', size(lines2, 2));

    %find the outer pair of lines
    lines11=[lines1(:,1) lines1(:,end)];
    lines22=[lines2(:,1) lines2(:,end)];

    % Intersect the outer pair of lines, one from set 1 and one from set 2.
    % Output is the x,y coordinates of the intersections:
    % xIntersections(i1,i2): x coord of intersection of i1 and i2
    % yIntersections(i1,i2): y coord of intersection of i1 and i2
    [xIntersections, yIntersections] = findIntersections(lines11, lines22);
    
    % Plot outer lines and intersection points.
    figure(1), 
    hold on
    drawLines( lines11(2,:), ... % rhos for the lines
        lines11(1,:), ... % thetas for the lines
        size(E), ... % size of image being displayed
        'g'); % color of line to display
    
    drawLines( lines22(2,:), ... % rhos for the lines
        lines22(1,:), ... % thetas for the lines
        size(E), ... % size of image being displayed
        'r'); % color of line to display
    
    plot(xIntersections(:),yIntersections(:),'s', 'MarkerSize',10, 'MarkerFaceColor','r');
    hold off
    
end

function drawLines(rhos, thetas, imageSize, color)
% This function draws lines on whatever image is being displayed.
% Input parameters:
% rhos,thetas: representation of the line (theta in degrees)
% imageSize: [height,width] of image being displayed
% color: color of line to draw
% Equation of the line is rho = x cos(theta) + y sin(theta), or
% y = (rho - x*cos(theta))/sin(theta)

for i=1:length(thetas)
%if majority of angles are >45 then line is is mostly horizontal. 
%Pick two values of x, and solve for y = (-ax-c)/b
%else the line is is mostly horizontal. Pick two values of y,
% and solve for x

%students write your code here

    % Convert theta to radians
    theta=thetas(i)*pi/180;
    % Extract rho for this line
    rho=rhos(i);
     % Compute coefficients for the line equation in the form ax + by + c = 0
    a=cos(theta);
    c=-rho;
    b=sin(theta);
     % Check if the line is mostly vertical (theta > 45 degrees)
    if(theta>45)
        % If mostly vertical, choose two x values and solve for corresponding y values
        x0 = 1;
        x0=1;
        x1=imageSize(2);
        y0=(-a*x0-c)/b;
        y1=(-a*x1-c)/b;
    else
        % If mostly horizontal, choose two y values and solve for corresponding x values
        y0=1;
        y1=imageSize(2);
        x0=(b*y0+c)/(-a);
        x1=(b*y1+c)/(-a);
    end
% Draw the line on the image
line([x0 x1], [y0 y1], 'Color', color);
 % Add text to display the line index
text(x0,y0,sprintf('%d', i), 'Color', color);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find two sets of orthogonal lines.
% Outputs:
% lines1, lines2: the two sets of lines, each stored as a 2xN array,
% where each column is [theta;rho]

function [lines1, lines2] = findOrthogonalLines( ...
rhoValues, ... % rhos for the lines
thetaValues) % thetas for the lines

% Find the largest two modes in the distribution of angles.
%create a set of angles in an array called bins from -90 to 90 with step 10
%then use histcounts to get the histogram
%sort and the first angle corresponds to the largest histogram count.
% The 2nd angle corresponds to the next largest count.


%students write your code here
 % Find the largest two modes in the distribution of angles.
    % create a set of angles in an array called bins from -90 to 90 with step 10
    ang_i = -90;
    ang_f = 90;
    ang_step = 10;
    ang_n = (ang_f - ang_i) / ang_step + 1;
    bins = ang_i:ang_step:ang_f;

    % use histcounts to get the histogram
    [counts, ~] = histcounts(thetaValues, bins);

    % sort and the first angle corresponds to the largest histogram count.
    % The 2nd angle corresponds to the next largest count.
    [~, indices] = sort(counts, 'descend');

    nIndices = length(indices);

    % Find most common angle for horizontal lines
    hAngle_index = 0;
    for i = nIndices:-1:1
        tmp_angle = bins(indices(i));
        if abs(tmp_angle) > 45
            % the line is mostly horizontal
            hAngle_index = tmp_angle;
            i = nIndices + 1;
        end
    end

    % Find most common angle for vertical lines
    vAngle_index = 0;
    for i = nIndices:-1:1
        tmp_angle = bins(indices(i));
        if abs(tmp_angle) <= 45
            % the line is mostly vertical
            vAngle_index = tmp_angle;
            i = nIndices + 1;
        end
    end

    a1 = vAngle_index;
    a2 = hAngle_index;

    fprintf('Most common angles: %f and %f\n', a1, a2);



% Get the two sets of lines corresponding to the two angles. Lines will
% be a 2xN array, where

lines1 = [];
lines2 = [];
for i=1:length(rhoValues)
% Extract rho, theta for this line
r = rhoValues(i);
t = thetaValues(i);
% Check if the line is close to one of the two angles.
D = 25; % threshold difference in angle
if abs(t-a1) < D || abs(t-180-a1) < D || abs(t+180-a1) < D
    lines1 = [lines1 [t;r]];
elseif abs(t-a2) < D || abs(t-180-a2) < D || abs(t+180-a2) < D
    lines2 = [lines2 [t;r]];
end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sort the lines.

function lines = sortLines(lines)

t = lines(1,:); % Get all thetas
r = lines(2,:); % Get all rhos
% If most angles are between -45 .. +45 degrees, lines are mostly
% vertical.
nLines = size(lines,2);
nVertical = sum(abs(t)<=45);
if nVertical/nLines >= 0.5
% Mostly vertical lines. r=x*cos(t)+y*sin(t) 
% we need x assuming y =1 
dist = (-sind(t)*1 + r)./cosd(t); % horizontal distance
else
% Mostly horizontal lines. 
% we need y assuming x=1
dist = (-cosd(t)*1 + r)./sind(t); % vertical distance
end
[~,indices] = sort(dist, 'ascend');
lines = lines(:,indices);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intersect pairs of lines, one from set 1 and one from set 2.
% Output arrays contain the x,y coordinates of the intersections of lines.
% xIntersections(i1,i2): x coord of intersection of i1 and i2
% yIntersections(i1,i2): y coord of intersection of i1 and i2
function [xIntersections, yIntersections] = findIntersections(lines1, lines2)
    N1 = size(lines1,2);
    N2 = size(lines2,2);
    xIntersections = zeros(N1,N2);
    yIntersections = zeros(N1,N2);
 
    %see Szeliski book section 2.1.1
    % A line is represented in homogenous coordiante system by (a,b,c), 
    % where ax+by+c=0.
    % We have r = x cos(t) + y sin(t), or x cos(t) + y sin(t) - r = 0
    % Two lines l1 and l2 intersect at a point p where p = l1 cross l2
    
    for i1=1:N1
        
        %students write your code here

        % Extract rho and theta for the first line in set 1
        r1 = lines1(2,i1);
        t1 = lines1(1,i1);
        % Convert theta to radians
        t1 = t1*pi/180;
        % Create a vector l1 representing the line in homogeneous coordinates (ax + by + c = 0)
        l1 = [cos(t1) sin(t1) -r1];
        
        % Iterate through all lines in set 2 to find intersections
        for i2=1:N2
            
            %students write your code here

             % Extract rho and theta for the second line in set 2
            r2 = lines2(2,i2);
            t2 = lines2(1,i2);
            % Convert theta to radians
            t2 = t2*pi/180;
             % Create a vector l2 representing the line in homogeneous coordinates
            l2 = [cos(t2) sin(t2) -r2];

             % Calculate the intersection point between lines l1 and l2
            p=cross(l1,l2);
            %bring point back from homogenous coord to the 2D coord
            p = p/p(3);
            % Store the x and y coordinates of the intersection
            xIntersections(i1,i2) = p(1);
            yIntersections(i1,i2) = p(2);
        end
    end
end

