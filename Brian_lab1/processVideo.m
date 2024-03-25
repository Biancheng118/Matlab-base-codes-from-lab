clear all
close all

% Specify the video file name
videoFilename = 'Movie1.MOV';

% Create a VideoReader object
videoReader = VideoReader(videoFilename);


% Process every 15th frame
frameCount = 1;
while hasFrame(videoReader)
    % Read the current frame
    currentFrame = readFrame(videoReader);

    % Reduce image size if needed
    if size(currentFrame, 2) > 640
        currentFrame = imresize(currentFrame, 640 / size(currentFrame, 2));
    end

    % Display the current frame
    figure(1), imshow(currentFrame), title('Original Frame');

    % Process the frame using the checkerboard detection function
    findCheckerBoard_students(currentFrame);

    % Display the results
    pause;

    % Skip frames
    for skip = 1:14
        if hasFrame(videoReader)
            readFrame(videoReader);
            frameCount = frameCount + 1;
        else
            break;
        end
    end
end
