%%part4 : use k-means to cluster superpixels
%%provide meaningful clustering
  
k_value = 20; % for k-means clustering 
threshold = 50; % for neighbour pixels  
numberOfSuperPixelsForSlic = 350; 
radius1 = 2; % for first neighbourhood 
radius2 = 3; % for second neighbourhood

fileNames = ["01.png" ; "02.png"; "03.png"; "04.png"; "05.png"; "06.png" ; "07.png" ; "08.png"; "09.png"; "10.png"];
[fileRow, fileColumn] = size(fileNames);

% find representations for each image
for counter = 1: fileRow
    [rep, outLabel] = findRepresentation(fileNames(counter), numberOfSuperPixelsForSlic);
    if(counter == 1)
         [r,c] = size(rep);
         [r2,c2] = size(outLabel);
         sizeOfSuperpixels = r;
         sizeLabels = r2;
         imageRepresentations = rep;
         labelsPerPixel = outLabel;
    else 
        [r,c] = size(rep);
        [r2,c2] = size(outLabel);

        imageRepresentations = [imageRepresentations ; rep];
        sizeOfSuperpixels = [sizeOfSuperpixels ; r ];

        labelsPerPixel = [labelsPerPixel; outLabel];
        sizeLabels = [sizeLabels; r2];
    end
end
labelsPerPixel = labelsPerPixel + 1; % start labels from 1 


startIndex =  sizeOfSuperpixels;
[a,b] = size(sizeOfSuperpixels);
for i = 2: a
    startIndex(i,1) = startIndex(i,1) + startIndex(i-1,1); 
end
startIndex = startIndex + 1;

startIndexForLabels = sizeLabels;
[a,b] = size(sizeLabels);
for i = 2: a
    startIndexForLabels(i,1) = startIndexForLabels(i,1) + startIndexForLabels(i-1,1); 
end
startIndexForLabels = startIndexForLabels + 1;


%k-means clustering
clusterIDs = kmeans(imageRepresentations, k_value);
numberOfClusters = max(clusterIDs);
cmap = colormap(hsv(numberOfClusters));

counterForClustering = 1;
while(counterForClustering <= fileRow)
    r = sizeLabels(counterForClustering);
    [a,c] = size(labelsPerPixel) ;
    falseColorLabel = zeros(r, c);
    if(counterForClustering == 1)
        startingIndexForFalseColor = 1;
        endIndexForFalseColor = sizeLabels(counterForClustering,1);
    else
        startingIndexForFalseColor = startIndexForLabels(counterForClustering - 1,1) ;
        endIndexForFalseColor = startIndexForLabels(counterForClustering - 1, 1) + sizeLabels(counterForClustering,1) - 1;
    end

    %find each pixels cluster ID
    for i = startingIndexForFalseColor : endIndexForFalseColor
        for j = 1: c
            superPixID = labelsPerPixel(i,j);
            clusterNo = clusterIDs(superPixID,1);
            if( counterForClustering == 1 )
                falseColorLabel(i,j) = clusterNo;
            else
                falseColorLabel(i - startingIndexForFalseColor + 1,j) = clusterNo;
            end

        end
    end

    falseColor = label2rgb(falseColorLabel, cmap);

    image = imread(fileNames(counterForClustering));
    figure;
    imshow(image);
    figure;
    imshow(imfuse(image, falseColor,'blend'))
    counterForClustering = counterForClustering + 1;
end

%%%part5 
[rowOfImageRep, colOfImageRep] = size(imageRepresentations); 
neighbourhoodAverageFeatures = zeros(rowOfImageRep,colOfImageRep+80); 

counterForCalculatingAverageFeature = 1;

%%% for each image find neigbourhoods
while (counterForCalculatingAverageFeature <= fileRow) 
    %find center and radius of the superpixel for image 

    if (counterForCalculatingAverageFeature == 1)
        properties = regionprops(labelsPerPixel(1: sizeLabels(counterForCalculatingAverageFeature),:),'centroid', 'EquivDiameter');
    else
        properties = regionprops(labelsPerPixel(startIndexForLabels(counterForCalculatingAverageFeature - 1,1) : startIndexForLabels(counterForCalculatingAverageFeature - 1, 1) + sizeLabels(counterForCalculatingAverageFeature,1) - 1 , :),'centroid', 'EquivDiameter');
    end
    radii = [properties.EquivDiameter].' /2;
    centers = cat(1, properties.Centroid);

    startIndexForComputing = 1;
    endIndexForComputing = sizeOfSuperpixels(counterForCalculatingAverageFeature,1);

    image = imread(fileNames(counterForCalculatingAverageFeature));
    [rowImage, columnImage] = size(image);
    columnImage = columnImage/3;

    %find neighbours
    while(startIndexForComputing <= endIndexForComputing )
            centerX = centers(startIndexForComputing,1);
            centerY = centers(startIndexForComputing,2);
            radius = radii(startIndexForComputing);

            [rows,columns] = meshgrid(1: columnImage, 1:rowImage);
            pixels = (rows-centerX).^2 +(columns-centerY).^2 <= radius.^2;

            %find first neighbourhood 
            %threshold to accept number of pixels between superpixel circle and its
            %neighbourhood
            radiusOfFirstNeighbouthood = radii(startIndexForComputing) * radius1;
            pixelsFirstNeighbourhood = (rows-centerX).^2 +(columns-centerY).^2 <= radiusOfFirstNeighbouthood.^2;
            pixelsFirstNeighbourhood = pixelsFirstNeighbourhood - pixels;

            minimumValue= sizeOfSuperpixels(counterForCalculatingAverageFeature,1); 
            maximumValue= 0;
            for i = 1 : rowImage
                for j = 1 : columnImage
                    if(pixelsFirstNeighbourhood(i,j) == 1)
                        value = labelsPerPixel(i,j);
                        if(value > maximumValue)
                            maximumValue = value;
                        end
                        if(value < minimumValue)
                            minimumValue = value;
                        end
                    end
                end
            end

            length = maximumValue - minimumValue + 1;
            counterOfPixels = zeros(length, 1) - 1;
            superPixLabels = zeros(length, 1) - 1;

            for i = 1 : rowImage
                for j = 1 : columnImage
                    if(pixelsFirstNeighbourhood(i,j) == 1)
                        value = labelsPerPixel(i,j);
                        counterOfPixels(value - minimumValue + 1 ,1) = counterOfPixels(value - minimumValue + 1 ,1) + 1; 
                        superPixLabels(value - minimumValue + 1 ,1) = value;
                    end
                end
            end
            counterOfPixels ;
            superPixLabels;
            numberOfFirstNeighbourhood = 0;


            for i = 1: length
                if(counterOfPixels(i,1) >= threshold) % threshold for pixel numbers of a superpixel to be accepted as neighbourhood
                    numberOfFirstNeighbourhood = numberOfFirstNeighbourhood + 1;
                end
            end
            firstNeighbourhoodVector = zeros(numberOfFirstNeighbourhood, 40);
            counter = 1;
            for i = 1: length
                if(counterOfPixels(i,1) >= threshold) % threshold for pixel numbers of a superpixel to be accepted as neighbourhood
                    IDofSuperPixel = superPixLabels(i,1);
                    if( counterForCalculatingAverageFeature == 1)
                        firstNeighbourhoodVector(counter,:) = imageRepresentations( IDofSuperPixel, :);
                    else
                        firstNeighbourhoodVector(counter,:) = imageRepresentations( startIndex(counterForCalculatingAverageFeature - 1,1) + IDofSuperPixel - 1, :);
                    end
                    counter = counter + 1;
                end
            end

            summationOfVectors = zeros(1,40);
            for i = 1: numberOfFirstNeighbourhood

                   summationOfVectors(1,:) =  summationOfVectors(1,:) + firstNeighbourhoodVector(i,:);
            end
            summationOfVectors = summationOfVectors/ numberOfFirstNeighbourhood;
            
            %%%%secondNeighbourhood 

            radiusOfSecondNeighbourhood = radii(startIndexForComputing) * radius2;
            pixelsSecondNeighbourhood = (rows-centerX).^2 +(columns-centerY).^2 <= radiusOfSecondNeighbourhood.^2;
            pixelsSecondNeighbourhood = pixelsSecondNeighbourhood - pixelsFirstNeighbourhood - pixels;

            minimumValue= sizeOfSuperpixels(counterForCalculatingAverageFeature,1);
            maximumValue= 0;
            for i = 1 : rowImage
                for j = 1 : columnImage
                    if(pixelsSecondNeighbourhood(i,j) == 1)
                        value = labelsPerPixel(i,j);
                        if(value > maximumValue)
                            maximumValue = value;
                        end
                        if(value < minimumValue)
                            minimumValue = value;
                        end
                    end
                end
            end

            length = maximumValue - minimumValue + 1;
            counterOfPixels = zeros(length, 1) - 1;
            superPixLabels = zeros(length, 1) - 1;

            for i = 1 : rowImage
                for j = 1 : columnImage
                    if(pixelsSecondNeighbourhood(i,j) == 1)
                        value = labelsPerPixel(i,j);
                        counterOfPixels(value - minimumValue + 1 ,1) = counterOfPixels(value - minimumValue + 1 ,1) + 1; 
                        superPixLabels(value - minimumValue + 1 ,1) = value;
                    end
                end
            end
            counterOfPixels; 
            superPixLabels;
            numberOfSecondNeighbourhood = 0;

            for i = 1: length
                if(counterOfPixels(i,1) >= threshold) % threshold for pixel numbers of a superpixel to be accepted as neighbourhood
                    numberOfSecondNeighbourhood = numberOfSecondNeighbourhood + 1;
                end
            end
            secondNeighbourhoodVector = zeros(numberOfSecondNeighbourhood, 40);
            counter = 1;
            for i = 1: length
                if(counterOfPixels(i,1) >= threshold) % threshold for pixel numbers of a superpixel to be accepted as neighbourhood
                    IDofSuperPixel = superPixLabels(i,1);
                    if( counterForCalculatingAverageFeature == 1)
                        secondNeighbourhoodVector(counter,:) = imageRepresentations( IDofSuperPixel, :);
                    else
                        secondNeighbourhoodVector(counter,:) = imageRepresentations( startIndex(counterForCalculatingAverageFeature - 1,1) + IDofSuperPixel - 1, :);
                    end
                    counter = counter + 1;
                end
            end

            summationOfVectors2 = zeros(1,40);
            for i = 1: numberOfSecondNeighbourhood

                   summationOfVectors2(1,:) =  summationOfVectors2(1,:) + secondNeighbourhoodVector(i,:);
            end
            summationOfVectors2 = summationOfVectors2/ numberOfSecondNeighbourhood;


            if(counterForCalculatingAverageFeature == 1)
                neighbourhoodAverageFeatures(startIndexForComputing,:) = [ imageRepresentations( startIndexForComputing, :) summationOfVectors summationOfVectors2];
            else
                neighbourhoodAverageFeatures(startIndex(counterForCalculatingAverageFeature - 1,1) + startIndexForComputing - 1,:) = [ imageRepresentations(  startIndex(counterForCalculatingAverageFeature - 1,1) + startIndexForComputing -1 , :) summationOfVectors summationOfVectors2];
            end
            startIndexForComputing = startIndexForComputing + 1;
    end
    counterForCalculatingAverageFeature = counterForCalculatingAverageFeature + 1;

end

%K-Means clustering for neighbourhood
clusterIDs = kmeans(neighbourhoodAverageFeatures, k_value);
numberOfClusters = max(clusterIDs);
cmap = colormap(hsv(numberOfClusters));

counterForClustering = 1;
while(counterForClustering <= fileRow)
    r = sizeLabels(counterForClustering);
    [a,c] = size(labelsPerPixel) ;
    falseColorLabel = zeros(r, c);
    if(counterForClustering == 1)
        startingIndexForFalseColor = 1;
        endIndexForFalseColor = sizeLabels(counterForClustering,1);
    else
        startingIndexForFalseColor = startIndexForLabels(counterForClustering - 1,1) ;
        endIndexForFalseColor = startIndexForLabels(counterForClustering - 1, 1) + sizeLabels(counterForClustering,1) - 1;
    end

    for i = startingIndexForFalseColor : endIndexForFalseColor
        for j = 1: c
            superPixID = labelsPerPixel(i,j);
            clusterNo = clusterIDs(superPixID,1);
            if( counterForClustering == 1 )
                falseColorLabel(i,j) = clusterNo;
            else
                falseColorLabel(i - startingIndexForFalseColor + 1,j) = clusterNo;
            end

        end
    end
    falseColor = label2rgb(falseColorLabel, cmap);
    image = imread(fileNames(counterForClustering));
    figure;
    imshow(image);
    figure;
    imshow(imfuse(image, falseColor,'blend'))
    counterForClustering = counterForClustering + 1;
end
    
    
    
        
    
    
function [output1, output2] = findRepresentation(fileName , numberOfSuperPixelsForSlic)
image = imread(fileName);

[labels, numberOfLabels] = slicomex(image,numberOfSuperPixelsForSlic); %250
figure;
boundaries = boundarymask(labels);
imshow(imoverlay(image,boundaries,'cyan'))

%number of required superpixels = 300, compactness factor = 10
%[labels, numberOfLabels] = slicmex(image,350,10); 
% figure;
% boundaries = boundarymask(labels);
% imshow(imoverlay(image,boundaries,'cyan'))

%get gray scale of image
grayImage = rgb2gray(image);

%obtain a filter bank of 4 scales and 4 orientations
wavelengths = [5 10 15 20];
orientations = [0 45 90 135]; 
filterBank = gabor(wavelengths, orientations);

% gabor filter 
[magnitude, phase] = imgaborfilt(grayImage, filterBank);

%magnitude
%size(magnitude); % 16 filter responses for each pixel 
%gaborResult = imgaborfilt(grayImage, filterBank);
% figure;
% subplot(4,4,1);
% for c = 1 : 16
%     subplot(4,4, c)
%     imshow(gaborResult(:,:,c), []);
%     theta = filterBank(c).Orientation;
%     lambda = filterBank(c).Wavelength;
%     title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
% end

%compute average of gabor features for each superpixel
[labelsRow, labelsColumn] = size(labels);

numberOfSuperpixels = numberOfLabels; % labeling starts from 0

counterOfPixels = zeros(numberOfSuperpixels, 16);
averageOfSuperpixelFeatures = zeros(numberOfSuperpixels, 16);

% collect summation in averageOfSuperpixelFeatures
% count the number of pixels that belong to same superPixID 
for k = 1 : 16
    for i = 1: labelsRow
        for j = 1:labelsColumn
            superPixID = labels(i,j) + 1;
            averageOfSuperpixelFeatures(superPixID,k) = averageOfSuperpixelFeatures(superPixID,k) + magnitude(i,j,k);
            counterOfPixels(superPixID,k) = counterOfPixels(superPixID,k) + 1;
        end
    end
end


% number of pixels x 16 texture feature matrix 
 for i = 1 : numberOfSuperpixels
     for j = 1 : 16
         averageOfSuperpixelFeatures(i,j) = averageOfSuperpixelFeatures(i,j)/ counterOfPixels(i,1);
     end
 end

%%compute color respresentation for each superpixel
%convert image to LAB color space
labImage = rgb2lab(image);

lValue = labImage(:,:,1);
aValue = labImage(:,:,2);
bValue = labImage(:,:,3);

[rowColor, columnColor] = size(lValue); %460x700
%%compute histogram of length 24 for each superpixel by dividing each
%%channel into 8 bins independently

%l value between 0 and 100
%0-12,5 ; 12,5-25; 25-37,5 ; 37,5-50; 50-62,5 ; 62,5-75; 75-87,5 ; 87,5-100 
l_value_Counter = zeros(numberOfSuperpixels,8);

%according to implementation of matlab a and b values are roughly between
%-110 and 110
% -110- -82,5 ; -82,5 - -55 ; -55 - -27,5 ; -27,5- 0 ; 0-27,5 ; 27,5- 55; 
% 55- 82,5 ; 82,5-110

a_value_Counter = zeros(numberOfSuperpixels,8);
b_value_Counter = zeros(numberOfSuperpixels,8);

%%for N superpixels X 24 feature matrix
for i = 1 : rowColor
    for j = 1 : columnColor
       l=lValue(i,j);
       a = aValue(i,j);
       b = bValue(i,j);
       
       l_index = floor( double(l) / 12.5 );
       a_index = floor (double ( a ) / 27.5);
       b_index = floor (double ( b )/ 27.5 );  
       superPixID = labels(i,j) + 1;
       if(l_index == 8)
           l_index = 7;
       end
       l_value_Counter(superPixID, l_index + 1) =  l_value_Counter(superPixID, l_index + 1 ) + 1;
       a_value_Counter(superPixID, a_index + 5) = a_value_Counter(superPixID, a_index + 5) + 1;
       b_value_Counter(superPixID, b_index + 5) = b_value_Counter(superPixID, b_index + 5) + 1;
    end
end

color_representation = [l_value_Counter a_value_Counter b_value_Counter ];


%%concatanate texture and color representations for each superpixel
%%normalize the vectors before concatenation

for i = 1: numberOfSuperpixels
    
    color_representation(i, :) = color_representation(i, :) / norm(color_representation(i, :));

    averageOfSuperpixelFeatures(i,:) = averageOfSuperpixelFeatures(i,:) /  norm(averageOfSuperpixelFeatures(i, :));

    concatenatedRepresentations = [color_representation averageOfSuperpixelFeatures];
    
end


output1 = concatenatedRepresentations;
output2 = labels;

end











