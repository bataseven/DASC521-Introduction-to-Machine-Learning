% Introduction to Machine Learning - HW3 
% Written by Berke Ataseven (54326)

close all
clear all
clc

%% Import data
labels = char(readcell('hw03_data_set_labels.csv'));
data_set = readmatrix('hw03_data_set_images.csv','HeaderLines',0);
%% Classify data as training and test
training_data_set = [];
training_labels = [];

testing_data_set = [];
testing_labels = [];

testing_index = 1;
training_index = 1;

for i = 1 : size(data_set,1) % Split the dataset in to training and testing
    
    if mod(i-1, 39) < 25 % First 25 element of each class is training data
        training_data_set = [training_data_set ; data_set(i,:)];
        if labels(i) == 'A'
            training_labels(training_index,1) = 1;
        elseif labels(i) == 'B'
            training_labels(training_index,1) = 2;
        elseif labels(i) == 'C'
            training_labels(training_index,1) = 3;
        elseif labels(i) == 'D'
            training_labels(training_index,1) = 4;
        else
            training_labels(training_index,1) = 5;
        end
        training_index = training_index + 1;
        
        
    else % last 14 image of each class is testing data
        testing_data_set = [testing_data_set; data_set(i,:)];
        if labels(i) == 'A'
            testing_labels(testing_index,1) = 1;
        elseif labels(i) == 'B'
            testing_labels(testing_index,1) = 2;
        elseif labels(i) == 'C'
            testing_labels(testing_index,1) = 3;
        elseif labels(i) == 'D'
            testing_labels(testing_index,1) = 4;
        else
            testing_labels(testing_index,1) = 5;
        end
        testing_index = testing_index + 1;
    end
end

%% Determine prior probabilities of classes
class_priors = estimate_class_prior(training_labels); % Class prior probability
classes = unique(training_labels); % Classes
K = length(classes); % Number of classes

p_hats = [];
%% Estimate P1,P2,P3,P4,P5 hats
for i = 1 : K
    p_hats = [p_hats ; estimate_Pij(training_data_set, training_labels, i)]; % p hat estimations
end
%% Draw figures
figure('Position', [50 300 1200 200]) % Set the canvas size
for class = 1 : K
    p_hat_image = reshape(p_hats(class,:),[20,16]); % Reshape the column vector to form 20 by 16 pixels
    subplot(1,5,class) % Draw figures side by side
    img = image((p_hat_image) * 255); % Map the pixels between 0 to 255 values
    colormap(flipud(gray)); % Inverse gray scale as the color map
end

%% Evaluate training data set with the constructed model
scores = [];

for class = 1 : K   % Create scores of the classes for the first data in the test set
    class_score = [];
    for i = 1 : size(training_data_set,1)
        class_score = [class_score; calculate_score(training_data_set(i,:), p_hats(class,:) ,class_priors(class))];
    end
    scores = [scores class_score];
end
[~, I] = max(scores,[],2);
Training_data_ConfusionMatrix  = confusionmat(training_labels, I)
%% Evaluate testing data set with the constructed model
scores = [];
for class = 1 : K   % Create scores of the classes for the first data in the test set
    class_score = [];
    for i = 1 : size(testing_data_set,1)
        class_score = [class_score; calculate_score(testing_data_set(i,:), p_hats(class,:) ,class_priors(class))];
    end
    scores = [scores class_score];
end
[~,I] = max(scores,[],2);
Testing_data_ConfusionMatrix  = confusionmat(testing_labels, I)
%% Functions
function scr = calculate_score(data_vector, p_ij, prior_prob)%Calculate score to determine discriminant
summ = 0;
d_feature = size(data_vector,2);
for j = 1 : d_feature
     summ = summ + (data_vector(j) * safelog(p_ij(j)) + (1-data_vector(j)) * safelog(1 - p_ij(j)));
end
scr = summ + safelog(prior_prob);
end
function p_ij = estimate_Pij(data_set, data_labels, class) % Calculates p hat for a single class

d_feature = size(data_set,2);
N_data = size(data_set,1);
p_ij = zeros(1,d_feature);

for j = 1 : d_feature
    sum_num = 0;
    sum_denom = 0;
    for t = 1 : N_data
        current_label = data_labels(t,1);
        if current_label == class
            r_it = 1;
        else
            r_it = 0;
        end
        sum_num = sum_num + (data_set(t,j) * r_it);
        sum_denom = sum_denom + r_it;
    end
    p_ij(1, j) = sum_num/sum_denom;
end
end
function priors = estimate_class_prior(labels) % Class prior probabilites
data_count = size(labels,1);
histogram = zeros(size(unique(labels),1),1);
for i = 1 : data_count
    histogram(labels(i,1)) = histogram(labels(i,1)) + 1;
end
priors = histogram / data_count;
end
function y = safelog(x) % Convert 0 elements to eps(2.2204e-16)
zero_indexes = find(x==0);
for i = 1 : length(zero_indexes)
    x(zero_indexes(i)) = eps;
end
y = log(x);
end