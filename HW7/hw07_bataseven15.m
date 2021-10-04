% Introduction to Machine Learning - HW7: Linear Discriminant Analysis
% Written by Berke Ataseven (54326)
close all
% clear
clc
%% Import data
fprintf('Importing data...')
training_set = readmatrix('hw07_training_images.csv','HeaderLines',0);
training_labels = readmatrix('hw07_training_labels.csv','HeaderLines',0);
fprintf('.')
test_set = readmatrix('hw07_test_images.csv','HeaderLines',0);
test_label = readmatrix('hw07_test_labels.csv','HeaderLines',0);
fprintf(' done\n')
%% Calculate means
K = length(unique(training_labels));
D = size(training_set,2);

average_mean_training = zeros(1,D);
average_mean_test = zeros(1,D);
for k = 1 : K
    class_data_training{k} = training_set(training_labels == k,:);
    class_means_training{k} = mean(cell2mat(class_data_training(k)));
    average_mean_training = average_mean_training + cell2mat(class_means_training(k));
    
    
    class_data_test{k} = test_set(test_label == k,:);
    class_means_test{k} = mean(cell2mat(class_data_test(k)));
    average_mean_test = average_mean_test + cell2mat(class_means_test(k));
end

average_mean_training = average_mean_training / K;
average_mean_test = average_mean_test / K;
%% Calculate Covariances
% For the training set
fprintf('Calculating class covariances.');
for k = 1 : K
    data_of_interest = cell2mat(class_data_training(k));
    N = size(data_of_interest);
    
    MEAN = cell2mat(class_means_training(k));
    sigma = zeros(D,D);
    for t = 1 : N
        sigma = sigma + (data_of_interest(t,:).' - MEAN.') * (data_of_interest(t,:).' - MEAN.').' ;
    end
    class_covariances_training{k} = sigma;
    fprintf('.');
end

% For the test set
for k = 1 : K
    data_of_interest = cell2mat(class_data_test(k));
    [N, D] = size(data_of_interest);
    
    MEAN = cell2mat(class_means_test(k));
    sigma = zeros(D,D);
    for t = 1 : N
        sigma = sigma + (data_of_interest(t,:).' - MEAN.') * (data_of_interest(t,:).' - MEAN.').' ;
    end
    class_covariances_test{k} = sigma;
    fprintf('.');
end
fprintf(' done\n');
%% Between and within class scatters
% Within class scatters
fprintf('Calculating scatters.');
Sw_training = zeros(D,D);
Sw_test = zeros(D,D);
for k = 1 : K
    Sw_training = Sw_training + cell2mat(class_covariances_training(k));
    Sw_test = Sw_test + cell2mat(class_covariances_test(k));
    fprintf('.');
end
Sw_training = Sw_training + eye(D)*1e-10;
Sw_test = Sw_test + eye(D)*1e-10;

% Between class scatters
Sb_training = zeros(D,D);
for k = 1 : K
    Ni = size(cell2mat(class_data_training(k)),1);
    MEAN = cell2mat(class_means_training(k));
    Sb_training = Sb_training + Ni * (MEAN.' - average_mean_training.') * (MEAN.' - average_mean_training.').' ;
end
fprintf('.');
Sb_test = zeros(D,D);
for k = 1 : K
    Ni = size(cell2mat(class_data_test(k)),1);
    MEAN = cell2mat(class_means_test(k));
    Sb_test = Sb_test + Ni * (MEAN.' - average_mean_test.') * (MEAN.' - average_mean_test.').' ;
end
fprintf(' done\n');
%% Calculating W
fprintf('Calculating W...')
[eigen_training, ~] = eig(Sw_training \ Sb_training);
W_training = [eigen_training(:,1) eigen_training(:,2)];

[eigen_test, ~] = eig(Sw_test \ Sb_test);
W_test = [eigen_test(:,1) eigen_test(:,2)];
fprintf(' done\n')
%% Project on to W
class_Z_data_training = {};
class_Z_data_test = {};

for k = 1 : K
    X = cell2mat(class_data_training(k));
    X = X - average_mean_training; % Centering
    class_Z_data_training{k} = W_training.' * X.';
    
    X = cell2mat(class_data_test(k));
    X = X - average_mean_test; % Centering
    class_Z_data_test{k} = W_test.' * X.';
end

figure('Position', [300 100 500 450])
hold on
Z1_training = cell2mat(class_Z_data_training(1));
plot(Z1_training(1,:),Z1_training(2,:),'.r','MarkerSize',10);
Z2_training = cell2mat(class_Z_data_training(2));
plot(Z2_training(1,:),Z2_training(2,:),'.b','MarkerSize',10);
Z3_training = cell2mat(class_Z_data_training(3));
plot(Z3_training(1,:),Z3_training(2,:),'.g','MarkerSize',10);
legend('T-shirt','trouser','dress');
xlabel('Dimension 1');
ylabel('Dimension 2');
title('Training points');

figure('Position', [300 100 500 450])
hold on
Z1_test = cell2mat(class_Z_data_test(1));
Z1_test = [-Z1_test(1,:);Z1_test(2,:)];
plot(Z1_test(1,:),Z1_test(2,:),'.r','MarkerSize',10);

Z2_test = cell2mat(class_Z_data_test(2));
Z2_test = [-Z2_test(1,:); Z2_test(2,:)];
plot(Z2_test(1,:),Z2_test(2,:),'.b','MarkerSize',10);

Z3_test = cell2mat(class_Z_data_test(3));
Z3_test = [-Z3_test(1,:) ;Z3_test(2,:)];
plot(Z3_test(1,:),Z3_test(2,:),'.g','MarkerSize',10);

legend('T-shirt','trouser','dress');
xlabel('Dimension 1');
ylabel('Dimension 2');
title('Test points');
%% KNN Training

Z_training = (W_training.' * (training_set - average_mean_training).').';


trained_labels = train_knn(5, Z_training, training_labels);
Confusion_Matrix = confusionmat(trained_labels,training_labels)

%% KNN Testing
Z_test = (W_test.' * (test_set - average_mean_test).').';
Z_test = [-Z_test(:,1) Z_test(:,2)];

tic
estimated_final = test_knn(5,Z_training,trained_labels,Z_test);
Confusion_Matrix = confusionmat(estimated_final,test_label)
toc
%% Training Function
function trained_labels = train_knn(k_nearest,training_data,training_labels)
trained_labels = zeros(size(training_labels));
N = size(training_data,1);
K = length(unique(training_labels));
for i = 1 : N
    bin = zeros(1,K);
    temp = [training_data training_labels];
    for j = 1 : k_nearest + 1
        [idx, ~] = dsearchn(temp(:,1:2),training_data(i,:));
        if j ~= 1
            bin(temp(idx,3)) = bin(temp(idx,3)) + 1;
        end
        temp(idx,:) = [];
    end
    [~,I] = max(bin);
    trained_labels(i) = I;
end
end
%% Testing Function
function estimated_labels = test_knn(k_nearest, training_data, trained_labels, test_data)
estimated_labels = zeros(size(trained_labels));
N = size(test_data,1);
K = length(unique(trained_labels));
for i = 1 : N
    bin = zeros(1,K);
    temp = [training_data trained_labels];
    for j = 1 : k_nearest
        [idx, ~] = dsearchn(temp(:,1:2),test_data(i,:));
        bin(temp(idx,3)) = bin(temp(idx,3)) + 1;
        temp(idx,:) = [];
    end
    [~,I] = max(bin);
    estimated_labels(i) = I;
end
end
