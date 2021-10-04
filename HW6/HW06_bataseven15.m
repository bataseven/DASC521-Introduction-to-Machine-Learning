% Introduction to Machine Learning - HW6: Expectation-Maximization Clustering
% Written by Berke Ataseven (54326)
clear
close all
clc

rng(10);

class_means = [[2.5; 2.5] [-2.5; 2.5] [-2.5; -2.5] [2.5; -2.5] [0;0]]; % These are given
class_covs = [[0.8 -0.6;-0.6 0.8] [0.8 0.6; 0.6 0.8] [0.8 -0.6; -0.6 0.8] [0.8 0.6; 0.6 0.8] [1.6 0; 0 1.6]]; % These are given
class_sizes = [50 50 50 50 100]; % These are given

data_points = [];

for i = 1 : length(class_sizes)
    data_points = [data_points; mvnrnd(class_means(:, i), class_covs(:,2*i-1: 2*i), class_sizes(i))];
end

label1 = ones(50,1);
label2 = 2*ones(50,1);
label3 = 3*ones(50,1);
label4 = 4*ones(50,1);
label5 = 5*ones(100,1);

data_labels = [label1 ; label2 ; label3 ; label4 ; label5];

figure('Position', [50 0 1200 600])

subplot(1,3,1)
gscatter(data_points(:,1),data_points(:,2),data_labels)
% plot(data_points(:,1),data_points(:,2),'.k','MarkerSize',15);
title('Raw Data');
axis equal
ylabel('x2');
xlabel('x1');

[Z,centers] = kmeans_cluster(data_points, 5, 10);

subplot(1,3,2)
gscatter(data_points(:,1),data_points(:,2),Z)
title('K-Means clustering')
axis equal
ylabel('x2');
xlabel('x1');


[Z, means, covs] = EM_Algorithm(data_points, Z, 5, 100);

subplot(1,3,3)
gscatter(data_points(:,1),data_points(:,2),Z)
title('EM algorithm clustering')
axis equal
ylabel('x2');
xlabel('x1');


function [class_memberships, centroids] = kmeans_cluster(X, K, max_iteration) % X is the data points, K number of classes
N = length(X); % Number of data points
class_memberships = zeros(N,1);
centroids = randi([floor(min(X,[],'all')),ceil(max(X,[],'all'))], K, 2);

for i = 1 : max_iteration
    idx = dsearchn(centroids,X);
    for j = 1 : K
        index_of_members = find(idx == j);
        if ~isempty(index_of_members)
            centroids(j,:) = mean(X(index_of_members,:),1); % Move centroid of classes to center of mass
        end
    end
    if isequal(idx,class_memberships)
        fprintf('Converged in %g iterations',i);
        break;
    end
    class_memberships = idx;
end
end
function [class_memberships, means, covs] = EM_Algorithm(X, Z, K, max_iteration)

N = length(X);
X_Z = [X Z];

% [~,~,ic] = unique(X_Z(:,3));
% seperated_points = accumarray(ic,1:size(X_Z,1),[],@(r){X_Z(r,:)});

means = [];
covs = {};
for i = 1 : K
    points = X_Z(X_Z(:,3) == i,:);
    means = [means ; mean(points(:,1)), mean(points(:,2))];
    covs{i} = [std(points(:,1)) 0; 0 std(points(:,2))];
    points = [];
end

priors = calculate_priors(X_Z, K);

for iteration = 1 : max_iteration
    % E-Step
    for i = 1 : N
        x = X_Z(i, 1:2);
        cluster_likelihoods = [];
        for k = 1 : length(means)
            meann = means(k,:);
            sigma = cell2mat(covs(k));
            prior = priors(k);
            cluster_likelihoods = [cluster_likelihoods calculate_likelihood(x, meann, sigma, prior)]; %calculate likelihood
            [~,idx] = max(cluster_likelihoods); % Find the highest likelihood
            X_Z(i,3) = idx; % Update label
        end
    end
    
    % M-Step
    means = [];
    covs = {};
    for i = 1 : K
        points = X_Z(X_Z(:,3) == i,:);
        means = [means ; mean(points(:,1)), mean(points(:,2))];
        covs{i} = [std(points(:,1)) 0; 0 std(points(:,2))];
        points = [];
    end    
    priors = calculate_priors(X_Z, K);
    
end
means
class_memberships = X_Z(:,3);
end
function likelihood = calculate_likelihood(data_point, mean, cov, prior)
likelihood = prior;
for j = 1 : length(data_point)
    likelihood = likelihood * normpdf(data_point(1,j), mean(1,j), cov(j,j));
end
end
function priors = calculate_priors(X_Z, K)
priors = zeros(1,K);
N = length(X_Z);
for i = 1 : K
    priors(i) = numel(find(X_Z(:,3) == i)) / N;
end
end