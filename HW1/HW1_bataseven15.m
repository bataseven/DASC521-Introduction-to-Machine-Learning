clc
rng(4); % You can change the value inside to observe for different random data points

mesh_number = 100; % Increasing this value results in a more continuos discriminant line between classes
                   % However, computation time is proportional to n^2

class_means = [[0; 2.5] [-2.5; -2.0] [2.5; -2.0]];                      % These are given
class_covs = [[3.2  0; 0 1.2] [1.2 -0.8; -0.8 1.2] [1.2 0.8; 0.8 1.2]]; % These are given
class_sizes = [120 90 90];                                              % These are given

red_points = mvnrnd(class_means(:, 1), class_covs(:,1:2), class_sizes(1));  % Create random bivariate normal points
green_points = mvnrnd(class_means(:, 2), class_covs(:,3:4), class_sizes(2));% Create random bivariate normal points
blue_points = mvnrnd(class_means(:, 3), class_covs(:,5:6), class_sizes(3)); % Create random bivariate normal points

priors = estimate_prior(red_points,green_points,blue_points); %D etermine priors

data_points = {};
% Store class parameters in a single cell array (Data point coordinates, mean estimates, cov estimates, priors)
data_points{1,1} = red_points;
data_points{1,2} = green_points;
data_points{1,3} = blue_points;

data_points{2,1} = estimate_mean(red_points);
data_points{2,2} = estimate_mean(green_points);
data_points{2,3} = estimate_mean(blue_points);

data_points{3,1} = estimate_cov(red_points);
data_points{3,2} = estimate_cov(green_points);
data_points{3,3} = estimate_cov(blue_points);

data_points{4,1} = priors(1);
data_points{4,2} = priors(2);
data_points{4,3} = priors(3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = linspace(-6,6, mesh_number); % Create Mesh
x2 = linspace(-6,6, mesh_number);
[X1, X2] = meshgrid(x1,x2);

score1 = [];
score2 = [];
score3 = [];

mislabeled_points = []; % Store points in 

for i = 1 : size(data_points,2)
    score1 =  calculate_score(data_points{1,i}, data_points{2,1}, data_points{3,1}, data_points{4,1});
    score2 =  calculate_score(data_points{1,i}, data_points{2,2}, data_points{3,2}, data_points{4,2});
    score3 =  calculate_score(data_points{1,i}, data_points{2,3}, data_points{3,3}, data_points{4,3});
    scores = [score1 score2 score3];
    [M, I] = max(scores,[],2); % Returns the max score among the 3 class
    for j = 1 : size(I,1)
    if I(j) ~= i
        mislabeled_points = [mislabeled_points; data_points{1,i}(j,:)]; % Store the point if the predicted class
                                                                    % does not match the  actual class
    end
    end    
end

score1 = [];
score2 = [];
score3 = [];

contour_points = []; % Required for contourf function to operate properly
multiplier = 1;

discriminant_line_points = [];
prev_predicted_class = I(1);

for i = 1 : numel(X1)
    score1 = [score1 ; calculate_score([X1(i) X2(i)], data_points{2,1}, data_points{3,1}, data_points{4,1})];
    score2 = [score2 ; calculate_score([X1(i) X2(i)], data_points{2,2}, data_points{3,2}, data_points{4,2})];
    score3 = [score3 ; calculate_score([X1(i) X2(i)], data_points{2,3}, data_points{3,3}, data_points{4,3})];
    scores = [score1 score2 score3];
    [M, I] = max(scores,[],2);
    
    if mod(i,mesh_number) == 0
        contour_points = [contour_points, I((multiplier-1) * mesh_number + 1 : (multiplier) * mesh_number)]; % fill contour point matrix, final size is mesh_number by mesh_number
        multiplier = multiplier + 1;
    end
    
%         predicted_class = I(i);
%         if predicted_class ~= prev_predicted_class
%             discriminant_line_points = [discriminant_line_points ; [X1(i), X2(i)]];
%         end
%         prev_predicted_class = predicted_class;
    
    if mod(i,mesh_number) == 0      % Since this for loop is the most time consuming loop, progress is shown here
        prog = i/numel(X1) * 100;
        fprintf('Progress: %g%%\n',prog);
    end
end

mymap = [0.9 0.65 0.65 % Create color map for countourf() function
    0.65 0.9 0.65
    0.65 0.9 0.65
    0.65 0.65 0.9];

figure
hold on

xlim([-6 6]); % Determine axis limits
ylim([-6 6]);
xlabel('x1'); % Write axis labels
ylabel('x2');

[M,c] = contourf(X1, X2, contour_points,3); % Create contour on a mesh
c.LineWidth = 1.5; % Countour line width
colormap(mymap); % Set colors in accordance with classes

plot(mislabeled_points(:,1), mislabeled_points(:,2), 'ko','MarkerSize',10); % Circle mislabeled data points
plot(red_points(:,1),red_points(:,2),'r.','MarkerSize',15); % Red class points
plot(green_points(:,1),green_points(:,2),'g.','MarkerSize',15); % Green class points
plot(blue_points(:,1),blue_points(:,2),'b.','MarkerSize',15); % Blue class points
%plot(discriminant_line_points(:,1), discriminant_line_points(:,2),'k.','MarkerSize',1)


function means = estimate_mean(X)
N_data_points = size(X,1);
d_features = size(X, 2);

means = [];
for column = 1 : d_features
    sum = 0;
    for row = 1 : N_data_points
        sum = sum + X(row, column);
    end
    means = [means sum/N_data_points];
end
end
function covs = estimate_cov(X)
N_data_points = size(X,1);
d_features = size(X, 2);

means = estimate_mean(X);

covs = [];
for i = 1 : d_features
    cov_row = [];
    for j = 1 : d_features
        sum = 0;
        for row = 1 : N_data_points
            sum = sum + ((X(row,i) - means(1, i)) * (X(row, j) - means(1, j)));
        end
        S_ij = sum / N_data_points;
        cov_row = [cov_row S_ij];
    end
    covs = [covs; cov_row];
end
end
function priors = estimate_prior(a, b, c)
total_count = numel(a)/2 + numel(b)/2 + numel(c)/2;

priors = [numel(a)/2/total_count numel(b)/2/total_count numel(c)/2/total_count];
end
function score = calculate_score(X, mean, cov, prior)
N_data_points = size(X,1);
score = [];
for i = 1 : N_data_points
    d_features = size(X, 2);
    first_term = sqrt( 1 / ( ((2*pi)^d_features) * det(cov)));
    second_term = exp(-0.5 * (X(i,:) - mean) * inv(cov) * transpose((X(i,:) - mean)));
    normal_dist = first_term * second_term;
    score = [score ;log(normal_dist * prior)];
end
end