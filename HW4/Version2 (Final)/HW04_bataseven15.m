% Introduction to Machine Learning - HW4: Nonparametric Regression
% Written by Berke Ataseven (54326)
close all
clear all
clc
%% Read data from file
data_set = readmatrix('hw04_data_set.csv','HeaderLines',1);

%% Divide the data into training and test
training_set = data_set(1:100,:);
test_set = data_set(101:end,:);

%% Plot the data points and regressogram
figure
hold on
plot(training_set(:,1),training_set(:,2),'c.','MarkerSize',15);
plot(test_set(:,1),test_set(:,2),'r.','MarkerSize',15);
xlim([-1 61]);
ylim([-150 100]);
title('Regressogram (h=3)');
xlabel('x');
ylabel('y');

% Calculate regressogram of the given set for the given bin width
bin_width = 3; % Bin width (h)
regressogram = calculate_regressogram(training_set, bin_width); % Calculate regressogram
plot(regressogram(:,1),regressogram(:,2),'k-','LineWidth',1);
legend({'training','test','Regressogram'});
%% Calculate and print RMSE of regressogram for the test set
fprintf('Regressogram => RMSE is %2.4f when h is %1.2f\n\n',calculate_RMSE(regressogram,test_set), bin_width);

%% Plot the data points and running mean smoother
figure
hold on
plot(training_set(:,1),training_set(:,2),'c.','MarkerSize',15);
plot(test_set(:,1),test_set(:,2),'r.','MarkerSize',15);
xlim([-1 61]);
ylim([-150 100]);
title('Running Mean Smoother (h=3)');
xlabel('x');
ylabel('y');

% Calculate running mean smoother of the given set for the given bin width
bin_width = 3; % Bin width (h)
mean_smoother = calculate_mean_smoother(training_set, bin_width); % Calculate running mean smoother
plot(mean_smoother(:,1),mean_smoother(:,2),'k','LineWidth',1);
legend({'training','test','Running Mean Smoother'});
%% Calculate and print RMSE of mean smoother for the test set
fprintf('Running Mean Smoother => RMSE is %2.4f when h is %1.2f\n\n',calculate_RMSE(mean_smoother,test_set), bin_width);

%% Plot the data points and kernel smoother
figure
hold on
plot(training_set(:,1),training_set(:,2),'c.','MarkerSize',15);
plot(test_set(:,1),test_set(:,2),'r.','MarkerSize',15);
xlim([-1 61]);
ylim([-150 100]);
title('Kernel Smoother (h=1)');
xlabel('x');
ylabel('y');

% Calculate kernel smoother of the given set for the given bin width
bin_width = 1; % Bin width (h)
kernel_smoother = calculate_kernel_smoother(training_set, bin_width); % Calculate kernel smoother
plot(kernel_smoother(:,1),kernel_smoother(:,2),'k','LineWidth',1);
legend({'training','test','Kernel Smoother'});

%% Calculate and print RMSE for the test set
fprintf('Kernel Smoother => RMSE is %2.4f when h is %1.2f\n\n',calculate_RMSE(kernel_smoother, test_set), bin_width);

%% Functions
function g_hat = calculate_kernel_smoother(X, h) % Returns estimated g_hat. X is the data set, h is the bin width
g_hat = [];
x_axis = linspace(min(X(:,1)) - 0.1, max(X(:,1)) + 0.1, 10000); % Create 10000 points on the x-axis to to make the calculation for
N = length(X); % Number of points in the data set
for i = 1 : length(x_axis) % Calculate for 1000 points on the x-axis between [0,60]
    sum_num = 0; % Set the summation to zero for a new point in the x-axis
    sum_denom = 0; % Set the summation to zero for a new point in the x-axis
    for j = 1 : N % Calculate for all points in the data set
        K = exp( -((x_axis(i) - X(j)) / (h)) ^ 2 / 2) / sqrt(2 * pi); % Gaussian density formula
        sum_num = sum_num + (K * X(j,2)); % Summation
        sum_denom = sum_denom + K;      % Summation
    end
    if sum_num == 0 || sum_denom == 0 % Append 0 if no element in the bin
        g_hat = [g_hat; x_axis(i) 0];
    else
        
        g_hat = [g_hat; [x_axis(i) sum_num/sum_denom]]; % Append the new point into kernel smoother curve
    end
end
end
function g_hat = calculate_mean_smoother(X, h) % Returns estimated g_hat. X is the data set, h is the bin width
g_hat = [];
x_axis = linspace(min(X(:,1)) - 0.1, max(X(:,1)) + 0.1, 10000); % Create 50000 points on the x-axis to to make the calculation for
N = length(X); % Number of points in the data set
for i = 1 : length(x_axis) % Calculate for 1000 points on the x-axis between [0,60]
    sum_num = 0; % Set the summation to zero for a new point in the x-axis
    sum_denom = 0; % Set the summation to zero for a new point in the x-axis
    for j = 1 : N % Calculate for all points in the data set
        if abs( (x_axis(i) - X(j)) / (h/2) ) < 1 % Define the bin symmetric around X (Textbook formula takes h as the bin radius)
            W = 1;
        else
            W = 0;
        end
        sum_num = sum_num + (W * X(j,2)); % Summation
        sum_denom = sum_denom + W; % Summation
    end
    if sum_num == 0 || sum_denom == 0 % Append 0 if no element in the bin
        g_hat = [g_hat; x_axis(i) 0];
    else
        g_hat = [g_hat; [x_axis(i) sum_num/sum_denom]]; % Append the new point into running mean smoother curve
    end
end
end
function g_hat = calculate_regressogram(X, h) % Returns estimated g_hat. X is the data set, h is the bin width
g_hat=[];
origin = 0; % Origin is defined as zero
min_value = origin;
max_value = 59;

left_borders = min_value : h : max_value; % Array of the left borders
right_borders = min_value + h : h : max_value + h; % Array of the right borders
x_values = X(:,1); % X points of the data set
y_values = X(:,2); % Y points of the data set

for i = 1 : length(left_borders) % Iterate as much as there are bins
    
    element_index = find(x_values > left_borders(i) & x_values <= right_borders(i)); % Indexes of the element within the bin borders
    element_count_in_a_bin = numel(element_index); % Number of elements in a bin
    
    if element_count_in_a_bin ~= 0 % To not divide by zero
        average = (sum(y_values(element_index)) / element_count_in_a_bin) * ones(1000, 1);
    else % If there is no element in the bin set the average to zero
        average = zeros(1000, 1);
    end
    line_of_the_bin = [linspace(left_borders(i), right_borders(i),1000)' average]; % Create the points that draw the lines of the regressogram
    g_hat = [g_hat; line_of_the_bin]; % Append the average lines into the regressogram
end
end
function RMSE = calculate_RMSE(trained_curve, X) % Returns RMSE, X is the data set to calculate the RMSE for
summ = 0;
N = length(X); % Number of data points in the data set
for i = 1 : N % Iterate for all the data points in the data set
    [~ , idx] = min(abs(X(i,1) - trained_curve(:,1))); % Index of the point on the x-axis whose value is the closest to Xj
    summ = summ + (X(i,2) - trained_curve(idx,2)) ^ 2; % Summation of the RMSE
end
RMSE = sqrt(summ / N);
end