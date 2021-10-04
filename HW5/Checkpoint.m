% Introduction to Machine Learning - HW5: Decision Tree Regression
% Written by Berke Ataseven (54326)
close all
clear all
clc
%% Read data from file
data_set = readmatrix('hw05_data_set.csv','HeaderLines',1);

%% Divide the data into training and test
training_set = data_set(1:100,:);
test_set = data_set(101:end,:);
%% Plot the data points
Pruning_Parameter = 15;
figure('Position', [180 280 800 300])
hold on
plot(training_set(:,1),training_set(:,2),'c.','MarkerSize',15);
plot(test_set(:,1),test_set(:,2),'r.','MarkerSize',15);
xlim([-1 61]);
ylim([-150 100]);
title(strcat("P = ",num2str(Pruning_Parameter)));
xlabel('x');
ylabel('y');
%% Calculate and plot the decision tree
for i = 2 : 2 : 50
[decision_x, decision_y] = create_decision_tree(training_set,i);
h = plot(decision_x, decision_y ,'-k','LineWidth',1);
title(strcat("P = ",num2str(i)));
hold on
drawnow
pause(0.1)
delete(h)
fprintf('RMSE is %g when P is %g\n', calculate_RMSE([decision_x decision_y], test_set), i);
end

[decision_x, decision_y] = create_decision_tree(training_set,Pruning_Parameter);
h = plot(decision_x, decision_y ,'-k','LineWidth',1);
title(strcat("P = ",num2str(Pruning_Parameter)));
hold on
%% Calculate and draw (RMSE vs Pre-Pruning Size)
P = 5 : 5 : 50;
RMSE = [];
for i = 1 : numel(P)
[decision_x, decision_y] = create_decision_tree(training_set,P(i));
RMSE = [RMSE calculate_RMSE([decision_x decision_y], test_set)];
end
figure('Position', [180 280 800 300])
plot(P,RMSE,'*-k','LineWidth',1,'MarkerSize',5);
ylabel('RMSE');
xlabel('Pre-pruning size(P)');
%% Functions
function [x_values, y_hat] = create_decision_tree(X, P)
x_values = [];
y_hat = [];
X = sortrows(X);
dx = 0.01;

x = X(:,1);
y = X(:,2);

splits = [x(1)-dx sort(find_splits(X, P)) x(end)+dx];

for i = 1 : length(splits) - 1
    x_value = (splits(i) : dx : splits(i+1) - dx)';
    x_values = [x_values; x_value];
    y_in_range = y(find(x >= splits(i) & x < splits(i+1)));
    y_hat_in_range = mean(y_in_range);
    y_hat = [y_hat; ones(length(x_value),1)*y_hat_in_range];
end
end
function splits = find_splits(X, P)
X = sortrows(X);
N_Parent = size(X,1);
x = X(:,1);
y = X(:,2);
splits = [];

Split_Errors = [];
Possible_Splits = [];

for i = 1 : N_Parent - 1
    if x(i) == x(i+1)
        continue;
    else
        Split = (x(i) + x(i+1)) / 2;
    end
    
    index_left = find(x < Split);
    index_right = find(x > Split);
    
    average_left = sum(y(index_left)) / numel(index_left);
    average_right = sum(y(index_right)) / numel(index_right);
    
    RMSE_left = sum((y(index_left) - average_left) .^ 2);
    RMSE_right = sum((y(index_right) - average_right) .^ 2);
    
    Split_Error = (numel(index_left) * RMSE_left + numel(index_right) * RMSE_right) / N_Parent;
    Split_Errors = [Split_Errors; Split_Error];
    Possible_Splits = [Possible_Splits; Split];
end
[~, I] = min(Split_Errors);
Best_Split = Possible_Splits(I);

splits = [splits Best_Split];

left_y_values = y(find(x < Best_Split));
left_x_values = x(find(x < Best_Split));
left_X = [left_x_values left_y_values];

right_y_values = y(find(x > Best_Split));
right_x_values = x(find(x > Best_Split));
right_X = [right_x_values right_y_values];

if numel(left_y_values) > P
    splits = [splits find_splits(left_X, P)];
end

if numel(right_y_values) > P
    splits = [splits find_splits(right_X, P)];
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