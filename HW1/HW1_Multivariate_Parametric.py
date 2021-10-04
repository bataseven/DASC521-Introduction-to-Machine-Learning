import math
import matplotlib.pyplot as plt
import numpy as np
import time


def mean_estimate(data_points):
    d_features = len(data_points[0])
    N_data_points = len(data_points)
    means = []
    for i in range(d_features):
        mean = sum(data_points[:, i]) / N_data_points
        means.append(mean)

    return np.array(means)


def cov_estimate(data_points):
    N_data_points = len(data_points)
    d_features = len(data_points[0])
    means = mean_estimate(data_points)
    Cov = []

    for r in range(d_features):
        cov_row = []
        for c in range(d_features):
            S_ij = sum( ((data_points[:, r] - means[r]) * (data_points[:, c] - means[c])) ) / N_data_points
            cov_row.append(S_ij)
        Cov.append(cov_row)

    return np.array(Cov)


def prior_estimate(multiple_class_data_points):
    no_of_points = 0
    priors = []
    for i in range(len(multiple_class_data_points)):
        no_of_points += len(multiple_class_data_points[i])
    for i in range(len(multiple_class_data_points)):
        priors.append(len(multiple_class_data_points[i]) / no_of_points)

    return np.array(priors)


def calculate_score(data_points, mean, cov, prior):
    g_score_array = []
    data_count = len(data_points)


    for i in range(data_count):
        print(data_points[i])
        difference = data_points[i] - mean
        first_term = np.sqrt(1 / (2 * math.pi * np.linalg.det(cov)))
        second_term = np.e ** (-0.5 * np.matmul(np.matmul(difference, np.linalg.inv(cov)), difference.T))
        class_conditional = first_term * second_term
        g_score = np.log(class_conditional * prior)
        g_score_array.append(g_score)
    # print(np.array(g_score_array))
    return g_score_array


def calculate_discriminant(data_points, mean, cov, prior):
    g_i_array = []
    for k in range(len(data_points)):
        x = data_points
        W_i = -0.5 * np.linalg.inv(cov)
        w_i = np.matmul(np.linalg.inv(cov), mean)
        w_i0 = -0.5 * np.matmul(mean.T, np.linalg.inv(cov)) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)
        g_i = (np.matmul(np.matmul(x.T, W_i), x)) + np.matmul(w_i.T, x) + w_i0
        g_i_array.append(g_i)
    # (np.array(g_i_array).shape)
    return np.array(g_i_array)


def create_disc_line(means, covs, priors):
    disc = []
    number_of_mesh = 10
    x1_interval = np.linspace(-6, 6, number_of_mesh)
    x2_interval = np.linspace(-6, 6, number_of_mesh)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    for i in range(number_of_mesh):
        predicted_class = -1
        for j in range(number_of_mesh):
            point_to_check = np.array([x1_grid[i, j], x2_grid[i, j]])
            point_to_check = point_to_check.reshape(point_to_check.shape[0], -1)
            # print(point_to_check)
            score_array = [0] * 3
            score_array[0] = calculate_score(point_to_check, means[0], covs[0], priors[0])
            score_array[1] = calculate_score(point_to_check, means[1], covs[1], priors[1])
            score_array[2] = calculate_score(point_to_check, means[2], covs[2], priors[2])
            # print(score_array)
            if predicted_class == -1:
                predicted_class = score_array.index(max(score_array))
            else:
                old_predicted_class = predicted_class
                predicted_class = score_array.index(max(score_array))
                if old_predicted_class != predicted_class:
                    disc.append(point_to_check)
            # print(predicted_class)
    return disc


np.random.seed(101)

class_means = np.array([[0, 2.5], [-2.5, -2], [2.5, -2]])
class_deviations = np.array([[[3.2, 0.0], [0.0, 1.2]], [[1.2, -0.8], [-0.8, 1.2]], [[1.2, 0.8], [0.8, 1.2]]])
class_sizes = np.array([120, 90, 90])

points = []

for i in range(3):
    points.append(np.array(np.random.multivariate_normal(class_means[i], class_deviations[i], class_sizes[i])))

mean_estimates = [mean_estimate(points[0]), mean_estimate(points[1]), mean_estimate(points[2])]
cov_estimates = [cov_estimate(points[0]), cov_estimate(points[1]), cov_estimate(points[1])]
class_priors = prior_estimate(points)

confusion_matrix = np.zeros((3, 3))

for i in range(3):
    scores = np.zeros((3, len(points[i])))
    scores[0] = calculate_score(points[i], mean_estimates[0], cov_estimates[0], class_priors[0])
    scores[1] = calculate_score(points[i], mean_estimates[1], cov_estimates[1], class_priors[1])
    scores[2] = calculate_score(points[i], mean_estimates[2], cov_estimates[2], class_priors[2])
    scores = scores.argmax(axis=0)
    confusion_matrix[i][0] = (scores == 0).sum()
    confusion_matrix[i][1] = (scores == 1).sum()
    confusion_matrix[i][2] = (scores == 2).sum()
print(confusion_matrix)
# print(calculate_discriminant(points[0], mean_estimates[0], cov_estimates[0], class_priors[0]))

# mesh = np.zeros((len(x1_interval), len(x2_interval), 2))
#
# discriminant = np.zeros((1, 2))
# prev_mesh_score = np.zeros(2, )
# for x1 in range(len(x1_interval)):
#     for x2 in range(len(x2_interval)):
#         mesh[x1][x2] = [x1_interval[x1], x2_interval[x2]]
#         mesh_score = np.zeros((3, 2))
#         # mesh_score[0] = calculate_score(mesh[x1][x2], mean_estimates[0], cov_estimates[0], class_priors[0])
#         # mesh_score[1] = calculate_score(mesh[x1][x2], mean_estimates[1], cov_estimates[1], class_priors[1])
#         # mesh_score[2] = calculate_score(mesh[x1][x2], mean_estimates[2], cov_estimates[2], class_priors[2])


plt.figure(figsize=(6, 6))
# discs = create_disc_line(mean_estimates, cov_estimates, class_priors)
# zip_array = list(zip(*discs))
# x1_array = list(x[0] for x in zip_array[0])
# x2_array = list(x[0] for x in zip_array[1])

# plt.plot(x1_array, x2_array, 'ko')
# plt.plot(discriminant[:, 1], discriminant[:, 0], 'ko', markersize=3)
# plt.plot(mesh[:, :, 0], mesh[:, :, 1], 'ko', markersize=5)
plt.plot(points[0][:, 0], points[0][:, 1], 'ro')
plt.plot(points[1][:, 0], points[1][:, 1], 'go')
plt.plot(points[2][:, 0], points[2][:, 1], 'bo')
plt.ylim([-6, 6])
plt.xlim([-6, 6])
plt.show()
