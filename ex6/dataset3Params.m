function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% try more values in practice
%C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
%sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];

% try less values to speed up submission
C_vec = [0.01 0.1 1 10];
sigma_vec = [0.01 0.1 1 10];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

err = zeros(length(C_vec), length(sigma_vec));
for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        kernel_function = @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j));
        model = svmTrain(X, y, C_vec(i), kernel_function);
        prediction = svmPredict(model, Xval);
        err(i, j) = mean(double(prediction ~= yval));
    end
end

[M, I] = min(err(:));
[I_row, I_col] = ind2sub(size(err), I);

C = C_vec(I_row);
sigma = sigma_vec(I_col);

% =========================================================================

end
