function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


q = [0.01 0.03 0.1 0.3 1 3 10 30];
e = 0;
k = 2;

for i = 1:size(q,2)
  temp = q(i);
  for j = 1:size(q,2)
    stemp = q(j);
    model= svmTrain(X, y, temp, @(x1, x2) gaussianKernel(x1, x2, stemp)); 
    predictions = svmPredict(model, Xval);
    etemp = mean(double(predictions ~= yval));

    if(k == 2)
      k = 3;
      e = etemp;
      C = temp;
      sigma = stemp;
    end
    
    if(etemp < e)
      e = etemp;
      C = temp;
      sigma = stemp;
    end   
    
  end
end




% =========================================================================

end
