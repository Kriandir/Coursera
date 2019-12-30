function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
x = ones(m,1);
X = [x X];

% X =
% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


sig1 = sigmoid(Theta1*X');
n = size(sig1,2);
sigsig = ones(n,1);
% realsig1 = sig1';
sig1 = [sigsig sig1'];
% realsig1 = [sigsig realsig1];
size(X)
size(Theta1)
size(sig1)

sig2 = sigmoid(sig1*Theta2');
size(sig2)
[max p] = max(sig2,[],2);





% =========================================================================


end
