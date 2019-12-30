function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




size(y);

s = ones(m,1);
X = [s X];
h1 = sigmoid(X* Theta1');
n = size(h1,1);
a1 = X;
n = ones(n,1);
h1 = [n h1];

z2 = (a1*Theta1');

size(X);
size(Theta1);

size(h1);
size(Theta2);
h = sigmoid(h1 * Theta2');
a2 = h1;
z3 = (h1*Theta2');
a3 = h;

realy = zeros(size(y,1),num_labels);

for i= 1:size(y,1);
  b = zeros(1,num_labels);
  b(y(i)) = 1;
  realy(i,:) = b;
endfor
size(realy);
size(h);

J = (1/m)*sum(sum(-realy.*log(h)-(1-realy).*log((1-h))));

% size(sum(sum(Theta1.*Theta1)))
Theta1(:,1) = 0;
Theta2(:,1) = 0;
J+= (lambda/(2*m))*(sum(sum(Theta1.*Theta1))+sum(sum(Theta2.*Theta2)));
% size(J);
% break

delta_3 = zeros(size(a3));
size(delta_3);
size(a3.-y);

% MAKE THE Y WORK ON THE INPUTS
sizey = ones(size(y,1),1);
numlabez = 1:num_labels;
realy2 = sizey * numlabez;
realy2 = realy2 == y;


delta3 = (a3.-realy2);
size(delta3);
% size(Theta2)
% size(sigmoidGradient(z2))
houop = ones(size(z2,1),1);

z2 = [houop z2];
size(z2);
delta2 = (delta3 * Theta2).*sigmoidGradient(z2);
delta2 = delta2(:,2:end);

% UNENECESARY CAUSE MATRIX MULTI
% D2= 0;
% for i = 1:size(delta3,1);
%   D2 = D2 + delta3(i,:)'*a2(i,:);
% endfor
% size(D2);

D2 = delta3'*a2;

% UNECESSARY CAUSE MATRIX MULTI
% D1 = 0;
% for i = 1:size(delta2,1);
%   D1 = D1 + delta2(i,:)'*a1(i,:);
%   % D1
% endfor

D1 = delta2'*a1;
size(D1)
D1
Theta1_grad = (1/m * D1);
Theta2_grad = (1/m) * D2;

size(D1);
size(D2);
% newtheta1 = Theta1(:,1) .*= 0;
% newtheta2 = Theta2(:,1) .*=0;
% D1
newtheta1 = Theta1;
newtheta2 = Theta2;
% newtheta1 = newtheta1(:,2:end);
% size(newtheta1)
newtheta1(:,1).*=0;
newtheta2(:,1).*=0;
% Dsom = D1(1,:end) = 0;
% size(Dsom)
Theta1_grad+= (lambda/m)*newtheta1;
Theta2_grad+= (lambda/m)*newtheta2;
% % size(delta2)

% y = [1;3;2]
% b =   rand(3,3)
% [max p] = max(b,[],2)
% p' == y





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
