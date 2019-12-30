function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% size(X)
% size(Theta)
h = X * Theta';
a = sum((h(R==1)-Y(R==1)).**2);
b = sum(sum((h.*R - Y.*R).**2));
J =  (1/2)*b;

nastytheta=  sum((h.*R)-(Y.*R));
nastyx = sum((h.*R)'-(Y.*R)');
% idx = find(R(1,:)==1)
size(R);
% R'
% find(R'(:,i)==1)
for i = 1:size(X)(1)
  idx = find(R(i,:)==1);
  theta_temp = Theta(idx,:);
  Ytemp = Y(i,idx);
  theta_temp;
  X(i,: );
  X_grad(i,:) = (X(i,:)*theta_temp'-Ytemp)*theta_temp+(lambda*X(i,:));
endfor

% now for the other you gotta get the movie scores
for i = 1:size(Theta)(1)
  idx = find(R(:,i)==1);
  % get the values for the features from x since that is the column we will get from this call
  x_temp = X(idx,:);
  % get the y values but be sure to flip because y = R in users*movies but now we are interested in movies
  Ytemp = Y(idx,i)';
  x_temp;

  Theta(i,:);
  Theta_grad(i,:) = (Theta(i,:)*x_temp'-Ytemp)*x_temp+(lambda*Theta(i,:));
endfor
Theta_grad;
X_grad;
J+= ((lambda/2)*sum(sum(Theta.**2)))+((lambda/2)*sum(sum(X.**2)));
% size(nastytheta)
% size(nastyx)
% size(X)
% size(Theta)
% size(h)

% fprintf("hoi")
% if a==b
  % fprintf("YES")
% fprintf("end")
% size(R)
% size(X)
% size(Theta)
% Theta(:,2)
% Theta
% size(X_grad)
% Theta(1,:)
% size(Theta)
% Theta
% size(X)
% Theta(:,1)
% X_grads

% for i = 1:size(X)(1)
%   i
%   g = zeros(1,(size(X)(2)));
%
%
%   for j = 1:size(X)(2)
%       b = sum(((h.*R)(i,:)(j)-(Y.*R)(i,:)(j))*Theta(:,j));
%       % X_grad((i,:)(j)) = b
%       % X_grad
%       g(j) = b;
%   endfor
%   X_grad(i,:) = g;
%
% endfor
% size(Theta)(1)
% size(X)(1)
% size(Theta)
% size(h)
% for i = 1:size(Theta)(1)
%   % i
%   g = zeros(1,(size(Theta)(2)));
%   g;
% % size(X)
% % size(Theta)
%   for j = 1:size(Theta)(2)
%       b = sum(((h.*R)'(i,:)(j)-(Y.*R)'(i,:)(j))*X(:,j));
%       % X_grad((i,:)(j)) = b
%       % X_grad
%
%       g(j) = b;
%
%   endfor
%
%   % g
%
%   Theta_grad(i,:) = g;
%
%   % i
% endfor

% Theta_grad = (Theta) .* nastytheta';
% size(Theta_grad);
% X_grad = (X) .* nastyx';
% Theta_grad
% X_grad
% Theta_grad = Theta_grad(:)
% X_grad = X_grad(:)
% size(X_grad)
% size(Theta_grad(:))

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
