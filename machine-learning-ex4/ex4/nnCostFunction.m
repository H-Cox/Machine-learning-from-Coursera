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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

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

Y = zeros(max(y),m);

% find co-ordinates for nn output and fill output matrix
ycoord = [y';linspace(1,m,m)];
Y(sub2ind([max(y) m],ycoord(1,:),ycoord(2,:))) = 1;

% add bias to input
a1 = [ones(m, 1) X];

% need to go through each layer calculating next activations
z2 = Theta1*a1';

a2 = sigmoid(z2)';
a2 = [ones(size(a2,1),1) a2];

z3 = Theta2*a2';

a3 = sigmoid(z3)';

% finally we have the output
h = a3;

% calculate cose elementwise and then sum over all samples and features
J = (-Y'.*log(h)-(1-Y').*log(1-h));
J = sum(sum(J))/m;

% calculate regularization term, removing the bias terms added
R = (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))).*lambda/(2*m);
J = J+R;

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

% find delta_3 and delta_2
d3 = h'-Y;
d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);

d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);

Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

Del1 = Del1 + d2*a1;
Del2 = Del2 + d3*a2;

Theta1_grad = Del1/m;
Theta2_grad = Del2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

R1 = Theta1.*lambda./m;
R2 = Theta2.*lambda./m;

R1(:,1) = 0;
R2(:,1) = 0;

Theta1_grad = Theta1_grad + R1;
Theta2_grad = Theta2_grad + R2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
