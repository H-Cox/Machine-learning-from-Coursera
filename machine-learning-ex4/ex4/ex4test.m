
clear

load('ex4data1.mat');
load('ex4weights.mat');
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10;  
lambda = 1;
m = size(X, 1);

Y = zeros(max(y),m);

% find co-ordinates for nn output and fill output matrix
ycoord = [y';linspace(1,m,m)];
Y(sub2ind([max(y) m],ycoord(1,:),ycoord(2,:))) = 1;
% size(Y) = 10 5000

% add bias to input
a1 = [ones(m, 1) X];

% need to go through each layer calculating next activations
z2 = Theta1*a1';

a2 = sigmoid(z2)';
a2 = [ones(size(a2,1),1) a2];

z3 = Theta2*a2';

a3 = sigmoid(z3)';

h = a3;

% size(h) = 5000 10

J = (-Y'.*log(h)-(1-Y').*log(1-h));

J = sum(sum(J))/m;
R = (sum(sum(Theta1.^2))+sum(sum(Theta2.^2))).*lambda/(2*m);


d3 = h'-Y;

d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);

Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

Del1 = Del1 + d2*a1;
Del2 = Del2 + d3*a2;

Theta1_grad = Del1/m;
Theta2_grad = Del2/m;

R1 = Theta1.*lambda./m;
R2 = Theta2.*lambda./m;

R1(:,1) = 0;
R2(:,1) = 0;

Theta1_grad = Theta1_grad + R1;
Theta2_grad = Theta2_grad + R2;

