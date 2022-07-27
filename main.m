clear;
dir;

%Load the training data and display a random 10x10 matrix of handwritten digits
load('ex4data1.mat');
m = size(X,1);

%Randomly select 100 data points to display
sel = randperm(size(X,1));
sel = sel(1:100);
displayData(X(sel,:));

%remember that the digit 0 is mapped onto an index of 10
%already trained network parameters trained and stored in 'ex4weights.mat'
%lets load the weights into variables 'Theta1' and 'Theta2'
load('ex4weights.mat');

input_layer_size = 400;%20x20 input images of Digits
hidden_layer_size = 25;%25 hidden layers
num_labels = 10;%10 labels, one each for the digits 1 to 10

%Unroll parameters:
nn_params = [Theta1(:) ; Theta2(:)];

%Weight regularization parameters (we set it to zero here)
lambda = 0;
J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda);

fprintf('Cost at parameters (loaded from ex4weights) : %f\n',J);


%Weight regularization parameter (we set this to 1 here)
lambda = 1;

J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda);
fprintf('Cost at parameters (loaded from ex4weights): %f\n',J);


%Random Initialization
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);
% Also output the costFunction debugging value 
% This value should be about 0.576051
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at (fixed) debugging parameters (w/ lambda = 3): %f\n', debug_J);


options = optimset('MaxIter', 50);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
 
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);