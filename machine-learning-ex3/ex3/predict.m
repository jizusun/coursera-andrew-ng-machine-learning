function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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

% Theta1: 25*401
% a_1: m * 401
a_1 = [ones(m,1), X];


z_2 = a_1 * Theta1';  % m * 25  
a_2 = sigmoid(z_2);
m_2 = size(a_2, 1);
a_2 = [ones(m_2, 1), a_2];  % m * 26

% Theta2 has size 10 x 26
z_3 = a_2 * Theta2'; % m * 10
h_theta_x = sigmoid(z_3); % m * 10

[probability, p] = max(h_theta_x, [], 2);


% =========================================================================


end
