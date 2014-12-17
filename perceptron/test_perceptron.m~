clear
clc

data = csvread('datasets/AND.csv'); n_in = 2;
%data = csvread('datasets/XOR.csv'); n_in = 2;

Inputs = data(:,1:n_in);
Targets = data(:,n_in+1:size(data,2));
perc_correct = perceptron(Inputs,Targets,0.05,100)
