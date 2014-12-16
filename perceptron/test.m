function [perc_correct] = test_perceptron(filename, n_in)
	data = csvread(filename);
	Inputs = data(:,:n_in);
	Targets = data(:,n_in+1:);
	perc_correct = perceptron(Inputs,Targets,0.1,100)
end
