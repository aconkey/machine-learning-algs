function [perc_correct] = perceptron(Inputs,Targets,eta,n_epoch)

	w_max = 0.1; 	% initial weights in range [-w_max,w_max]
	n_inst = size(Inputs,1);
	n_in = size(Inputs,2);

	% concatenate a column for bias node:
	Inputs = [ones(n_inst,1) Inputs];

	% initialize random weights:
	W = (2*w_max).*rand(n_in,1)-w_max;
	dW = zeros(size(W));
	
	% train for specified epochs:
	for i = 1:n_epoch
		correct = 0;
		% iterate over each training instance:
		for d = 1:n_inst
			x = Inputs(d,:);
			targ_d = Targets(d,:);
			% threshold the output:
			out_d = sign(x*W);
			if (out_d == 0)
				out_d = -1;
			% update weights:
			dW = eta.*(targ_d-out_d).*x;
			W = W+dW;
			
			if (out_d == targ_d)
				correct = correct+1;
		end
	end
	perc_correct = correct/n_inst
end
