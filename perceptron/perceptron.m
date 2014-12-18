function [perc_correct] = perceptron(Inputs,Targets,eta,n_epoch)

	w_max = 0.1; 	% initial weights in range [-w_max,w_max]
	n_inst = size(Inputs,1);
	n_in = size(Inputs,2);

	% concatenate a column for bias node:
	Inputs = [ones(n_inst,1) Inputs];
	Outputs = zeros(size(Targets));

	% initialize random weights:
	W = (2*w_max).*rand(n_in+1,1)-w_max;
	
	% train for specified epochs:
	for i = 1:n_epoch
		correct = 0;
		% iterate over each training instance:
		for d = 1:n_inst
			x = Inputs(d,:);
			% threshold the output:
			Outputs(d) = sign(x*W);
			if Outputs(d) == 0
				Outputs(d) = -1;
			end
			% update weights:
			dW = eta.*(Targets(d)-Outputs(d)).*x;
			W = W+dW';
			
			if Outputs(d) == Targets(d)
				correct = correct+1;
			end
		end
	end
	perc_correct = correct/n_inst*100;
end
