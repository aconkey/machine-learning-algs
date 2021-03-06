function [err] = gd(Inputs,Targets,eta,n_epoch)

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
		dW = zeros(size(W));
		err = 0;
		% iterate over each training instance:
		for d = 1:n_inst
			x = Inputs(d,:);
			Outputs(d) = x*W;
			dW = dW+eta.*(Targets(d)-Outputs(d)).*x';
			err = err+0.5.*(Targets(d)-Outputs(d)).^2;
		end
		W = W+dW;
		
		% shuffle data:
		p = randperm(n_inst);
		Inputs = Inputs(p,:);
		Targets = Targets(p,:);
	end
end
