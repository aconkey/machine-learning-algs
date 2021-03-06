function [err] = gd_stoch(Inputs,Targets,eta,n_epoch)

	w_max = 0.1; 	% weights initialized in [-w_max,w_max]
	n_inst = size(Inputs,1);
	n_in = size(Inputs,2);

	% concatenate a column for bias node:
	Inputs = [ones(n_inst,1) Inputs];
	Outputs = zeros(size(Targets));

	% initialize small random weights
	W = (2*w_max).*rand(n_in+1,1)-w_max;

	% train for specified epochs:
	for i = 1:n_epoch
		err = 0;
		% iterate over each training instance
		for d = 1:n_inst
			x = Inputs(d,:);
			Outputs(d) = x*W;
			W = W+eta.*(Targets(d)-Outputs(d)).*x';
			err = err+0.5.*(Targets(d)-Outputs(d)).^2;
		end
		
		% shuffle data:
		p = randperm(n_inst);
		Inputs = Inputs(p,:);
		Targets = Targets(p,:);
	end
end
