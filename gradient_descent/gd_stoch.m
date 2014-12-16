function [] = gd_stoch(Input,Target,eta,n_epoch)

w_max = 0.1;
n_inst = size(Input,1);
n_in = size(Input,2);

% concatenate a column for bias node:
Input = [ones(n_inst,1) Input];

W = (2*w_max).*rand(n_in,1)-w_max;

% train for specified epochs:
for i = 1:n_epoch
	% iterate over each training instance
	for d = 1:n_inst
		x = Input(d,:);
		out = x*W;
		W = W + eta.*(Target(d,:)-out).*x;
	end
end
