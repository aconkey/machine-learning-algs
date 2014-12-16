function [] = bp(Inputs,Targets,n_hid,n_out,eta,n_epoch)

w_max = 0.1
n_inst = size(Inputs,1);
n_in = size(inputs,2);

% concatenate a column for bias node:
Inputs = [ones(n_inst,1) Inputs];

W_1 = (2*w_max).*rand(n_in,n_hid)-w_max; 	% w_ij in [-w_max,w_max] connects i to j
W_2 = (2*w_max).*rand(n_hid+1,n_out)-w_max; % +1 for bias node

% train for specified epochs:
for i = 1:n_epoch
	% iterate over each training instance
	for d = 1:n_inst
		Act_hid = logsig(Inputs(d,:)*W_1);
		Act_out = logsig([1 Act_hid]*W_2);	% bias node concatenated
		W
	end
end
