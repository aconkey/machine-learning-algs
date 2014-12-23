function [err] = bp_stoch(Inputs,Targets,n_hid,n_out,eta,n_epoch)

	w_max = 0.1
	n_inst = size(Inputs,1);
	n_in = size(Inputs,2);


	% concatenate a column for bias node:
	Inputs = [ones(n_inst,1) Inputs];
	Outputs = zeros(size(Targets)); % temporary

	W_hid = (2*w_max).*rand(n_in+1,n_hid)-w_max; 	% w_ij in [-w_max,w_max] connects i to j
	W_out = (2*w_max).*rand(n_hid+1,n_out)-w_max; 	% +1 for bias node

	% train for specified epochs:
	for i = 1:n_epoch
		err = 0;
		% iterate over each training instance
		for d = 1:n_inst
			x = Inputs(d,:);
			Act_hid = logsig(x*W_hid);  
			Act_out = logsig([1 Act_hid]*W_out);  % +1 for bias
			Delta_out = Act_out.*(1.-Act_out).*(Targets(d)-Act_out);
			Delta_hid = Act_hid.*(1.-Act_hid).*(W_out(2:end)*Delta_out)';
			W_hid = W_hid+(eta.*x'*Delta_hid);
			W_out = W_out+(eta.*Delta_out.*[1 Act_hid]'); % +1 for bias
		
        		Outputs(d) = Act_out; % temporary
			% compute error:
			for k = 1:n_out
				err = err+0.5.*(Targets(d,k)-Act_out(k)).^2;
			end
		end
	end
end

