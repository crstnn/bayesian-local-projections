function corr_out = serial_correlation_fcn(tmpm1,order,mn_out)
% James Morley
% Updated 31 July 2026
% Calculates serial correlation for a given horizon
%
%% INPUTS
%
%tmpm1       data to consider serial correlation in
%order       order of serial correlation to consider
%mn_out      mean of data
%
%% OUTPUTS
%sort_out       moments (mean, std dev, median, 0.025, 0.975, correlations b/w draws) for simulated draws

var_out=0;
cov_out=0;

nnn=length(tmpm1);
          
jjj=order+1;
while jjj<=nnn
cov_out=cov_out+(tmpm1(jjj)-mn_out)*(tmpm1(jjj-order)-mn_out);
var_out=var_out+(tmpm1(jjj)-mn_out)*(tmpm1(jjj)-mn_out);
jjj=jjj+1;
end

corr_out=cov_out./var_out;  %calculates nth order serial correlation
          
