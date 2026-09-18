function [sort_out] = moments_fcn(fnl_mat,c_level)
% James Morley
% Updated 31 July 2026
% Calculates moments given simulated draws
%
%% INPUTS
%
%fnl_mat        simulated draws
%c_level        confidence level (e.g., 95%) for equal-tailed bands
%
%
%% OUTPUTS
%sort_out       moments (mean, std dev, median, (1-c_level)/2, 1-(1-c_level)/2, correlations b/w draws) for simulated draws


sort_out=[];

i_ndx=1;
while i_ndx <= size(fnl_mat,2)
          
          tmpm1=fnl_mat(:,i_ndx);
          mn_out=mean(tmpm1)';
          std_out=std(tmpm1)';
          med_out=median(tmpm1)';

          corr_out1=serial_correlation_fcn(tmpm1,1,mn_out);
          corr_out5=serial_correlation_fcn(tmpm1,5,mn_out);
          corr_out10=serial_correlation_fcn(tmpm1,10,mn_out);
          corr_out20=serial_correlation_fcn(tmpm1,20,mn_out);

          tmpm2=sort(tmpm1);
          up_0=tmpm2(ceil((1-(1-c_level)/2)*length(tmpm1)));
          low_0=tmpm2(ceil(((1-c_level)/2)*length(tmpm1)));

          sort_out=[sort_out;
          mn_out,std_out,med_out,low_0,up_0,corr_out1,corr_out5,corr_out10,corr_out20];
         
i_ndx=i_ndx+1;
end
