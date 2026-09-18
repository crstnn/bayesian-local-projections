function [b,u,sigu2,se,x_hat] = iv_fcn(y,x,z)
% James Morley
% Updated 4 August 2026
% Calculates IV estimates, residuals, and standard errors
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%z        Instruments
%
%
%% OUTPUTS
%b        IV point estimates
%u        residuals based on IV estimates
%sigu2    residual variance
%se       IV SEs
%x_hat    fitted x
    

b = (z'*x)\(z'*y); 
u = y-x*b;
x_hat=z*((z'*z)\(z'*x));
sigu2 = (u'*u)/(length(u)-length(b));   
se = sqrt(diag(sigu2*inv(x_hat'*x_hat)));
