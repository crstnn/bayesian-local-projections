function [b,u,sigu2,se] = ols_fcn(y,x)
% James Morley
% Updated 31 July 2026
% Calculates OLS estimates, residuals, and standard errors
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%
%
%% OUTPUTS
%b        OLS point estimates
%u        OLS residuals
%sigu2    residual variance
%se       OLS SEs
    
b = (x'*x)\(x'*y); 
u = y-x*b;
sigu2 = (u'*u)/(size(u,1)-size(b,1));
se = (sqrt(diag(sigu2))*sqrt(diag(inv(x'*x)))')';
