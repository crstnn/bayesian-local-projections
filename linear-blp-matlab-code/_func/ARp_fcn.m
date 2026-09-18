function [sige2,resids] = ARp_fcn(endogenous_variables,p,dummies1,dummies2,ld)
% James Morley
% Updated 25 July 2026
% AR(p) estimation of sig_ei^2 for Minnesota priors on lagged coefficients
% or residuals for serially-correlated instrument
%
%% INPUTS
%
%endogenous_variables     Endogenous variables to project
%
%
%% OUTPUTS
%sige2                   Vector of OLS residual variances from AR(4)models


T = length(endogenous_variables);
N = size(endogenous_variables,2); %number of endogenous variables
p_max=p;

sige2=[];
resids=[];
i=1;        %indicator for LHS variable@  
while i<=N

p = p_max; %reset back to original p if changed for long differences    

if ld == 1
    p=p_max+1;
y=endogenous_variables(p+1:T,i)-endogenous_variables(p:T-1,i);
else
y=endogenous_variables(p+1:T,i);
end
    
x=ones(T-p,1);   %constant
    j=1;
    while j<=p_max
        if ld == 1
        x=[x,endogenous_variables(p+1-j:T-j,i)-endogenous_variables(p-j:T-1-j,i)];  %lags of variable
        else
        x=[x,endogenous_variables(p+1-j:T-j,i)];  %lags of variable
        end
    j=j+1;
    end    

if sum(sum(dummies1))>0 && i==1
    x=[x,dummies1(p+1:T,:)];
end

if sum(sum(dummies2))>0 && i==2
    x=[x,dummies2(p+1:T,:)];
end

[~,u,sigei2,~] = ols_fcn(y,x);  

sige2=[sige2;sigei2];
resid=[zeros(p,1);u];
resids=[resids,resid];
 
i=i+1;   
end