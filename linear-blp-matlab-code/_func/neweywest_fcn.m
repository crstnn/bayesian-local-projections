function [se_nw] = neweywest_fcn(x,b,u,h)
% James Morley
% Updated 31 July 2026
% Calculates Newey West standard errors given OLS output
%
%% INPUTS
%
%x        RHS variables
%b        OLS estimates
%u        OLS residuals
%h        truncation horizon
%
%
%% OUTPUTS
%se_nw    Newey West standard errors


emat=[];
k=1;
while k <= length(b)
        emat=[emat;u'];
k=k+1;
end

hhat=emat.*x';
G=zeros(length(b),length(b)); w=zeros(2*(h+1)+1,1);
a=0;

while a <= h+1
    ga=zeros(length(b),length(b));
    w(h+1+1+a,1)=(h+1+1-a)/(h+1+1);
    za=hhat(:,(a+1):length(u))*hhat(:,1:length(u)-a)';
    if a==0
        ga=ga+za;
    else
        ga=ga+za+za';
    end
    G=G+w(h+1+1+a,1)*ga;
a=a+1;
end    

V=inv(x'*x)*G*inv(x'*x);
se_nw=sqrt(diag(V));